"""Classic single-node job entrypoints. Spark is imported only by this adapter."""
import argparse
import json
from pathlib import Path
import re
from . import batch
from .config import Config
from .evidence import source,write,environment


def _table(value):
    if not re.fullmatch(r'[A-Za-z_][\w]*(\.[A-Za-z_][\w]*){0,2}', value):
        raise ValueError('invalid table identifier')
    return value


def names(catalog='workspace', schema='shardserve'):
    if not all(re.fullmatch(r'[A-Za-z_][\w]*', value) for value in (catalog, schema)):
        raise ValueError('invalid catalog or schema identifier')
    return {
        'input_table': f'{catalog}.{schema}.inputs',
        'results_table': f'{catalog}.{schema}.results',
        'volume': f'/Volumes/{catalog}/{schema}/artifacts',
        'experiment': '/Shared/ShardServe',
    }


def bootstrap(spark, catalog='workspace', schema='shardserve'):
    plane=names(catalog, schema)
    spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}')
    spark.sql(f'CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.artifacts')
    spark.sql(f"CREATE TABLE IF NOT EXISTS {plane['input_table']} (request_id STRING, token_ids ARRAY<INT>, max_new_tokens INT) USING DELTA")
    batch.DeltaResults(spark, plane['results_table'])
    return plane


def status(spark, catalog='workspace', schema='shardserve'):
    plane=names(catalog, schema)
    plane['input_rows']=spark.table(plane['input_table']).count()
    results=spark.table(plane['results_table'])
    plane['result_rows']=results.count()
    plane['runs']=results.select('run_id').distinct().count()
    return plane


def prepare_snapshot(spark, input_table, version, hardware, directory, config=None, max_attempts=2):
    input_table=_table(input_table)
    frame=spark.read.option('versionAsOf',version).table(input_table)
    from pyspark.sql import functions as F
    if frame.groupBy('request_id').count().filter(F.col('count')>1).limit(1).count(): raise ValueError('duplicate input IDs')
    # ponytail: bounded 100k-input job, partitioned manifest indexing if larger jobs are needed.
    if frame.limit(100001).count()>100000: raise ValueError('v1 input bound is 100000 rows')
    rows=[r.asDict(recursive=True) for r in frame.select('request_id','token_ids','max_new_tokens').toLocalIterator()]
    cfg=config or Config().dict()
    return batch.prepare(rows,{'table':input_table,'version':version},cfg,source(),hardware,directory,max_attempts)


def commit(spark, directory, results_table):
    sink=batch.DeltaResults(spark,_table(results_table))
    with batch.committer(directory):
        return {'attempt_rows_considered':batch.commit_staged(directory,sink)}


def validate(spark, directory, results_table):
    return batch.validate_results(directory,batch.DeltaResults(spark,_table(results_table)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=['bootstrap','status','prepare','run','resume','commit','validate'])
    p.add_argument('--directory')
    p.add_argument('--catalog',default='workspace'); p.add_argument('--schema',default='shardserve')
    p.add_argument('--input-table'); p.add_argument('--version',type=int)
    p.add_argument('--results-table'); p.add_argument('--model'); p.add_argument('--hardware')
    p.add_argument('--config'); p.add_argument('--max-attempts',type=int,default=2)
    args=p.parse_args()
    if args.operation not in ('bootstrap','status') and not args.directory: p.error(f'{args.operation} needs --directory')
    if args.operation=='prepare':
        if args.version is None or not args.input_table or not args.hardware: p.error('prepare needs --input-table, --version and --hardware')
    elif args.operation not in ('bootstrap','status'):
        if not args.results_table: p.error(f'{args.operation} needs --results-table')
        if args.operation in ('run','resume') and not args.model: p.error(f'{args.operation} needs --model')
    from pyspark.sql import SparkSession
    spark=SparkSession.builder.getOrCreate()
    if args.operation=='bootstrap': print(json.dumps(bootstrap(spark,args.catalog,args.schema))); return
    if args.operation=='status': print(json.dumps(status(spark,args.catalog,args.schema))); return
    if args.operation=='prepare':
        cfg=json.loads(Path(args.config).read_text()) if args.config else Config().dict()
        directory=prepare_snapshot(spark,args.input_table,args.version,json.loads(Path(args.hardware).read_text()),args.directory,cfg,args.max_attempts)
        print(directory); return
    if args.operation in ('run','resume'):
        sink=batch.DeltaResults(spark,args.results_table)
        manifest,_=batch.read_run(args.directory)
        if manifest['identity']['code_revision']!=source(): raise ValueError('source changed: prepare a new run')
        write(Path(args.directory)/'environment.json',environment())
        from .engine import Engine
        result=batch.execute(args.directory,sink,lambda:Engine(args.model,Config(**manifest['identity']['runtime'])))
    elif args.operation=='commit':
        result=commit(spark,args.directory,args.results_table)
    else: result=validate(spark,args.directory,args.results_table)
    write(Path(args.directory)/'durable-validation.json',result); print(json.dumps(result))

if __name__=='__main__': main()
