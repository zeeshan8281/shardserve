"""Classic single-node job entrypoints. Spark is imported only by this adapter."""
import argparse
import json
from pathlib import Path
from . import batch
from .config import Config
from .evidence import source,write,environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=['prepare','run','resume','commit','validate'])
    p.add_argument('--directory',required=True)
    p.add_argument('--input-table'); p.add_argument('--version',type=int)
    p.add_argument('--results-table'); p.add_argument('--model'); p.add_argument('--hardware')
    p.add_argument('--config'); p.add_argument('--max-attempts',type=int,default=2)
    args=p.parse_args()
    if args.operation=='prepare':
        if args.version is None or not args.input_table or not args.hardware: p.error('prepare needs --input-table, --version and --hardware')
    else:
        if not args.results_table: p.error(f'{args.operation} needs --results-table')
        if args.operation in ('run','resume') and not args.model: p.error(f'{args.operation} needs --model')
    from pyspark.sql import SparkSession
    spark=SparkSession.builder.getOrCreate()
    if args.operation=='prepare':
        frame=spark.read.option('versionAsOf',args.version).table(args.input_table)
        from pyspark.sql import functions as F
        if frame.groupBy('request_id').count().filter(F.col('count')>1).limit(1).count(): raise ValueError('duplicate input IDs')
        # ponytail: bounded 100k-input job, partitioned manifest indexing if larger jobs are needed.
        if frame.limit(100001).count()>100000: raise ValueError('v1 input bound is 100000 rows')
        rows=[r.asDict(recursive=True) for r in frame.select('request_id','token_ids','max_new_tokens').toLocalIterator()]
        cfg=json.loads(Path(args.config).read_text()) if args.config else Config().dict()
        directory=batch.prepare(rows,{'table':args.input_table,'version':args.version},cfg,source(),json.loads(Path(args.hardware).read_text()),args.directory,args.max_attempts)
        print(directory); return
    sink=batch.DeltaResults(spark,args.results_table)
    if args.operation in ('run','resume'):
        manifest,_=batch.read_run(args.directory)
        if manifest['identity']['code_revision']!=source(): raise ValueError('source changed: prepare a new run')
        write(Path(args.directory)/'environment.json',environment())
        from .engine import Engine
        result=batch.execute(args.directory,sink,lambda:Engine(args.model,Config(**manifest['identity']['runtime'])))
    elif args.operation=='commit':
        with batch.committer(args.directory): result={'attempt_rows_considered':batch.commit_staged(args.directory,sink)}
    else: result=batch.validate_results(args.directory,sink)
    write(Path(args.directory)/'durable-validation.json',result); print(json.dumps(result))

if __name__=='__main__': main()
