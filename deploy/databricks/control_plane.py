# Databricks notebook source
# MAGIC %pip install --no-deps /Workspace/Shared/shardserve-data-plane/shardserve-0.2.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %pip install mlflow==3.3.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

from pyspark.sql import SparkSession
from shardserve.databricks import _table, bootstrap, commit, names, prepare_snapshot, status, validate
from shardserve.evidence import upload, write


def parameter(name, default=''):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)


spark=SparkSession.builder.getOrCreate()
operation=parameter('operation','status')
catalog=parameter('catalog','workspace')
schema=parameter('schema','shardserve')
plane=names(catalog,schema)

if operation=='bootstrap':
    result=bootstrap(spark,catalog,schema)
elif operation=='status':
    result=status(spark,catalog,schema)
elif operation=='prepare':
    input_table=_table(parameter('input_table',plane['input_table']))
    version=parameter('input_version')
    if not version:
        version=str(spark.sql(f'DESCRIBE HISTORY {input_table}').first()['version'])
    hardware=parameter('hardware_json')
    if not hardware: raise ValueError('prepare requires hardware_json for the external GPU allocation')
    config=parameter('config_json')
    directory=prepare_snapshot(
        spark,input_table,int(version),json.loads(hardware),
        parameter('volume_root',plane['volume']+'/runs'),
        json.loads(config) if config else None,
        int(parameter('max_attempts','2')),
    )
    result={'directory':str(directory),'input_table':input_table,'input_version':int(version)}
elif operation=='commit':
    result=commit(spark,parameter('run_directory'),parameter('results_table',plane['results_table']))
elif operation=='validate':
    result=validate(spark,parameter('run_directory'),parameter('results_table',plane['results_table']))
elif operation=='upload':
    directory=parameter('run_directory')
    upload(directory,parameter('experiment',plane['experiment']))
    result=json.loads(open(directory+'/tracking-upload.json').read())
elif operation=='finalize':
    directory=parameter('run_directory')
    result={
        'commit':commit(spark,directory,parameter('results_table',plane['results_table'])),
        'validation':validate(spark,directory,parameter('results_table',plane['results_table'])),
    }
    upload(directory,parameter('experiment',plane['experiment']))
    result['tracking']=json.loads(open(directory+'/tracking-upload.json').read())
    write(directory+'/data-plane-finalize.json',result)
else:
    raise ValueError(f'unknown operation: {operation}')

dbutils.notebook.exit(json.dumps(result,sort_keys=True))
