#!/usr/bin/env python3
"""Move small sealed shards between the Databricks data plane and Modal."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time


def command(*args,capture=False):
    return subprocess.run(args,check=True,text=True,capture_output=capture)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory')
    parser.add_argument('--job-id',type=int,required=True)
    parser.add_argument('--profile',default='shardserve')
    parser.add_argument('--results-table',default='workspace.shardserve.results')
    args=parser.parse_args()
    if not args.run_directory.startswith('/Volumes/'):
        parser.error('run_directory must be a Unity Catalog Volume path')

    remote='dbfs:'+args.run_directory.rstrip('/')
    with tempfile.TemporaryDirectory(prefix='shardserve-relay-') as temporary:
        local=Path(temporary)
        command('databricks','fs','cp',remote+'/run.json',str(local/'run.json'),'-p',args.profile)
        manifest=json.loads((local/'run.json').read_text())
        if args.run_directory.rstrip('/').split('/')[-1]!=manifest['run_id']:
            raise ValueError('remote directory does not match run identity')
        for shard in manifest['shards']:
            if Path(shard['path']).name!=shard['path']: raise ValueError('invalid shard path')
            command('databricks','fs','cp',remote+'/'+shard['path'],str(local/shard['path']),'-p',args.profile)
        command('modal','run','deploy/modal/worker.py','--run-dir',str(local))
        command('databricks','fs','mkdir',remote+'/staging','-p',args.profile)
        for path in sorted((local/'staging').iterdir()):
            command('databricks','fs','cp',str(path),remote+'/staging/'+path.name,'-p',args.profile,'--overwrite')
        command('databricks','fs','cp',str(local/'external-hardware.json'),remote+'/external-hardware.json','-p',args.profile,'--overwrite')

        request={
            'job_id':args.job_id,
            'job_parameters':{
                'operation':'finalize','run_directory':args.run_directory,
                'results_table':args.results_table,
            },
        }
        request_file=local/'finalize.json'
        request_file.write_text(json.dumps(request))
        run=json.loads(command('databricks','jobs','run-now','--json','@'+str(request_file),'--no-wait','-p',args.profile,'-o','json',capture=True).stdout)
        deadline=time.monotonic()+1000
        while time.monotonic()<deadline:
            detail=json.loads(command('databricks','jobs','get-run',str(run['run_id']),'-p',args.profile,'-o','json',capture=True).stdout)
            if detail['state']['life_cycle_state'] in ('TERMINATED','SKIPPED','INTERNAL_ERROR'):
                break
            time.sleep(5)
        else:
            raise TimeoutError('Databricks finalize job did not terminate')
        if detail['state'].get('result_state')!='SUCCESS':
            raise RuntimeError(f"Databricks finalize failed: {detail['state'].get('state_message','unknown error')}")
        task_id=detail['tasks'][0]['run_id']
        output=json.loads(command('databricks','jobs','get-run-output',str(task_id),'-p',args.profile,'-o','json',capture=True).stdout)
        result=json.loads(output['notebook_output']['result'])
        print(json.dumps({'run_id':manifest['run_id'],'finalize_job_run_id':run['run_id'],'result':result}))


if __name__=='__main__': main()
