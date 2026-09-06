import json
from pathlib import Path
from uuid import uuid4

import modal


app=modal.App('shardserve-data-plane-worker')
models=modal.Volume.from_name('shardserve-models',create_if_missing=True)
image=modal.Image.from_id('im-L7ze4ET7ZLcnqJvqXoZ6GP').add_local_python_source('shardserve')


@app.function(image=image,gpu='L4:2',cpu=4,memory=32768,volumes={'/models':models},timeout=1800)
def infer(manifest,rows,attempt_id):
    import subprocess
    from shardserve.batch import digest
    from shardserve.config import Config
    from shardserve.engine import Engine
    from shardserve.evidence import source
    from shardserve.weights import fetch

    if source()!=manifest['identity']['code_revision']:
        raise ValueError('worker source differs from prepared run')
    model=fetch('/models')
    models.commit()
    engine=Engine(model,Config(**manifest['identity']['runtime']))
    records=[]
    try:
        for row in rows:
            internal=dict(row,request_id=f'{digest(row)[:32]}-{attempt_id}')
            request=engine.submit(internal)
            for event in engine.events(request):
                if 'terminal' in event: result=event['terminal']
            result.update(
                request_id=row['request_id'],run_id=manifest['run_id'],attempt_id=attempt_id,
                output_text=engine.tokenizer.decode(result['output_token_ids']),
                error_code=None if result['terminal_status']=='completed' else result['reason'],
                provenance=manifest['identity'],
            )
            records.append(result)
    finally:
        engine.close()
    hardware=subprocess.run(
        ['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv,noheader'],
        capture_output=True,text=True,check=True,
    ).stdout.strip().splitlines()
    return {'records':records,'hardware':hardware}


@app.local_entrypoint()
def main(run_dir:str):
    from shardserve import batch
    from shardserve.evidence import write

    directory=Path(run_dir)
    manifest,rows=batch.read_run(directory)
    attempt_id='01-modal-'+uuid4().hex[:20]
    result=infer.remote(manifest,rows,attempt_id)
    batch.stage(directory,manifest,result['records'],attempt_id)
    write(directory/'external-hardware.json',{
        'provider':'Modal','requested':'L4:2','reserved_gpu_count':2,
        'rank_mapping':[0,1],'cpu_cores':4,'memory_mib':32768,'gpus':result['hardware'],
    })
    print(json.dumps({'run_id':manifest['run_id'],'attempt_id':attempt_id,'rows':len(rows),'hardware':result['hardware']}))
