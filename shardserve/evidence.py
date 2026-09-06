import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from .weights import digest_file
from .batch import digest


def source():
    package=Path(__file__).resolve().parent
    paths=sorted(package.rglob('*.py'))
    return digest({str(p.relative_to(package.parent)):digest_file(p) for p in paths})


def command(args):
    try:
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'command':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc: return {'command':args,'unavailable':str(exc)}


def environment():
    import torch
    versions={}
    for name in ('torch','transformers','safetensors','huggingface-hub','triton','mlflow'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return dict(timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        command=sys.argv,source_digest=source(),python=sys.version,platform=platform.platform(),versions=versions,
        cuda_available=torch.cuda.is_available(),cuda_count=torch.cuda.device_count(),torch_cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        devices=[{'rank':i,'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        runtime_identity={key:os.environ.get(key) for key in ('DATABRICKS_RUNTIME_VERSION','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')},
        inventory=command(['nvidia-smi','-q']),topology=command(['nvidia-smi','topo','-m']),git=command(['git','rev-parse','HEAD']))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def upload(directory,experiment):
    directory=Path(directory)
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run() as run:
            mlflow.set_tag('shardserve.source_digest',source())
            manifest=directory/'run.json'
            if manifest.exists(): mlflow.set_tag('shardserve.run_id',json.loads(manifest.read_text())['run_id'])
            mlflow.log_artifacts(str(directory))
            write(directory/'tracking-upload.json',{'status':'uploaded','mlflow_run_id':run.info.run_id})
    except Exception as exc:
        write(directory/'tracking-upload.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','retry':['python','-m','shardserve','upload',str(directory),'--experiment',experiment]})
        raise
