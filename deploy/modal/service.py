"""Public Modal web service for the authenticated retrieval-aware API."""
import os
import subprocess
import sys

import modal


app=modal.App('shardserve-ir')
models=modal.Volume.from_name('shardserve-models',create_if_missing=True)
image=modal.Image.from_id('im-L7ze4ET7ZLcnqJvqXoZ6GP').add_local_python_source('shardserve')
service_secret=modal.Secret.from_name('shardserve-service')


@app.function(
    image=image,gpu='L4:2',cpu=4,memory=32768,volumes={'/models':models},
    secrets=[service_secret],timeout=86400,scaledown_window=300,
)
@modal.concurrent(max_inputs=72)
@modal.web_server(8080,startup_timeout=900)
def api():
    from shardserve.weights import fetch
    model=fetch('/models'); models.commit()
    subprocess.Popen([
        sys.executable,'-m','shardserve','serve','--model',model,'--world-size','2',
        '--host','0.0.0.0','--port','8080','--elastic',
    ],env=os.environ.copy())
