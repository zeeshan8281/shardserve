import threading
import time
from .runtime import Group
from .scheduler import Scheduler, validate_request, TERMINAL

class Engine:
    def __init__(self, root, config):
        from transformers import AutoTokenizer
        self.config=config
        self.group=Group(root,config)
        self.tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        import json
        from pathlib import Path
        self.vocab_size=json.loads((Path(root)/'config.json').read_text())['vocab_size']
        if len(self.tokenizer)>self.vocab_size:
            self.group.close(force=True)
            raise ValueError('tokenizer exceeds model vocabulary')
        # Instruct generation stops on im_end and endoftext, matching generation_config.
        generation=json.loads((Path(root)/'generation_config.json').read_text())
        eos=generation['eos_token_id']
        self.scheduler=Scheduler(config,eos if isinstance(eos,list) else [eos])
        self.lock=threading.Condition()
        self.stopping=False; self.error=None; self.last={}
        self.worker=threading.Thread(target=self.run,daemon=True)
        self.worker.start()

    def submit(self, row):
        r=validate_request(row,self.tokenizer,self.config,self.vocab_size)
        with self.lock:
            if self.error or self.stopping:
                raise RuntimeError(self.error or 'engine stopped')
            self.scheduler.submit(r); self.lock.notify_all()
        return r

    def cancel(self, key):
        with self.lock:
            r=self.scheduler.requests.get(key)
            if r is None: raise KeyError(key)
            r.cancelled=True; self.lock.notify_all()

    def events(self, request):
        with self.lock:
            if request.consumed:
                raise ValueError('stream already claimed')
            request.consumed=True
        while True:
            with self.lock:
                if request.events:
                    event=request.events.popleft()
                elif request.state in TERMINAL:
                    event={'seq':len(request.output)+1,'terminal':request.result()}
                else:
                    self.lock.wait(0.1); continue
            yield event
            if 'terminal' in event:
                return

    def run(self):
        try:
            while True:
                with self.lock:
                    if self.stopping: break
                    active=any(r.state not in TERMINAL for r in self.scheduler.requests.values())
                    if not active and not self.scheduler.blocks.tables:
                        self.lock.wait(0.1); continue
                    plan=self.scheduler.plan()
                result=self.group.step(plan)
                with self.lock:
                    self.scheduler.acknowledge(result['samples']); self.last=result
                    self.lock.notify_all()
        except Exception as exc:
            self.group.close(force=True)
            with self.lock:
                self.error=f'{type(exc).__name__}: {exc}'
                self.scheduler.fail('replica_failed'); self.lock.notify_all()

    def health(self):
        with self.lock:
            return {'ready':not self.error and not self.stopping,'error':self.error,
                'waiting':sum(r.state=='waiting' for r in self.scheduler.requests.values()),
                'running':sum(r.state=='running' for r in self.scheduler.requests.values()),
                'rank_startup':self.group.info,'last_iteration':self.last}

    def close(self):
        with self.lock:
            self.stopping=True; self.lock.notify_all()
        self.worker.join(self.config.watchdog+5)
        self.group.close(force=self.worker.is_alive())
        with self.lock:
            self.scheduler.fail('engine_shutdown'); self.lock.notify_all()
