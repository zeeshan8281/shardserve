"""Deterministic FIFO admission and decode-first packed iterations, coordinator only."""
from collections import deque
from dataclasses import dataclass, field
import math
import re
import time
from .cache import Blocks

TERMINAL = {'completed','cancelled','failed','timed_out'}


def validate_request(row, tokenizer, config, vocab):
    if not isinstance(row,dict) or set(row)-{'request_id','prompt','token_ids','max_new_tokens','deadline'}:
        raise ValueError('unknown request/sampling fields')
    key = row.get('request_id')
    if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',key):
        raise ValueError('invalid request ID')
    if ('prompt' in row) == ('token_ids' in row):
        raise ValueError('provide exactly one prompt representation')
    n = row.get('max_new_tokens')
    if type(n) is not int or not 1 <= n <= config.context:
        raise ValueError('positive bounded output limit required')
    if 'prompt' in row:
        if not isinstance(row['prompt'],str) or not row['prompt'].strip():
            raise ValueError('empty/invalid text prompt')
        if tokenizer is None:
            raise ValueError('text requires pinned tokenizer')
        ids = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True)
    else:
        ids = row['token_ids']
    if not isinstance(ids,list) or not ids or any(type(t) is not int or not 0 <= t < vocab for t in ids):
        raise ValueError('invalid token IDs')
    if len(ids)+n > config.context:
        raise ValueError('context limit exceeded')
    deadline = row.get('deadline')
    if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline <= 0):
        raise ValueError('deadline must be a finite Unix timestamp')
    return Request(key,list(ids),n,deadline)

@dataclass
class Request:
    key: str
    prompt: list
    limit: int
    deadline: object = None
    state: str = 'waiting'
    reason: object = None
    cursor: int = 0
    output: list = field(default_factory=list)
    events: deque = field(default_factory=deque)
    times: list = field(default_factory=list)
    submitted: float = field(default_factory=time.time)
    finished: object = None
    cancelled: bool = False
    consumed: bool = False

    def finish(self, state, reason):
        if self.state not in TERMINAL:
            self.state,self.reason,self.finished = state,reason,time.time()

    def result(self):
        return dict(request_id=self.key,terminal_status=self.state,reason=self.reason,
            output_token_ids=list(self.output),input_tokens=len(self.prompt),output_tokens=len(self.output),
            timing={'submitted':self.submitted,'tokens':list(self.times),'finished':self.finished})

class Scheduler:
    def __init__(self, config, eos):
        self.config,self.eos = config,set(eos)
        self.blocks = Blocks(config.kv_blocks)
        self.requests = {}
        self.seen = set()
        self.iteration = 0
        self.inflight = None

    def submit(self, request):
        # Retain bounded tombstones until group restart: retries cannot create second streams.
        if request.key in self.seen:
            raise ValueError('duplicate request ID; choose a new ID')
        if len(self.seen) >= 100000:
            raise OverflowError('ID ledger full; restart idle replica')
        if sum(r.state=='waiting' for r in self.requests.values()) >= self.config.queue_size:
            raise OverflowError('waiting queue full')
        if math.ceil((len(request.prompt)+request.limit)/16) > self.blocks.count:
            raise ValueError('request exceeds replica KV capacity')
        self.requests[request.key] = request
        self.seen.add(request.key)

    def plan(self):
        if self.inflight is not None:
            raise RuntimeError('previous iteration not acknowledged')
        release = []
        now = time.time()
        for r in self.requests.values():
            if r.state not in TERMINAL:
                if r.cancelled:
                    r.finish('cancelled','cancelled')
                elif r.deadline is not None and now >= r.deadline:
                    r.finish('timed_out','deadline')
                elif len(r.events) >= self.config.output_buffer:
                    r.finish('failed','output_backpressure')
            if r.state in TERMINAL and r.key in self.blocks.tables:
                self.blocks.release(r.key)
                release.append(r.key)
        # Retain only ID tombstones; connected consumers own their terminal buffers.
        self.requests = {key:r for key,r in self.requests.items() if r.state not in TERMINAL}
        reserve = {}
        live = sum(r.state=='running' for r in self.requests.values())
        for r in self.requests.values():
            if r.state!='waiting':
                continue
            n = len(r.prompt)+r.limit
            if live >= self.config.max_live or math.ceil(n/16)>len(self.blocks.free):
                break  # FIFO head waits for enough space; existing decodes continue.
            self.blocks.reserve({r.key:n})
            reserve[r.key] = n
            r.state = 'running'
            live += 1
        running = [r for r in self.requests.values() if r.state=='running']
        running.sort(key=lambda r:r.cursor<len(r.prompt))
        rows,offset,budget = [],0,self.config.token_budget
        for r in running:
            if budget<=0:
                break
            if r.cursor<len(r.prompt):
                tokens = r.prompt[r.cursor:r.cursor+min(budget,self.config.prefill_chunk)]
                sample = r.cursor+len(tokens)==len(r.prompt)
            else:
                tokens,sample = [r.output[-1]],True
            rows.append(dict(request_id=r.key,tokens=tokens,context=r.cursor,positions=list(range(r.cursor,r.cursor+len(tokens))),offset=offset,query_length=len(tokens),sample=sample))
            offset+=len(tokens)
            budget-=len(tokens)
        self.iteration+=1
        plan = dict(iteration=self.iteration,reserve=reserve,release=release,rows=rows,
            mode='decode' if rows and all(r['query_length']==1 and r['context']>0 for r in rows) else 'eager')
        self.inflight = plan
        return plan

    def acknowledge(self, samples):
        if self.inflight is None:
            raise RuntimeError('no iteration to acknowledge')
        expected = {row['request_id'] for row in self.inflight['rows'] if row['sample']}
        if set(samples)!=expected:
            raise RuntimeError('rank sample agreement mismatch')
        for row in self.inflight['rows']:
            r = self.requests[row['request_id']]
            r.cursor += row['query_length']
            # Cancellation/deadline arriving inside a collective applies after it completes.
            if r.cancelled:
                r.finish('cancelled','cancelled')
            elif r.deadline is not None and time.time()>=r.deadline:
                r.finish('timed_out','deadline')
            elif row['sample']:
                token = samples[r.key]
                r.output.append(token)
                r.times.append(time.time())
                r.events.append({'seq':len(r.output),'token_id':token})
                if token in self.eos or len(r.output)>=r.limit:
                    r.finish('completed','eos' if token in self.eos else 'length')
        self.inflight = None

    def fail(self, reason):
        for r in self.requests.values():
            r.finish('failed',reason)
            if r.key in self.blocks.tables:
                self.blocks.release(r.key)
        self.inflight=None
