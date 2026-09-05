"""Bounded HTTP load, deterministic token workloads and least-inflight TP1 dispatch."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import math
import os
from pathlib import Path
import random
import threading
import time
import urllib.request
from .batch import digest
from .config import MODEL,REVISION
from .evidence import environment,write


def workload(prompt_tokens=128,output_tokens=32,count=100,seed=9281,repeated=False):
    if prompt_tokens<1 or output_tokens<1 or prompt_tokens+output_tokens>4096 or not 1<=count<=10000:
        raise ValueError('invalid bounded workload')
    rng=random.Random(seed)
    prefix=[rng.randrange(100,150000) for _ in range(prompt_tokens//2)] if repeated else []
    rows=[dict(request_id=f'r{i:06d}',token_ids=prefix+[rng.randrange(100,150000) for _ in range(prompt_tokens-len(prefix))],max_new_tokens=output_tokens) for i in range(count)]
    return dict(seed=seed,repeated_prefix=repeated,digest=digest(rows),rows=rows,model=MODEL,revision=REVISION)


def percentile(values,q):
    if not values: return None
    x=sorted(values); index=(len(x)-1)*q; low=math.floor(index); high=math.ceil(index)
    return x[low]+(x[high]-x[low])*(index-low)


def stream(url,row,backend):
    if backend=='custom':
        route='/stream'; body=row
    else:
        # vLLM 0.10.1 logprobs exposes one token entry per generated token.
        route='/v1/completions'
        body=dict(model=MODEL,prompt=row['token_ids'],max_tokens=row['max_new_tokens'],temperature=0,
            stream=True,logprobs=1,seed=0,ignore_eos=False)
    headers={'Content-Type':'application/json'}
    token=os.environ.get('SHARDSERVE_API_TOKEN')
    if token: headers['Authorization']='Bearer '+token
    req=urllib.request.Request(url.rstrip('/')+route,data=json.dumps(body).encode(),headers=headers)
    started=time.perf_counter(); timestamps=[]; terminal=None; generated=[]; last_seq=0
    with urllib.request.urlopen(req,timeout=120) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            payload=line[5:].strip()
            if payload==b'[DONE]': break
            event=json.loads(payload)
            now=time.perf_counter()
            if backend=='custom':
                if event['seq']!=last_seq+1 or terminal is not None: raise ValueError('unordered/duplicate SSE event')
                last_seq=event['seq']
                if 'token_id' in event:
                    generated.append(event['token_id']); timestamps.append(now)
                else: terminal=event['terminal']
            else:
                for choice in event.get('choices',[]):
                    lp=choice.get('logprobs') or {}
                    count=len(lp.get('tokens',[]))
                    timestamps.extend([now]*count)
                    if choice.get('finish_reason'):
                        terminal={'terminal_status':'completed','reason':choice['finish_reason']}
    finished=time.perf_counter()
    if terminal is None: raise ValueError('stream ended without terminal result')
    return dict(started=started,finished=finished,token_timestamps=timestamps,terminal=terminal,output_tokens=len(timestamps),output_token_ids=generated if backend=='custom' else None,
        ttft=timestamps[0]-started if timestamps else None,tpot=(timestamps[-1]-timestamps[0])/(len(timestamps)-1) if len(timestamps)>1 else None,
        latency=finished-started)


def summarize(records,duration,ttft_slo,tpot_slo):
    good=[r for r in records if r.get('terminal',{}).get('terminal_status')=='completed']
    slo=[r for r in good if r['ttft'] is not None and r['ttft']<=ttft_slo and (r['tpot'] is None or r['tpot']<=tpot_slo)]
    out=dict(requests=len(records),successes=len(good),failures=len(records)-len(good),duration_seconds=duration,
        useful_output_tokens=sum(r['output_tokens'] for r in good),output_tokens_per_second=sum(r['output_tokens'] for r in good)/duration,
        slo_goodput_requests_per_second=len(slo)/duration)
    for key in ('ttft','tpot','latency'):
        vals=[r[key] for r in good if r[key] is not None]
        out[key]={f'p{int(q*100)}':percentile(vals,q) for q in (.5,.95)}
    return out


def benchmark(work,urls,output,concurrency=4,repeats=3,warmup=8,rate=None,backend='custom',ttft_slo=2.,tpot_slo=.1,hardware=None):
    if not 1<=concurrency<=8 or repeats<3 or warmup<1 or (rate is not None and rate<=0): raise ValueError('invalid experiment bounds')
    if digest(work['rows'])!=work['digest']: raise ValueError('workload digest mismatch')
    if len(work['rows'])<100: raise ValueError('at least 100 measured requests required for p95 reports')
    if not hardware: raise ValueError('record actual reserved hardware before benchmarking')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    config=dict(workload_digest=work['digest'],urls=urls,backend=backend,concurrency=concurrency,repeats=repeats,warmup=warmup,
        rate=rate,mode='offered_load' if rate else 'finite_batch',ttft_slo=ttft_slo,tpot_slo=tpot_slo,hardware=hardware,
        environment=environment(),cost='unavailable unless actual billed scope supplied')
    write(output/'config.json',config)
    inflight=[0]*len(urls); lock=threading.Lock(); summaries=[]
    nonce=f'{time.time_ns()}'
    def one(row,repeat):
        with lock:
            target=min(range(len(urls)),key=lambda i:inflight[i]); inflight[target]+=1
        request=dict(row,request_id=f'{nonce}-{repeat}-{row["request_id"]}')
        try:
            record=stream(urls[target],request,backend)
        except Exception as exc:
            record={'error':f'{type(exc).__name__}: {exc}','terminal':{'terminal_status':'failed'},'output_tokens':0}
        finally:
            with lock: inflight[target]-=1
        return dict(record,request_id=row['request_id'],repeat=repeat,target=target)
    for target,url in enumerate(urls):
        for i in range(warmup):
            r=dict(work['rows'][i%len(work['rows'])],request_id=f'{nonce}-warm-{target}-{i}')
            result=stream(url,r,backend)
            if result['terminal']['terminal_status']!='completed': raise RuntimeError('warmup failed')
    with open(output/'raw.jsonl','x') as raw:
        for repeat in range(repeats):
            records=[]; started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures=[]
                slots=threading.BoundedSemaphore(concurrency)
                for i,row in enumerate(work['rows']):
                    if rate:
                        delay=started+i/rate-time.perf_counter()
                        if delay>0: time.sleep(delay)
                        if not slots.acquire(False):
                            records.append({'request_id':row['request_id'],'repeat':repeat,'error':'offered_load_drop','terminal':{'terminal_status':'failed'},'output_tokens':0}); continue
                        f=pool.submit(one,row,repeat); f.add_done_callback(lambda _:slots.release()); futures.append(f)
                    else: futures.append(pool.submit(one,row,repeat))
                for f in as_completed(futures): records.append(f.result())
            duration=time.perf_counter()-started
            for record in records: raw.write(json.dumps(record)+'\n')
            raw.flush()
            summaries.append(summarize(records,duration,ttft_slo,tpot_slo))
    write(output/'summary.json',summaries)
    return summaries
