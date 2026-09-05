"""Hardware acceptance harness. Requires local pinned model + two real CUDA GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import time
from .config import Config
from .engine import Engine
from .evidence import environment,write


def consume(engine,row):
    r=engine.submit(row)
    for event in engine.events(r):
        if 'terminal' in event: return event['terminal']


def compare(root,output):
    rows=[dict(request_id=f'mixed-{n}',token_ids=[9707,11,1879]*n,max_new_tokens=8) for n in range(1,9)]
    results=[]
    for world,graphs in ((1,False),(2,False),(2,True)):
        engine=Engine(root,Config(world_size=world,graphs=graphs))
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                records=list(pool.map(lambda row:consume(engine,row),rows))
            # Exercise 8->4->2->1 transitions with fresh request IDs; cancellation is separate.
            for size in (4,2,1):
                subset=[dict(r,request_id=f'b{size}-{r["request_id"]}') for r in rows[:size]]
                with ThreadPoolExecutor(max_workers=size) as pool:
                    records.extend(pool.map(lambda row:consume(engine,row),subset))
            cancelled=engine.submit(dict(request_id='cancel',token_ids=[9707]*128,max_new_tokens=64))
            for event in engine.events(cancelled):
                if 'token_id' in event: engine.cancel(cancelled.key)
                else:
                    assert event['terminal']['terminal_status']=='cancelled'
            results.append(dict(world_size=world,graphs=graphs,records=records,health=engine.health()))
            if graphs:
                assert engine.health()['last_iteration']['graph_replays']>0, 'no actual replay'
                assert all(r['graph_captures']>0 for r in engine.group.info), 'every rank must capture'
        finally: engine.close()
    baseline=[r['output_token_ids'] for r in results[0]['records']]
    for result in results:
        assert all(r['terminal_status']=='completed' for r in result['records'])
        assert [r['output_token_ids'] for r in result['records']]==baseline
    write(output,dict(environment=environment(),results=results,status='passed'))


def fault(root,output):
    records=[]
    # Deterministic injected process exit immediately before and immediately after
    # a model all-reduce while the other rank is executing that collective sequence.
    for phase in ('before_reduce','after_reduce'):
        os.environ['SHARDSERVE_TEST_CRASH']=phase
        engine=Engine(root,Config(watchdog=10))
        try:
            started=time.monotonic()
            result=consume(engine,dict(request_id=phase,token_ids=[9707]*32,max_new_tokens=64))
            elapsed=time.monotonic()-started
            assert result['terminal_status']=='failed'
            assert elapsed<=14.5, f'watchdog bound exceeded: {elapsed}'
            assert all(not p.is_alive() for p in engine.group.processes)
            records.append(dict(phase=phase,shutdown_seconds=elapsed,result=result))
        finally: engine.close(); os.environ.pop('SHARDSERVE_TEST_CRASH',None)
        restarted=Engine(root,Config())
        try:
            result=consume(restarted,dict(request_id='restart',token_ids=[9707],max_new_tokens=2))
            assert result['terminal_status']=='completed'
            records[-1]['restart']=result
        finally: restarted.close()
    # External kill after the stream has actually emitted a token.
    engine=Engine(root,Config(watchdog=10))
    try:
        r=engine.submit(dict(request_id='external-kill',token_ids=[9707]*32,max_new_tokens=1024))
        for event in engine.events(r):
            if 'token_id' in event:
                started=time.monotonic(); engine.group.processes[1].kill(); break
        else: raise AssertionError('stream finished before fault injection')
        with engine.lock:
            while r.state not in ('completed','failed','cancelled','timed_out'): engine.lock.wait(.05)
        elapsed=time.monotonic()-started
        assert r.state=='failed' and elapsed<=14.5
        records.append(dict(phase='external_kill_after_token',shutdown_seconds=elapsed,result=r.result()))
    finally: engine.close()
    write(output,dict(environment=environment(),faults=records,status='passed',watchdog_seconds=10,cleanup_allowance_seconds=4.5))


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['graphs','fault']); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    (compare if a.mode=='graphs' else fault)(a.model,a.output)

if __name__=='__main__': main()
