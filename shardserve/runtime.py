"""Spawned rank group, ordered RPC, NCCL collectives and bounded parent watchdog."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from datetime import timedelta
import torch
import torch.distributed as dist
from .cache import Cache
from .config import Config
from .model import Model
from .weights import load, verify


def reduce(x):
    fault=os.environ.get('SHARDSERVE_TEST_CRASH')
    if fault=='before_reduce' and dist.get_rank()==1:
        os._exit(86)
    if dist.get_world_size()>1:
        dist.all_reduce(x)
    if fault=='after_reduce' and dist.get_rank()==1:
        os._exit(87)
    return x


def metadata(plan, cache, device, context, bucket=None):
    rows = plan['rows']
    n = bucket or len(rows)
    tables = torch.zeros((n,(context+15)//16),dtype=torch.int32,device=device)
    tokens,positions,seqs,slots,selected = [],[],[],[],[]
    for i,row in enumerate(rows):
        table = cache.blocks.tables[row['request_id']]
        tables[i,:len(table)] = torch.tensor(table,dtype=torch.int32,device=device)
        tokens.extend(row['tokens']); positions.extend(row['positions'])
        seqs.extend([i]*row['query_length'])
        slots.extend(table[p//16]*16+p%16 for p in row['positions'])
        if row['sample']:
            selected.append(row['offset']+row['query_length']-1)
    for i in range(len(rows),n):
        block = cache.blocks.count+i
        tables[i,0] = block
        tokens.append(0); positions.append(0); seqs.append(i); slots.append(block*16)
    return dict(ids=torch.tensor(tokens,dtype=torch.long,device=device),
        positions=torch.tensor(positions,dtype=torch.long,device=device),tables=tables,
        seqs=torch.tensor(seqs,dtype=torch.int32,device=device),
        slots=torch.tensor(slots,dtype=torch.long,device=device),
        selected=torch.tensor(selected,dtype=torch.long,device=device))

class DecodeGraphs:
    def __init__(self, model, cache, config, device):
        self.entries={}; self.captures=0; self.replays=0
        start=time.monotonic()
        for size in (1,2,4,8):
            if size>config.max_live:
                continue
            data=metadata({'rows':[]},cache,device,config.context,size)
            data.pop('selected')
            stream=torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model.forward(cache=cache,**data)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out=model.forward(cache=cache,**data)
            self.entries[size]=(graph,data,out)
            self.captures+=1
        self.startup_seconds=time.monotonic()-start

    def run(self, data, size):
        graph,buffers,out=self.entries[size]
        for key,value in buffers.items():
            value.copy_(data[key])
        graph.replay(); self.replays+=1
        return out


def rank_main(rank, root, cfg, rendezvous, conn):
    config=Config(**cfg)
    if rank != 0:
        conn.close()
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING','1')
    torch.cuda.set_device(rank)
    device=torch.device('cuda',rank)
    dist.init_process_group('nccl',init_method=rendezvous,rank=rank,world_size=config.world_size,
        timeout=timedelta(seconds=config.watchdog))
    try:
        c=json.loads((Path(root)/'config.json').read_text())
        model=Model(c,load(root,c,rank,config.world_size,device,torch.bfloat16),config.world_size,reduce,config.context)
        cache=Cache(c,config.world_size,config.kv_blocks,device,torch.bfloat16)
        graphs=DecodeGraphs(model,cache,config,device) if config.graphs else None
        info=dict(rank=rank,device=torch.cuda.get_device_name(rank),weights_bytes=sum(w.numel()*w.element_size() for w in model.w.values()),
            kv_bytes=cache.bytes,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            graph_capture_seconds=graphs.startup_seconds if graphs else 0,graph_captures=graphs.captures if graphs else 0)
        infos=[None]*config.world_size
        dist.all_gather_object(infos,info)
        if rank==0:
            conn.send({'ready':infos})
        profiler=None
        trace_dir=os.environ.get('SHARDSERVE_TRACE_DIR')
        if trace_dir:
            Path(trace_dir).mkdir(parents=True,exist_ok=True)
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0,warmup=1,active=8,repeat=1),
                on_trace_ready=lambda p:p.export_chrome_trace(str(Path(trace_dir)/f'rank-{rank}.json')))
            profiler.__enter__()
        iteration=0
        while True:
            command=[conn.recv() if rank==0 else None]
            dist.broadcast_object_list(command,src=0,device=device)
            command=command[0]
            if command['op']=='stop':
                break
            plan=command['plan']
            if plan['iteration']!=iteration+1:
                raise RuntimeError('non-monotonic iteration')
            iteration=plan['iteration']
            for key in plan['release']:
                cache.blocks.release(key)
            allocated=False
            try:
                cache.blocks.reserve(plan['reserve'])
                allocated=True
            except MemoryError:
                pass
            ok=torch.tensor(int(allocated),device=device)
            dist.all_reduce(ok,op=dist.ReduceOp.MIN)
            if not ok.item():
                if allocated:
                    for key in plan['reserve']:
                        cache.blocks.release(key)
                raise MemoryError('coordinated allocation failed; group aborted')
            rows=plan['rows']
            samples={}
            if rows:
                size=next((s for s in (1,2,4,8) if s>=len(rows)),None)
                use_graph=graphs is not None and plan['mode']=='decode' and size in graphs.entries
                data=metadata(plan,cache,device,config.context,size if use_graph else None)
                if use_graph:
                    logits=graphs.run(data,size)[:len(rows)]
                else:
                    logits=model.forward(cache=cache,diagnostic=config.diagnostic,**data)
                chosen=[row['request_id'] for row in rows if row['sample']]
                tokens=torch.empty(len(chosen),dtype=torch.long,device=device)
                if rank==0:
                    if use_graph:
                        indices=[i for i,r in enumerate(rows) if r['sample']]
                        tokens.copy_(logits[indices].argmax(-1))
                    else:
                        tokens.copy_(logits.argmax(-1))
                dist.broadcast(tokens,src=0)
                samples=dict(zip(chosen,tokens.tolist()))
            cache.blocks.check()
            state=dict(live=sorted(cache.blocks.tables),blocks_by_request={key:len(value) for key,value in sorted(cache.blocks.tables.items())},free=len(cache.blocks.free),iteration=iteration)
            states=[None]*config.world_size
            dist.all_gather_object(states,state)
            if any(s!=states[0] for s in states):
                raise RuntimeError('logical KV state diverged')
            measurements=[None]*config.world_size
            dist.all_gather_object(measurements,dict(rank=rank,graph_replays=graphs.replays if graphs else 0,peak_bytes=torch.cuda.max_memory_allocated(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
            if profiler: profiler.step()
            if rank==0:
                conn.send(dict(samples=samples,state=state,graph_replays=graphs.replays if graphs else 0,ranks=measurements))
        if profiler: profiler.__exit__(None,None,None)
    finally:
        dist.destroy_process_group()
        conn.close()

class Group:
    def __init__(self, root, config):
        self.config=config
        if not torch.cuda.is_available() or torch.cuda.device_count()<config.world_size:
            raise RuntimeError(f'{config.world_size} CUDA GPUs required; found {torch.cuda.device_count()}')
        for rank in range(config.world_size):
            with torch.cuda.device(rank):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f'rank {rank} lacks BF16 support')
        verify(root)
        self.ctx=mp.get_context('spawn')
        self.temp=tempfile.TemporaryDirectory(prefix='shardserve-rendezvous-')
        rendezvous=Path(self.temp.name,'init').as_uri()
        self.conn,child=self.ctx.Pipe()
        self.processes=[]
        self.closed=False
        try:
            for rank in range(config.world_size):
                p=self.ctx.Process(target=rank_main,args=(rank,str(root),config.dict(),rendezvous,child))
                p.start(); self.processes.append(p)
            child.close()
            self.info=self.receive(config.startup_timeout)['ready']
        except BaseException:
            child.close(); self.close(force=True); raise

    def receive(self, timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if any(p.exitcode is not None for p in self.processes):
                self.close(force=True)
                raise RuntimeError('rank exited; entire replica failed')
            if self.conn.poll(min(0.05,max(0,end-time.monotonic()))):
                try:
                    return self.conn.recv()
                except EOFError:
                    self.close(force=True)
                    raise RuntimeError('rank connection closed')
        self.close(force=True)
        raise TimeoutError('rank watchdog expired')

    def step(self, plan):
        if self.closed:
            raise RuntimeError('group is closed')
        self.conn.send({'op':'step','plan':plan})
        return self.receive(self.config.watchdog)

    def close(self, force=False):
        if self.closed:
            return
        self.closed=True
        if not force:
            try:
                self.conn.send({'op':'stop'})
            except (OSError,EOFError):
                pass
        deadline=time.monotonic()+(0 if force else self.config.watchdog)
        for p in self.processes:
            p.join(max(0,deadline-time.monotonic()))
        for p in self.processes:
            if p.is_alive(): p.terminate()
        deadline=time.monotonic()+2
        for p in self.processes: p.join(max(0,deadline-time.monotonic()))
        for p in self.processes:
            if p.is_alive(): p.kill()
        deadline=time.monotonic()+2
        for p in self.processes: p.join(max(0,deadline-time.monotonic()))
        self.conn.close(); self.temp.cleanup()
