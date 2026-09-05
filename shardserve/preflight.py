"""Run via torchrun: compile the actual paged kernel and verify collectives."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from .evidence import environment,write


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); args=p.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,2) or torch.cuda.device_count()<world: raise RuntimeError('TP1/TP2 CUDA ranks required')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=30))
    try:
        x=torch.tensor([rank+1.],device='cuda'); dist.all_reduce(x)
        assert x.item()==world*(world+1)/2
        dist.broadcast(x,0)
        from .kernel import ragged_attention_direct
        q=torch.ones((1,8,128),device='cuda',dtype=torch.bfloat16)
        k=torch.ones((1,16,1,128),device='cuda',dtype=torch.bfloat16); v=k*2
        table=torch.zeros((1,1),device='cuda',dtype=torch.int32)
        index=torch.zeros(1,device='cuda',dtype=torch.int32)
        y=ragged_attention_direct(q,k,v,table,index,index,128**-.5)
        torch.testing.assert_close(y,torch.full_like(y,2))
        dist.barrier(); torch.cuda.synchronize()
        if rank==0: write(args.output,dict(environment=environment(),collective_and_kernel='passed',clean_shutdown='verify torchrun exit code'))
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
