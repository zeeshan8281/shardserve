"""Real pinned-model reference checks via torchrun; never calibrate tolerance silently."""
import argparse
import json
import os
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from .weights import verify,load
from .model import Model
from .cache import Cache
from .runtime import metadata,reduce
from .evidence import environment,write

CORPUS=[
    [48,25,220,17,488,220,17,4035,32,25],
    [2507,39772,25,220,16,11,220,17,11,220,18,11,220,19,11],
    [785,6722,315,9625,374],
    [12548,279,5383,25,220,16,220,17,220,18,220,16,220,17,220,18,220,16,220,17],
    [20841,448,6896,825,3409,25,6303],
    [40,1101,264,1091,13],
    [16,17,18,19,20]*5,
]


def errors(actual,expected):
    a,b=actual.float(),expected.float()
    delta=(a-b).abs()
    top=b.topk(2,dim=-1).values
    return {'max_absolute':delta.max().item(),'max_relative':(delta/b.abs().clamp_min(1e-6)).max().item(),
        'normalized_rmse':(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-6)).item(),
        'min_top2_margin':(top[:,0]-top[:,1]).min().item(),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
    p.add_argument('--envelope'); p.add_argument('--repeats',type=int,default=3); args=p.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,2) or not torch.cuda.is_available() or args.repeats<3: raise RuntimeError('CUDA TP1/TP2 and at least three repeats required')
    c=verify(args.model); torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=60))
    report={'environment':environment(),'world_size':world,'records':[],'correctness_gate':'unverified_missing_numerical_envelope'}
    try:
        references=[]
        if rank==0:
            from transformers import AutoModelForCausalLM
            hf=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
            with torch.inference_mode():
                for prompt in CORPUS:
                    ids=torch.tensor(prompt,device='cuda')[None]
                    output=hf(ids,output_hidden_states=True)
                    reference=output.logits[0].float().cpu()
                    # HF hidden_states[1:-1] are completed intermediate decoder layers.
                    layers=[t[0].float().cpu() for t in output.hidden_states[1:-1]]
                    sequence=list(prompt); generated=[]
                    for _ in range(8):
                        token=int(hf(torch.tensor(sequence,device='cuda')[None]).logits[0,-1].argmax())
                        generated.append(token); sequence.append(token)
                        if token in (151643,151645): break
                    references.append((reference,layers,generated))
            del hf,output
            torch.cuda.empty_cache()
        dist.barrier()
        model=Model(c,load(args.model,c,rank,world,'cuda',torch.bfloat16),world,reduce)
        cache=Cache(c,world,512,'cuda',torch.bfloat16)
        for repeat in range(args.repeats):
            for i,prompt in enumerate(CORPUS):
                ids=torch.tensor(prompt,device='cuda'); positions=torch.arange(len(prompt),device='cuda'); trace=[]
                logits=model.forward(ids,positions,diagnostic=True,trace=trace)
                cache.blocks.reserve({'check':4096})
                outputs=[]
                for begin in range(0,len(prompt),7):
                    end=min(len(prompt),begin+7)
                    row=dict(request_id='check',tokens=prompt[begin:end],positions=list(range(begin,end)),context=begin,offset=0,query_length=end-begin,sample=True)
                    data=metadata({'rows':[row]},cache,'cuda',4096); data['selected']=None
                    outputs.append(model.forward(cache=cache,**data))
                cached=torch.cat(outputs)
                generated=[]; sequence=list(prompt); decode_errors=[]
                last_logits=cached[-1:]
                for step in range(8):
                    greedy=model.forward(torch.tensor(sequence,device='cuda'),torch.arange(len(sequence),device='cuda'),diagnostic=True)
                    if rank==0: decode_errors.append(errors(last_logits.cpu(),greedy[-1:].cpu()))
                    token=last_logits[-1].argmax().reshape(1) if rank==0 else torch.empty(1,device='cuda',dtype=torch.long)
                    dist.broadcast(token,0); value=int(token.item())
                    generated.append(value); sequence.append(value)
                    if value in (151643,151645): break
                    if step<7:
                        pos=len(sequence)-1
                        row=dict(request_id='check',tokens=[value],positions=[pos],context=pos,offset=0,query_length=1,sample=True)
                        last_logits=model.forward(cache=cache,**metadata({'rows':[row]},cache,'cuda',4096))
                cache.blocks.release('check')
                if rank==0:
                    ref,layers,expected=references[i]
                    report['records'].append(dict(repeat=repeat,corpus=i,teacher_forced=errors(logits.cpu(),ref),cache=errors(cached.cpu(),ref),
                        layers=[errors(a,b) for a,b in zip(trace[:-1],layers)],decode=decode_errors,generated=generated,reference_generated=expected,greedy_equal=generated==expected))
        if rank==0:
            if args.envelope:
                envelope=json.loads(Path(args.envelope).read_text())
                if not envelope.get('evidence_files') or not envelope.get('numerical_explanation'): raise ValueError('envelope requires evidence and numerical explanation')
                checks=[]
                for r in report['records']:
                    for check in (r['teacher_forced'],r['cache'],*r['layers'],*r['decode']):
                        checks.extend(check[key]<=envelope[key] for key in ('max_absolute','max_relative','normalized_rmse'))
                    checks.append(r['greedy_equal'])
                report['correctness_gate']='passed' if all(checks) else 'failed'
            write(args.output,report)
            if report['correctness_gate']!='passed': raise RuntimeError(report['correctness_gate'])
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
