import argparse
import json
from pathlib import Path
from .config import Config
from .evidence import environment,write,source


def main():
    parser=argparse.ArgumentParser(prog='shardserve')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('env'); p.add_argument('--output',default='artifacts/environment.json')
    p=sub.add_parser('fetch-model'); p.add_argument('--cache',default='models')
    p=sub.add_parser('check'); p.add_argument('--output',default='artifacts/cpu-tests.txt')
    p=sub.add_parser('serve'); p.add_argument('--model',required=True); p.add_argument('--world-size',type=int,default=2); p.add_argument('--graphs',action='store_true'); p.add_argument('--port',type=int,default=8080); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--config')
    p=sub.add_parser('workload'); p.add_argument('--output',required=True); p.add_argument('--prompt-tokens',type=int,default=128); p.add_argument('--output-tokens',type=int,default=32); p.add_argument('--count',type=int,default=100); p.add_argument('--seed',type=int,default=9281); p.add_argument('--repeated-prefix',action='store_true')
    p=sub.add_parser('benchmark'); p.add_argument('--workload',required=True); p.add_argument('--urls',nargs='+',required=True); p.add_argument('--output',required=True); p.add_argument('--concurrency',type=int,default=4); p.add_argument('--rate',type=float); p.add_argument('--backend',choices=['custom','vllm'],default='custom'); p.add_argument('--hardware',required=True); p.add_argument('--ttft-slo',type=float,default=2); p.add_argument('--tpot-slo',type=float,default=.1)
    p=sub.add_parser('prepare'); p.add_argument('--workload',required=True); p.add_argument('--directory',required=True); p.add_argument('--hardware',required=True); p.add_argument('--config'); p.add_argument('--max-attempts',type=int,default=2)
    for name in ('batch','resume','validate'):
        p=sub.add_parser(name); p.add_argument('directory'); p.add_argument('--model'); p.add_argument('--results',required=True)
    p=sub.add_parser('upload'); p.add_argument('directory'); p.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if args.command=='env':
        report=environment(); write(args.output,report); print(json.dumps(report,indent=2))
    elif args.command=='fetch-model':
        from .weights import fetch
        print(fetch(args.cache))
    elif args.command=='check':
        import subprocess,sys,os
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        with open(args.output,'w') as f:
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        print(Path(args.output).read_text()); raise SystemExit(result.returncode)
    elif args.command=='serve':
        from .engine import Engine
        from .api import serve
        cfg=Config(**json.loads(Path(args.config).read_text())) if args.config else Config(world_size=args.world_size,graphs=args.graphs)
        if args.host not in ('127.0.0.1','localhost','::1') and not __import__('os').environ.get('SHARDSERVE_API_TOKEN'): parser.error('non-loopback binding requires SHARDSERVE_API_TOKEN')
        serve(Engine(args.model,cfg),args.host,args.port)
    elif args.command=='workload':
        from .benchmark import workload
        write(args.output,workload(args.prompt_tokens,args.output_tokens,args.count,args.seed,args.repeated_prefix))
    elif args.command=='benchmark':
        from .benchmark import benchmark
        benchmark(json.loads(Path(args.workload).read_text()),args.urls,args.output,args.concurrency,rate=args.rate,backend=args.backend,ttft_slo=args.ttft_slo,tpot_slo=args.tpot_slo,hardware=json.loads(Path(args.hardware).read_text()))
    elif args.command=='prepare':
        from .batch import prepare
        work=json.loads(Path(args.workload).read_text()); cfg=json.loads(Path(args.config).read_text()) if args.config else Config().dict()
        print(prepare(work['rows'],{'workload_digest':work['digest']},cfg,source(),json.loads(Path(args.hardware).read_text()),args.directory,args.max_attempts))
    elif args.command in ('batch','resume','validate'):
        from .batch import LocalResults,read_run,execute,validate_results
        sink=LocalResults(args.results)
        try:
            if args.command=='validate': print(json.dumps(validate_results(args.directory,sink)))
            else:
                from .engine import Engine
                manifest,_=read_run(args.directory)
                if manifest['identity']['code_revision']!=source(): raise ValueError('code changed; prepare new run')
                if not args.model: parser.error('batch/resume requires --model')
                print(json.dumps(execute(args.directory,sink,lambda:Engine(args.model,Config(**manifest['identity']['runtime'])))))
        finally: sink.close()
    elif args.command=='upload':
        from .evidence import upload
        upload(args.directory,args.experiment)
