"""Two real custom FP32 rank forwards with thread-synchronized sum reductions.
This tests full-model partition algebra, not NCCL or GPU execution.
"""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
import torch
from shardserve.weights import specs,shard
from shardserve.model import Model
from shardserve.cache import Cache
from shardserve.runtime import metadata
from test_core import tiny,weights

class ParallelAlgebra(unittest.TestCase):
    def test_two_rank_forward_matches_tp1_and_cached_mixed_batch(self):
        torch.set_num_threads(1)
        c=tiny(); full=weights(c); barrier=threading.Barrier(2,timeout=10); partial=[None,None]
        def worker(rank):
            def reduction(value):
                partial[rank]=value
                barrier.wait()
                result=partial[0]+partial[1]
                barrier.wait()
                return result
            w={key:shard(value,specs(c)[key][1],rank,2) for key,value in full.items()}
            model=Model(c,w,2,reduction,context=64)
            ids=torch.arange(33)%64
            output=model.forward(ids,torch.arange(len(ids)),diagnostic=True)
            cache=Cache(c,2,8,'cpu',torch.float32); cache.blocks.reserve({'a':48,'b':16})
            per_request={'a':[],'b':[]}
            for a_begin,a_end,b_begin,b_end in ((0,15,0,3),(15,17,3,4),(17,33,4,5)):
                rows=[]; offset=0
                for key,begin,end in (('a',a_begin,a_end),('b',b_begin,b_end)):
                    tokens=list(range(begin,end)) if key=='a' else list(range(40+begin,40+end))
                    rows.append(dict(request_id=key,tokens=tokens,context=begin,positions=list(range(begin,end)),offset=offset,query_length=end-begin,sample=True)); offset+=end-begin
                data=metadata({'rows':rows},cache,'cpu',64); data['selected']=None
                logits=model.forward(cache=cache,diagnostic=True,**data)
                for row in rows: per_request[row['request_id']].append(logits[row['offset']:row['offset']+row['query_length']])
            for key in ('a','b'): cache.blocks.release(key)
            cache.blocks.check()
            return output,{k:torch.cat(v) for k,v in per_request.items()}
        with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(worker,range(2)))
        baseline=Model(c,full,context=64)
        for output,cached in results:
            expected=baseline.forward(torch.arange(33),torch.arange(33),diagnostic=True)
            torch.testing.assert_close(output,expected,atol=1e-5,rtol=1e-4)
            torch.testing.assert_close(cached['a'],expected,atol=1e-5,rtol=1e-4)
            expected_b=baseline.forward(torch.arange(40,45),torch.arange(5),diagnostic=True)
            torch.testing.assert_close(cached['b'],expected_b,atol=1e-5,rtol=1e-4)
