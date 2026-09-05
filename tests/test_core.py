import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
import torch
import torch.nn.functional as F
from shardserve.config import Config, validate_model
from shardserve.weights import shard, specs, load
from shardserve.model import Model
from shardserve.cache import Blocks, Cache
from shardserve.runtime import metadata
from shardserve.scheduler import Scheduler, Request, validate_request
from shardserve import batch


def tiny():
    return dict(model_type='qwen2',hidden_act='silu',hidden_size=32,num_attention_heads=4,num_key_value_heads=2,
        intermediate_size=48,num_hidden_layers=2,vocab_size=64,rms_norm_eps=1e-6,rope_theta=1000000.,tie_word_embeddings=True)


def weights(c):
    torch.manual_seed(9281)
    return {key:torch.randn(shape)*(.1 if len(shape)>1 else .03)+(1 if 'norm.weight' in key else 0) for key,(shape,axis) in specs(c).items()}

class Algebra(unittest.TestCase):
    def test_shards_bias_residual_and_paired_mlp(self):
        torch.manual_seed(7)
        for hidden,inter,tokens in ((12,18,5),(16,30,7),(32,48,3)):
            x=torch.randn(tokens,hidden); residual=torch.randn(tokens,hidden)
            gate,up=torch.randn(inter,hidden),torch.randn(inter,hidden)
            down=torch.randn(hidden,inter); bias=torch.randn(hidden)
            expected=residual+F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down,bias)
            partial=[]
            for rank in range(2):
                partial.append(F.linear(F.silu(F.linear(x,shard(gate,0,rank,2)))*F.linear(x,shard(up,0,rank,2)),shard(down,1,rank,2)))
            torch.testing.assert_close(residual+sum(partial)+bias,expected,atol=1e-5,rtol=1e-4)
            for tensor,axis in ((gate,0),(down,1),(bias,0)):
                self.assertTrue(torch.equal(torch.cat([shard(tensor,axis,r,2) for r in range(2)],axis),tensor))

    def test_loader_slices_and_reassembles(self):
        from safetensors.torch import save_file
        c=tiny(); w=weights(c)
        with tempfile.TemporaryDirectory() as tmp:
            save_file(w,str(Path(tmp)/'model.safetensors'))
            ranks=[load(tmp,c,r,2,'cpu',torch.float32) for r in range(2)]
            for key,(_,axis) in specs(c).items():
                actual=ranks[0][key] if axis is None else torch.cat([r[key] for r in ranks],axis)
                self.assertTrue(torch.equal(actual,w[key]))

    def test_real_custom_cache_rope_chunks_and_boundaries(self):
        c=tiny(); model=Model(c,weights(c),context=64)
        ids=torch.arange(35)%c['vocab_size']
        expected=model.forward(ids,torch.arange(len(ids)),diagnostic=True)
        cache=Cache(c,1,8,'cpu',torch.float32); cache.blocks.reserve({'a':48})
        logits=[]
        for begin,end in ((0,7),(7,16),(16,17),(17,32),(32,35)):
            row=dict(request_id='a',tokens=ids[begin:end].tolist(),positions=list(range(begin,end)),context=begin,query_length=end-begin,offset=0,sample=True)
            data=metadata({'rows':[row]},cache,'cpu',64); data['selected']=None
            logits.append(model.forward(cache=cache,diagnostic=True,**data))
        torch.testing.assert_close(torch.cat(logits),expected,atol=1e-5,rtol=1e-4)
        cache.blocks.release('a'); cache.blocks.check()

    def test_hf_tiny_reference(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        c=tiny(); w=weights(c)
        reference=Qwen2ForCausalLM(Qwen2Config(**c)).eval()
        reference.load_state_dict(dict(w,**{'lm_head.weight':w['model.embed_tokens.weight']}))
        reference.config._attn_implementation='eager'
        ids=torch.tensor([3,9,8,12,18,1,27,4,14,13,11,16,24,32,2,6,1])
        with torch.inference_mode(): expected=reference(ids[None]).logits[0]
        actual=Model(c,w).forward(ids,torch.arange(len(ids)),diagnostic=True)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)

class Control(unittest.TestCase):
    def test_invalid(self):
        for kw in ({'world_size':4},{'max_live':9},{'context':4097},{'token_budget':1},{'kv_blocks':0}):
            with self.assertRaises(ValueError): Config(**kw)
        c=tiny(); c['intermediate_size']=47
        with self.assertRaises(ValueError): validate_model(c,2,False)
        valid=dict(request_id='a',token_ids=[1],max_new_tokens=2)
        for patch in ({'request_id':''},{'token_ids':[True]},{'token_ids':[-1]},{'prompt':'also'},{'temperature':0},{'deadline':float('nan')},{'max_new_tokens':True}):
            with self.assertRaises(ValueError): validate_request(dict(valid,**patch),None,Config(),64)

    def test_allocator_transaction(self):
        ranks=[Blocks(4),Blocks(2)]
        for b in ranks: b.reserve({'a':17})
        before=[copy.deepcopy(b.tables) for b in ranks]
        succeeded=[]
        try:
            for b in ranks: b.reserve({'b':1}); succeeded.append(b)
        except MemoryError:
            for b in succeeded: b.release('b')
        self.assertEqual(before,[b.tables for b in ranks])
        for b in ranks:
            b.release('a'); b.check()
            with self.assertRaises(ValueError): b.release('a')
        with self.assertRaises(ValueError): ranks[0].reserve({'negative':-1})

    def test_scheduling_terminal_paths_and_late_arrival(self):
        s=Scheduler(Config(kv_blocks=16,prefill_chunk=3,token_budget=8,output_buffer=2),[63])
        a=Request('a',[1]*7,3); b=Request('b',[2],8)
        s.submit(a); s.submit(b)
        p=s.plan(); self.assertEqual([r['query_length'] for r in p['rows']],[3,1]); s.acknowledge({'b':4})
        s.submit(Request('late',[3],2))
        p=s.plan(); self.assertEqual(p['rows'][0]['request_id'],'b'); b.cancelled=True
        s.acknowledge({r['request_id']:5 for r in p['rows'] if r['sample']})
        self.assertEqual(b.state,'cancelled')
        p=s.plan(); self.assertIn('b',p['release']); s.acknowledge({r['request_id']:63 for r in p['rows'] if r['sample']})
        for _ in range(5):
            p=s.plan(); s.acknowledge({r['request_id']:63 for r in p['rows'] if r['sample']})
        self.assertFalse(s.blocks.tables); s.blocks.check()
        self.assertEqual(a.state,'completed')
        a.finish('failed','later'); self.assertEqual(a.state,'completed')
        with self.assertRaises(ValueError): s.submit(Request('a',[1],1))

    def test_pressure_deadline_backpressure(self):
        s=Scheduler(Config(kv_blocks=1,max_live=1,queue_size=1,output_buffer=1),[63])
        r=Request('a',[1],3); s.submit(r)
        with self.assertRaises(OverflowError): s.submit(Request('b',[1],1))
        s.acknowledge({'a':2}) if s.plan() else None
        p=s.plan(); self.assertEqual(r.reason,'output_backpressure'); s.acknowledge({})
        self.assertFalse(s.blocks.tables)
        r=Request('deadline',[1],1,deadline=time.time()-1); s.submit(r)
        s.plan(); s.acknowledge({}); self.assertEqual(r.state,'timed_out')
        s.fail('crash'); self.assertEqual(r.state,'timed_out')

class Recovery(unittest.TestCase):
    def test_commit_replay_corruption_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[dict(request_id='a',token_ids=[1,2],max_new_tokens=1)]
            root=batch.prepare(rows,'delta:table@7',Config().dict(),'test-source',{'type':'CPU-test'},tmp)
            other=batch.prepare(rows,'delta:table@7',Config(graphs=True).dict(),'test-source',{'type':'CPU-test'},tmp)
            self.assertNotEqual(root,other)
            manifest,_=batch.read_run(root)
            sink=batch.LocalResults(root/'results.db')
            record=dict(run_id=manifest['run_id'],request_id='a',attempt_id='one',terminal_status='completed',output_token_ids=[3],output_tokens=1,input_tokens=2,output_text='test',timing={},error_code=None,reason='length',provenance=manifest['identity'])
            path=batch.stage(root,manifest,[record],'one')
            batch.commit_staged(root,sink); batch.commit_staged(root,sink)
            self.assertEqual(batch.validate_results(root,sink)['successful'],1)
            modified=dict(record,attempt_id='two',output_token_ids=[4])
            batch.stage(root,manifest,[modified],'two'); batch.commit_staged(root,sink)
            self.assertEqual(sink.records(manifest['run_id'])[0]['output_token_ids'],[3])
            (root/'staging'/'one.json').write_text('corrupt')
            with self.assertRaises(ValueError): batch.validate_stage(root,path)
            sink.close()
            with self.assertRaises(ValueError): batch.prepare(rows+rows,'delta:table@7',Config().dict(),'source',{'type':'CPU'},tmp)

if __name__=='__main__': unittest.main()
