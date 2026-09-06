import hashlib
from pathlib import Path
import tempfile
import unittest

from deploy.elastic.bootstrap import chunks, publish
from shardserve.retrieval import ElasticConfig, ElasticSearch, RetrievalUnavailable, prepare_answer, reciprocal_rank_fusion


def hit(identity,score=1.):
    content='document '+identity
    return {'index':'docs-v1','elastic_id':identity,'doc_id':identity,'content':content,
        'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'source_url':'https://example.test/'+identity,
        'repository':'repo','revision':'abc','score':score,'rank':1}


class FakeElastic(ElasticSearch):
    def __init__(self,mode,responses):
        super().__init__(ElasticConfig('https://example.test','secret',mode=mode))
        self.responses=list(responses); self.requests=[]
    def request(self,method,path,body=None,content_type='application/json',allow=()):
        self.requests.append((method,path,body)); return self.responses.pop(0)


def response(*hits):
    return {'hits':{'hits':[{'_index':h['index'],'_id':h['elastic_id'],'_score':h['score'],'_source':{
        k:h[k] for k in ('doc_id','content','content_sha256','source_url','repository','revision')}} for h in hits]}}


class Retrieval(unittest.TestCase):
    def test_rrf_and_client_hybrid(self):
        a,b,c=hit('a'),hit('b'),hit('c')
        fused=reciprocal_rank_fusion([[a,b],[b,c]],3)
        self.assertEqual([h['doc_id'] for h in fused],['b','a','c'])
        client=FakeElastic('client_hybrid',[response(a,b),response(b,c)])
        result=client.search('queue pressure','repo',2)
        self.assertEqual([h['doc_id'] for h in result['hits']],['b','a'])
        self.assertEqual(len(client.requests),2)

    def test_prepare_bounds_and_content_integrity(self):
        client=FakeElastic('lexical',[response(hit('a'))])
        prepared=prepare_answer({'request_id':'q1','question':'why?','max_new_tokens':12},client)
        self.assertIn('[source: a]',prepared['engine_row']['prompt'])
        self.assertEqual(prepared['retrieval']['index_names'],['docs-v1'])
        broken=hit('bad'); broken['content_sha256']='0'*64
        with self.assertRaises(RetrievalUnavailable): FakeElastic('lexical',[response(broken)]).search('why')
        for row in ({'request_id':'q1','question':''},{'request_id':'q1','question':'x','top_k':0},{'request_id':'q1','question':'x','extra':1}):
            with self.assertRaises(ValueError): prepare_answer(row,client)

    def test_config_rejects_insecure_remote_url(self):
        with self.assertRaises(ValueError): ElasticConfig('http://example.test','secret')
        with self.assertRaises(ValueError): ElasticConfig('https://user:pass@example.test','secret')

    def test_bounded_chunking_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'module.py'; source.write_text(''.join(f'line {n}\n' for n in range(140)))
            documents=list(chunks(root,[source],'repo','abc','https://example.test/blob/abc'))
            self.assertEqual(len(documents),2)
            self.assertEqual(documents[1]['line_start'],101)
            self.assertEqual(hashlib.sha256(documents[0]['content'].encode()).hexdigest(),documents[0]['content_sha256'])
        class Client:
            def __init__(self): self.calls=[]
            def request(self,method,path,body=None,content_type='application/json',allow=()):
                self.calls.append((method,path,body,content_type))
                if method=='HEAD': return None
                if path=='/_bulk': return {'errors':False,'items':[{} for _ in documents]}
                if path.endswith('/_count'): return {'count':len(documents)}
                if path.startswith('/_alias/'): return None
                return {'acknowledged':True}
        client=Client()
        self.assertEqual(publish(client,'docs-vabc','docs-live',documents),2)
        self.assertEqual(client.calls[-1][1],'/_aliases')
        self.assertEqual(client.calls[-1][2]['actions'],[{'add':{'index':'docs-vabc','alias':'docs-live'}}])


if __name__=='__main__': unittest.main()
