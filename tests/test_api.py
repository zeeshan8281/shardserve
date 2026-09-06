"""HTTP adapter validation using a CPU control stub, no inference claims."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import urllib.request
from http.server import ThreadingHTTPServer
from shardserve.api import serve
from shardserve.config import Config
from shardserve.retrieval import ElasticConfig
from shardserve.scheduler import validate_request
from shardserve.benchmark import stream

class LocalTokenizer:
    vocab_size=64
    def apply_chat_template(self,*args,**kwargs): return [1,2]
    def decode(self,tokens): return 'answer text'

class LocalEngine:
    config=Config()
    tokenizer=LocalTokenizer()
    def __init__(self): self.seen=set()
    def health(self): return {'ready':True}
    def submit(self,row):
        r=validate_request(row,self.tokenizer,self.config,64)
        if r.key in self.seen: raise ValueError('duplicate ID')
        self.seen.add(r.key); return r
    def events(self,r):
        yield {'seq':1,'token_id':3}
        r.output=[3]; r.state='completed'
        yield {'seq':2,'terminal':r.result()}
    def cancel(self,key): pass
    def close(self): pass

class LocalRetriever:
    config=ElasticConfig('https://example.test','key')
    def search(self,question,repository,top_k):
        content='Full request capacity is reserved before admission.'
        import hashlib
        return {'config_sha256':'a'*64,'hits':[{'index':'docs-v1','elastic_id':'1','doc_id':'scheduler.py:L1-L2',
            'content':content,'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
            'source_url':'https://example.test/scheduler.py#L1-L2','repository':'shardserve','revision':'abc','rank':1,'score':1.0}]}

class API(unittest.TestCase):
    def test_http_stream_and_validation(self):
        server=[]; ready=threading.Event()
        original=ThreadingHTTPServer.serve_forever
        def start(s):
            server.append(s); ready.set(); original(s,poll_interval=.01)
        with patch.object(ThreadingHTTPServer,'serve_forever',start):
            thread=threading.Thread(target=serve,args=(LocalEngine(),'127.0.0.1',0),daemon=True); thread.start()
            if not ready.wait(5): self.fail('HTTP server failed to start')
            url=f'http://127.0.0.1:{server[0].server_port}'
            try:
                row=dict(request_id='http',token_ids=[1],max_new_tokens=1)
                result=stream(url,row,'custom')
                self.assertEqual(result['output_tokens'],1)
                for bad in (row,dict(row,request_id='other',temperature=1)):
                    request=urllib.request.Request(url+'/stream',data=json.dumps(bad).encode(),headers={'Content-Type':'application/json'})
                    with self.assertRaises(urllib.error.HTTPError) as raised: urllib.request.urlopen(request)
                    self.assertEqual(raised.exception.code,400)
            finally:
                server[0].shutdown(); thread.join(5)

    def test_answer_contract_and_disabled_retrieval(self):
        for retriever,expected in ((None,503),(LocalRetriever(),200)):
            server=[]; ready=threading.Event()
            original=ThreadingHTTPServer.serve_forever
            def start(s): server.append(s); ready.set(); original(s,poll_interval=.01)
            with patch.object(ThreadingHTTPServer,'serve_forever',start):
                thread=threading.Thread(target=serve,args=(LocalEngine(),'127.0.0.1',0,retriever),daemon=True); thread.start()
                self.assertTrue(ready.wait(5)); url=f'http://127.0.0.1:{server[0].server_port}'
                try:
                    row={'request_id':'answer-1','question':'Why reserve KV?','repository':'shardserve','include_debug':True}
                    request=urllib.request.Request(url+'/answer',data=json.dumps(row).encode(),headers={'Content-Type':'application/json'})
                    if expected==503:
                        with self.assertRaises(urllib.error.HTTPError) as raised: urllib.request.urlopen(request)
                        self.assertEqual(raised.exception.code,expected)
                    else:
                        result=json.loads(urllib.request.urlopen(request).read())
                        self.assertEqual(result['answer'],'answer text')
                        self.assertEqual(result['citations'][0]['doc_id'],'scheduler.py:L1-L2')
                        self.assertEqual(result['engine']['tensor_parallel_size'],2)
                        self.assertEqual(len(result['retrieval']['prompt_token_sha256']),64)
                        command=subprocess.run([sys.executable,'-m','shardserve','ask','--url',url,
                            '--question','Why reserve KV?','--repository','shardserve'],capture_output=True,text=True)
                        self.assertEqual(command.returncode,0,command.stderr)
                        self.assertIn('answer text',command.stdout)
                        self.assertIn('scheduler.py:L1-L2',command.stdout)
                finally:
                    server[0].shutdown(); thread.join(5)
