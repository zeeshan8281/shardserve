"""HTTP adapter validation using a CPU control stub, no inference claims."""
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import urllib.request
from http.server import ThreadingHTTPServer
from shardserve.api import serve
from shardserve.config import Config
from shardserve.scheduler import validate_request
from shardserve.benchmark import stream

class LocalEngine:
    config=Config()
    def __init__(self): self.seen=set()
    def health(self): return {'ready':True}
    def submit(self,row):
        r=validate_request(row,None,self.config,64)
        if r.key in self.seen: raise ValueError('duplicate ID')
        self.seen.add(r.key); return r
    def events(self,r):
        yield {'seq':1,'token_id':3}
        r.output=[3]; r.state='completed'
        yield {'seq':2,'terminal':r.result()}
    def cancel(self,key): pass
    def close(self): pass

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
