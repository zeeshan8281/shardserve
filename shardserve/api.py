"""Authenticated HTTP adapter; raw generation and retrieval share one engine."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from urllib.parse import unquote
from .retrieval import RetrievalUnavailable, answer_response, prepare_answer

MAX_BODY=65536


def serve(engine, host='127.0.0.1', port=8080, retriever=None):
    secret=os.environ.get('SHARDSERVE_API_TOKEN')
    if host not in ('127.0.0.1','localhost','::1') and not secret:
        raise ValueError('non-loopback binding requires SHARDSERVE_API_TOKEN')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # No prompts/IDs/secrets in access logs.
        def setup(self):
            super().setup(); self.connection.settimeout(10)
        def authorized(self):
            return not secret or hmac.compare_digest(self.headers.get('Authorization',''),f'Bearer {secret}')
        def send_json(self,status,value):
            data=json.dumps(value,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if not self.authorized(): return self.send_json(401,{'error':'unauthorized'})
            if self.path not in ('/health','/metrics'): return self.send_json(404,{'error':'not_found'})
            health=engine.health(); health['retrieval_configured']=retriever is not None
            self.send_json(200 if health['ready'] else 503,health)
        def do_DELETE(self):
            if not self.authorized(): return self.send_json(401,{'error':'unauthorized'})
            if not self.path.startswith('/requests/'): return self.send_json(404,{'error':'not_found'})
            try:
                engine.cancel(unquote(self.path[len('/requests/'):]))
                self.send_json(202,{'status':'cancellation_requested'})
            except KeyError: self.send_json(404,{'error':'unknown_request'})
        def do_POST(self):
            if not self.authorized(): return self.send_json(401,{'error':'unauthorized'})
            if self.path not in ('/generate','/stream','/answer'): return self.send_json(404,{'error':'not_found'})
            request=None
            streaming=False
            prepared=None
            try:
                if self.headers.get('Transfer-Encoding'): raise ValueError('chunked requests unsupported')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_BODY: return self.send_json(413,{'error':'body_size'})
                body=self.rfile.read(length)
                if len(body)!=length: raise ValueError('truncated body')
                row=json.loads(body,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if self.path=='/answer':
                    if retriever is None: return self.send_json(503,{'error':'retrieval_not_configured'})
                    prepared=prepare_answer(row,retriever); row=prepared['engine_row']
                request=engine.submit(row)
                if self.path=='/stream':
                    self.send_response(200); self.send_header('Content-Type','text/event-stream')
                    self.send_header('Cache-Control','no-cache'); self.end_headers(); streaming=True
                for event in engine.events(request):
                    if streaming:
                        self.wfile.write(f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n".encode()); self.wfile.flush()
                    elif 'terminal' in event:
                        result=event['terminal']; result['output_text']=engine.tokenizer.decode(result['output_token_ids'])
                        if prepared:
                            result=answer_response(result,prepared,request.prompt,engine.config.world_size)
                        self.send_json(200,result)
            except (ValueError,TypeError,UnicodeError) as exc:
                if not streaming: self.send_json(400,{'error':str(exc)})
            except LookupError as exc:
                if not streaming: self.send_json(404,{'error':str(exc)})
            except OverflowError as exc:
                self.send_json(429,{'error':str(exc)})
            except RetrievalUnavailable as exc:
                if not streaming: self.send_json(502,{'error':str(exc)})
            except RuntimeError as exc:
                if not streaming: self.send_json(503,{'error':str(exc)})
            except (BrokenPipeError,ConnectionResetError,TimeoutError,OSError):
                pass
            finally:
                if request and request.state not in ('completed','failed','cancelled','timed_out'):
                    engine.cancel(request.key)
    # Bound simultaneous transports independently of waiting/request queues.
    import threading
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        def __init__(self,*args):
            self.slots=threading.BoundedSemaphore(engine.config.queue_size+engine.config.max_live)
            super().__init__(*args)
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                request.close(); return
            try: super().process_request(request,address)
            except BaseException:
                self.slots.release(); raise
        def process_request_thread(self,request,address):
            try: super().process_request_thread(request,address)
            finally: self.slots.release()
    server=Server((host,port),Handler)
    try: server.serve_forever()
    finally: server.server_close(); engine.close()
