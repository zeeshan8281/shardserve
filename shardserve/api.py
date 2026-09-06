"""Authenticated HTTP adapter; raw generation and retrieval share one engine."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from urllib.parse import unquote
from .retrieval import RetrievalUnavailable, answer_response, prepare_answer

MAX_BODY=65536
UI='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ShardServe IR</title><style>
body{margin:0;background:#090b10;color:#eef2ff;font:16px system-ui,sans-serif}main{max-width:760px;margin:8vh auto;padding:24px}
h1{font-size:clamp(2rem,7vw,4.5rem);margin:0}.sub{color:#9da8bd;margin:8px 0 36px}label{display:block;margin:18px 0 7px;color:#c9d2e3}
input,textarea,button{box-sizing:border-box;width:100%;border:1px solid #31394a;border-radius:10px;background:#111620;color:#eef2ff;padding:13px;font:inherit}
textarea{min-height:120px;resize:vertical}button{margin-top:18px;background:#5b7cfa;border:0;font-weight:700;cursor:pointer}button:disabled{opacity:.6}
#status{color:#9da8bd;margin:18px 0}#answer{white-space:pre-wrap;line-height:1.6}.cite{display:block;color:#91a9ff;margin:8px 0;overflow-wrap:anywhere}
</style><main><h1>ShardServe IR</h1><p class="sub">Elastic hybrid retrieval + a custom tensor-parallel Qwen engine on two L4 GPUs.</p>
<form id="form"><label for="token">Access token</label><input id="token" type="password" autocomplete="off" required>
<label for="question">Question</label><textarea id="question" required>Why does ShardServe reserve full KV capacity before admitting a request?</textarea>
<button id="ask">Ask ShardServe</button></form><p id="status">The token stays in this page and is sent only to this service.</p><div id="answer"></div><div id="citations"></div></main>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#ask'),status=document.querySelector('#status'),answer=document.querySelector('#answer'),citations=document.querySelector('#citations');
form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;answer.textContent='';citations.replaceChildren();status.textContent='Retrieving sources and running TP2 inference...';
try{const response=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+document.querySelector('#token').value},body:JSON.stringify({request_id:'web-'+crypto.randomUUID(),question:document.querySelector('#question').value,repository:'shardserve',top_k:6,max_new_tokens:160,include_debug:true})});const data=await response.json();if(!response.ok)throw new Error(data.error||('HTTP '+response.status));answer.textContent=data.answer;for(const item of data.citations||[]){const link=document.createElement('a');link.className='cite';link.textContent=item.doc_id;link.href=item.source_url;link.target='_blank';link.rel='noreferrer';citations.append(link)}status.textContent=`${data.terminal_status} · ${data.usage.input_tokens} input tokens · ${data.usage.output_tokens} output tokens`;}
catch(error){status.textContent=error.message==='unauthorized'?'Wrong or expired access token.':error.message}finally{button.disabled=false}});
</script></html>'''.encode()


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
            if self.path=='/':
                self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self.send_header('Content-Length',str(len(UI))); self.end_headers(); self.wfile.write(UI); return
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
