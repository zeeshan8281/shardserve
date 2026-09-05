"""Immutable artifacts + single-committer insert-only terminal results.
SQLite is a local recovery-test sink. Delta is the Databricks authoritative sink.
Neither sink promises exactly-once GPU execution.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from .config import MODEL, REVISION, Config
from .scheduler import validate_request
from .weights import digest_file


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def immutable(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=canonical(value)
    # A seal/checksum is published separately. Interrupted writes remain observable
    # and fail validation; POSIX hard links are not required on durable volumes.
    try:
        with open(path,'xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes()!=data: raise ValueError(f'immutable conflict: {path}')



def prepare(rows, snapshot, runtime, code_revision, hardware, root, attempts=2, shard_size=64):
    if type(attempts) is not int or not 1<=attempts<=5 or not 1<=shard_size<=1024:
        raise ValueError('attempts 1..5 and shard_size 1..1024 required')
    config=Config(**runtime)
    rows=list(rows)
    if len(rows)>100000: raise ValueError('v1 job bound is 100000 inputs')
    seen=set()
    for row in rows:
        # Preparation accepts normalized tokens, ensuring comparisons use identical inputs.
        r=validate_request(row,None,config,151936)
        if r.key in seen: raise ValueError('duplicate input request ID')
        if 'deadline' in row: raise ValueError('persisted batch inputs cannot carry wall-clock deadlines')
        seen.add(r.key)
    if not rows: raise ValueError('empty workload')
    rows.sort(key=lambda row:row['request_id'])
    identity=dict(input_snapshot=snapshot,input_digest=digest(rows),model=MODEL,model_revision=REVISION,
        tokenizer_revision=REVISION,generation={'sampling':'greedy','prefix_reuse':False},
        code_revision=code_revision,runtime=runtime,hardware=hardware,max_attempts=attempts,
        retryable_error_codes=['replica_failed','TimeoutError','ConnectionError','transient'])
    if not snapshot or not code_revision or not hardware: raise ValueError('snapshot/code/hardware identity required')
    run=digest(identity)
    directory=Path(root)/run
    shards=[]
    for i in range(0,len(rows),shard_size):
        name=f'input-{i//shard_size:06d}.json'
        immutable(directory/name,rows[i:i+shard_size])
        shards.append({'path':name,'sha256':digest_file(directory/name),'row_count':len(rows[i:i+shard_size])})
    manifest=dict(run_id=run,identity=identity,shards=shards,row_count=len(rows),unique_key_count=len(seen))
    immutable(directory/'run.json',manifest)
    return directory


def read_run(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'run.json').read_text())
    if manifest['run_id']!=digest(manifest['identity']): raise ValueError('run identity mismatch')
    rows=[]
    for shard in manifest['shards']:
        path=directory/shard['path']
        if Path(shard['path']).name!=shard['path'] or digest_file(path)!=shard['sha256']:
            raise ValueError('input checksum mismatch')
        part=json.loads(path.read_text())
        if len(part)!=shard['row_count']: raise ValueError('input shard count mismatch')
        rows.extend(part)
    if digest(rows)!=manifest['identity']['input_digest'] or len(rows)!=manifest['row_count'] or len({r['request_id'] for r in rows})!=manifest['unique_key_count'] or len(rows)!=manifest['unique_key_count']:
        raise ValueError('input identity/count mismatch')
    for row in rows: validate_request(row,None,Config(**manifest['identity']['runtime']),151936)
    return manifest,rows


def stage(directory, manifest, records, attempt_id):
    if not re.fullmatch(r'[a-zA-Z0-9-]{1,80}',attempt_id): raise ValueError('invalid attempt ID')
    records=list(records)
    if len({r['request_id'] for r in records})!=len(records): raise ValueError('duplicate staged keys')
    for r in records:
        if r['run_id']!=manifest['run_id'] or r['attempt_id']!=attempt_id: raise ValueError('attempt identity mismatch')
    path=Path(directory)/'staging'/f'{attempt_id}.json'
    immutable(path,records)
    staging=dict(path=path.name,sha256=digest_file(path),row_count=len(records),unique_key_count=len(records),attempt_id=attempt_id,run_id=manifest['run_id'])
    immutable(path.with_suffix('.manifest'),staging)
    return path.with_suffix('.manifest')


def validate_stage(directory, path):
    manifest,rows=read_run(directory)
    inputs={r['request_id']:r for r in rows}
    m=json.loads(Path(path).read_text())
    if m['run_id']!=manifest['run_id'] or Path(m['path']).name!=m['path']: raise ValueError('staging identity mismatch')
    data=Path(directory)/'staging'/m['path']
    if digest_file(data)!=m['sha256']: raise ValueError('staging checksum mismatch')
    records=json.loads(data.read_text())
    if len(records)!=m['row_count'] or len({r['request_id'] for r in records})!=len(records) or m['unique_key_count']!=len(records):
        raise ValueError('staging duplicate/count mismatch')
    for r in records:
        if r['request_id'] not in inputs or r['run_id']!=m['run_id'] or r['attempt_id']!=m['attempt_id']:
            raise ValueError('invalid output key')
        validate_record(r,inputs[r['request_id']],manifest)
    return records


def validate_record(r, source, manifest):
    if r['run_id'] != manifest['run_id'] or r.get('provenance') != manifest['identity']:
        raise ValueError('result provenance mismatch')
    if r['request_id'] != source['request_id']:
        raise ValueError('result input identity mismatch')
    if not isinstance(r.get('attempt_id'),str) or not r['attempt_id']:
        raise ValueError('missing attempt identity')
    tokens=r['output_token_ids']
    if not isinstance(tokens,list) or any(type(t) is not int or not 0<=t<151936 for t in tokens): raise ValueError('invalid output tokens')
    if type(r['output_tokens']) is not int or type(r['input_tokens']) is not int: raise ValueError('invalid count types')
    if r['output_tokens']!=len(tokens) or len(tokens)>source['max_new_tokens'] or r['input_tokens']!=len(source['token_ids']): raise ValueError('invalid token accounting')
    if r['terminal_status'] not in ('completed','failed','cancelled','timed_out'): raise ValueError('nonterminal attempt')
    if r['terminal_status']=='completed' and (not tokens or r.get('error_code') or r.get('reason') not in ('eos','length')): raise ValueError('invalid successful output')
    if r['terminal_status']=='completed' and r['reason']=='length' and len(tokens)!=source['max_new_tokens']: raise ValueError('short successful length output')
    if r['terminal_status']=='completed' and r['reason']=='eos' and tokens[-1] not in (151643,151645): raise ValueError('false EOS')
    if not isinstance(r.get('output_text'),str) or not isinstance(r.get('timing'),dict): raise ValueError('missing output/timing')
    canonical(r)  # Reject NaN/Infinity anywhere in persisted measurements.

class LocalResults:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS results (run TEXT, request TEXT, payload TEXT, PRIMARY KEY(run,request))')
    def keys(self,run): return {r[0] for r in self.db.execute('SELECT request FROM results WHERE run=?',(run,))}
    def commit(self,records):
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO results VALUES (?,?,?)',[(r['run_id'],r['request_id'],canonical(r).decode()) for r in records])
    def records(self,run): return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM results WHERE run=?',(run,))]
    def close(self): self.db.close()

class DeltaResults:
    def __init__(self,spark,table):
        if not re.fullmatch(r'[A-Za-z_][\w]*(\.[A-Za-z_][\w]*){0,2}',table): raise ValueError('invalid table identifier')
        self.spark,self.table=spark,table
        spark.sql(f'CREATE TABLE IF NOT EXISTS {table} (run_id STRING, request_id STRING, attempt_id STRING, payload STRING) USING DELTA')
    def keys(self,run):
        from pyspark.sql import functions as F
        return {r.request_id for r in self.spark.table(self.table).where(F.col('run_id')==run).select('request_id').collect()}
    def commit(self,records):
        from delta.tables import DeltaTable
        if not records: return
        data=self.spark.createDataFrame([(r['run_id'],r['request_id'],r['attempt_id'],canonical(r).decode()) for r in records],schema='run_id string, request_id string, attempt_id string, payload string')
        DeltaTable.forName(self.spark,self.table).alias('t').merge(data.alias('s'),'t.run_id=s.run_id AND t.request_id=s.request_id').whenNotMatchedInsertAll().execute()
    def records(self,run):
        from pyspark.sql import functions as F
        return [json.loads(r.payload) for r in self.spark.table(self.table).where(F.col('run_id')==run).select('payload').collect()]
    def close(self): pass

@contextmanager
def committer(directory):
    path=Path(directory)/'committer.lock'
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError: raise RuntimeError('committer lock exists; verify prior job has stopped before removing it')
    try:
        os.write(fd,str(os.getpid()).encode()); os.fsync(fd)
        yield
    finally:
        os.close(fd); path.unlink()


def commit_staged(directory,sink):
    committed=0
    for path in sorted((Path(directory)/'staging').glob('*.manifest')):
        try:
            records=validate_stage(directory,path)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            incident={'manifest':path.name,'manifest_digest':digest_file(path),'error':str(exc)}
            immutable(Path(directory)/'validation-errors'/f'{digest(incident)}.json',incident)
            continue
        # Workers cannot finalize failed outputs; exhausted failure is coordinator-owned.
        successful=[r for r in records if r['terminal_status']=='completed']
        sink.commit(successful); committed+=len(successful)
    return committed


def validate_results(directory,sink,complete=True):
    manifest,inputs=read_run(directory)
    results=sink.records(manifest['run_id'])
    keys=[r['request_id'] for r in results]
    expected={r['request_id'] for r in inputs}
    if len(keys)!=len(set(keys)) or not set(keys)<=expected or (complete and set(keys)!=expected):
        raise ValueError('authoritative result key mismatch')
    by_key={r['request_id']:r for r in inputs}
    for result in results: validate_record(result,by_key[result['request_id']],manifest)
    return {'run_id':manifest['run_id'],'unique_terminal_results':len(keys),'successful':sum(r['terminal_status']=='completed' for r in results),'failed':sum(r['terminal_status']!='completed' for r in results),'complete':set(keys)==expected}



def permanent_failures(directory,manifest):
    permanent={}
    for path in sorted((Path(directory)/'staging').glob('*.manifest')):
        try: records=validate_stage(directory,path)
        except (ValueError,KeyError,TypeError,OSError): continue
        for record in records:
            if record['terminal_status']!='completed' and record['error_code'] not in manifest['identity']['retryable_error_codes'] and not record['attempt_id'].startswith('coordinator-'):
                permanent.setdefault(record['request_id'],record)
    return permanent


def execute(directory,sink,engine_factory):
    """One engine group per GPU task; bounded input shards and bounded attempt policy.
    Durable issuance precedes GPU work, so interruption consumes an observable attempt.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    directory=Path(directory)
    manifest,inputs=read_run(directory)
    run=manifest['run_id']; maximum=manifest['identity']['max_attempts']
    with committer(directory):
        validate_results(directory,sink,complete=False)
        commit_staged(directory,sink)
        engine=None
        try:
            for attempt in range(1,maximum+1):
                done=sink.keys(run) | set(permanent_failures(directory,manifest))
                pending=[row for row in inputs if row['request_id'] not in done and not (directory/'issued'/f'{digest(row)}-{attempt}.json').exists()]
                if not pending: continue
                if engine is None: engine=engine_factory()
                width=engine.config.max_live
                for start in range(0,len(pending),width):
                    rows=pending[start:start+width]
                    attempt_id=f'{attempt:02d}-{digest(rows)[:24]}'
                    for row in rows:
                        immutable(directory/'issued'/f'{digest(row)}-{attempt}.json',{'request_id':row['request_id'],'attempt':attempt,'attempt_id':attempt_id,'run_id':run})
                    def generate(row):
                        request=None
                        try:
                            # Internal IDs distinguish retries; durable key stays the input ID.
                            internal=dict(row,request_id=f'{digest(row)[:32]}-{attempt}')
                            request=engine.submit(internal)
                            for event in engine.events(request):
                                if 'terminal' in event: result=event['terminal']
                        except Exception as exc:
                            result=dict(terminal_status='failed',reason=type(exc).__name__,output_token_ids=[],output_tokens=0,input_tokens=len(row['token_ids']),timing={})
                        result.update(request_id=row['request_id'],run_id=run,attempt_id=attempt_id,
                            output_text=engine.tokenizer.decode(result['output_token_ids']),error_code=None if result['terminal_status']=='completed' else result['reason'],provenance=manifest['identity'])
                        return result
                    with ThreadPoolExecutor(max_workers=width) as pool: records=list(pool.map(generate,rows))
                    path=stage(directory,manifest,records,attempt_id)
                    sink.commit([r for r in validate_stage(directory,path) if r['terminal_status']=='completed'])
                if engine.error:
                    engine.close(); engine=None
            done=sink.keys(run)
            failed=[]
            permanent=permanent_failures(directory,manifest)
            for row in inputs:
                if row['request_id'] in done: continue
                issued=[directory/'issued'/f'{digest(row)}-{n}.json' for n in range(1,maximum+1)]
                if row['request_id'] not in permanent and not all(path.exists() for path in issued): raise RuntimeError('cannot finalize unexhausted request')
                reason='permanent_failure' if row['request_id'] in permanent else 'attempts_exhausted'
                failed.append(dict(run_id=run,request_id=row['request_id'],attempt_id='coordinator-exhausted',terminal_status='failed',reason=reason,error_code=reason,output_token_ids=[],output_text='',input_tokens=len(row['token_ids']),output_tokens=0,timing={'finished':time.time()},provenance=manifest['identity']))
            if failed:
                # Unique sealed files survive partial final commits and lost acknowledgments.
                # A retry may repeat coordinator work but cannot overwrite authoritative keys.
                from uuid import uuid4
                final_id='coordinator-'+uuid4().hex
                for record in failed: record['attempt_id']=final_id
                path=stage(directory,manifest,failed,final_id)
                sink.commit(validate_stage(directory,path))
            return validate_results(directory,sink)
        finally:
            if engine: engine.close()
