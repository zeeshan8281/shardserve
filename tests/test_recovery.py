import json
from pathlib import Path
import tempfile
import unittest
from shardserve import batch
from shardserve.config import Config

class FakeTokenizer:
    def decode(self,tokens): return ' '.join(map(str,tokens))

class ControlledEngine:
    """Only tests persistence/control. Never counted as model or TP evidence."""
    config=Config()
    tokenizer=FakeTokenizer()
    error=None
    def __init__(self,fail=False): self.fail=fail
    def submit(self,row): return row
    def events(self,row):
        yield {'terminal':dict(request_id=row['request_id'],terminal_status='failed' if self.fail else 'completed',reason='transient' if self.fail else 'length',output_token_ids=[] if self.fail else [3],output_tokens=0 if self.fail else 1,input_tokens=len(row['token_ids']),timing={})}
    def close(self): pass

class Recoveries(unittest.TestCase):
    def setup_run(self,tmp):
        rows=[dict(request_id='a',token_ids=[1,2],max_new_tokens=1),dict(request_id='b',token_ids=[4],max_new_tokens=1)]
        root=batch.prepare(rows,'snapshot@1',Config().dict(),'test',{'test':'CPU'},tmp)
        return root,rows,batch.LocalResults(root/'results.db')

    def test_resume_before_staging_and_after_commit_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,rows,sink=self.setup_run(tmp)
            run,_=batch.read_run(root)
            # Interrupted attempt was issued but never staged.
            batch.immutable(root/'issued'/f'{batch.digest(rows[0])}-1.json',{'interrupted':True})
            report=batch.execute(root,sink,ControlledEngine)
            self.assertEqual(report['successful'],2)
            first=sink.records(run['run_id'])
            report=batch.execute(root,sink,lambda: self.fail('resume must skip terminal keys'))
            self.assertEqual(sink.records(run['run_id']),first)
            self.assertEqual(report['successful'],2)
            sink.close()

    def test_after_staging_corrupt_attempt_retry_and_final_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,rows,sink=self.setup_run(tmp); manifest,_=batch.read_run(root)
            records=[]
            for row in rows:
                records.append(dict(request_id=row['request_id'],run_id=manifest['run_id'],attempt_id='first',terminal_status='completed',reason='length',output_token_ids=[3],output_tokens=1,input_tokens=len(row['token_ids']),output_text='3',error_code=None,timing={},provenance=manifest['identity']))
            path=batch.stage(root,manifest,records,'first')
            # No commit yet, resume must consume sealed successes without a new GPU.
            report=batch.execute(root,sink,lambda:self.fail('staged success must be reused'))
            self.assertEqual(report['successful'],2); sink.close()
        with tempfile.TemporaryDirectory() as tmp:
            root,rows,sink=self.setup_run(tmp); manifest,_=batch.read_run(root)
            bad=dict(records[0],run_id=manifest['run_id'])
            batch.stage(root,manifest,[bad],'first'); (root/'staging'/'first.json').write_text('bad bytes')
            report=batch.execute(root,sink,lambda:ControlledEngine(fail=True))
            self.assertEqual(report['failed'],2)
            self.assertTrue(list((root/'validation-errors').glob('*.json')))
            self.assertEqual(batch.execute(root,sink,lambda:self.fail('final failures must be skipped'))['failed'],2)
            sink.close()

class PartialCommit(unittest.TestCase):
    def test_final_failure_partial_commit_then_resume(self):
        class InterruptingSink(batch.LocalResults):
            interrupted=False
            def commit(self,records):
                if records and records[0]['terminal_status']=='failed' and not self.interrupted:
                    self.interrupted=True
                    super().commit(records[:1])
                    raise ConnectionError('commit acknowledgment lost after partial write')
                super().commit(records)
        with tempfile.TemporaryDirectory() as tmp:
            root,rows,initial=Recoveries().setup_run(tmp); initial.close()
            sink=InterruptingSink(root/'results.db')
            with self.assertRaises(ConnectionError): batch.execute(root,sink,lambda:ControlledEngine(fail=True))
            report=batch.execute(root,sink,lambda:self.fail('exhausted keys must not rerun'))
            self.assertEqual(report['failed'],2)
            self.assertEqual(len(sink.records(batch.read_run(root)[0]['run_id'])),2)
            sink.close()

    def test_result_provenance_and_token_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,rows,sink=Recoveries().setup_run(tmp)
            batch.execute(root,sink,ControlledEngine)
            manifest,_=batch.read_run(root)
            record=sink.records(manifest['run_id'])[0]
            for field,value in [('provenance',{}),('output_token_ids',[-1]),('output_tokens',True),('terminal_status','running')]:
                altered=dict(record,**{field:value})
                with self.assertRaises(ValueError): batch.validate_record(altered,rows[0],manifest)
            sink.close()
