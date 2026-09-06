#!/usr/bin/env python3
"""Publish a small Git corpus as an immutable Elasticsearch index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urlparse

from shardserve.retrieval import ElasticConfig, ElasticSearch


EXTENSIONS={'.py','.md','.json','.toml','.yml','.yaml'}
EXCLUDED={'.git','.venv','.cache','artifacts','models','__pycache__'}


def git_files(root):
    result=subprocess.run(['git','-C',str(root),'status','--porcelain'],capture_output=True,text=True,check=True)
    if result.stdout.strip(): raise ValueError('refuse to publish a dirty source tree')
    revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    raw=subprocess.run(['git','-C',str(root),'ls-files','-z'],capture_output=True,check=True).stdout
    return revision,[root/part.decode() for part in raw.split(b'\0') if part]


def chunks(root, files, repository, revision, source_url_base, lines_per_chunk=120, overlap=20):
    if not 20<=lines_per_chunk<=500 or not 0<=overlap<lines_per_chunk: raise ValueError('invalid chunk bounds')
    total=0
    for path in files:
        relative=path.relative_to(root)
        if path.suffix.lower() not in EXTENSIONS or any(part in EXCLUDED for part in relative.parts): continue
        data=path.read_bytes()
        if len(data)>512000: continue
        total+=len(data)
        if total>20_000_000: raise ValueError('corpus exceeds 20 MB source bound')
        try: text=data.decode('utf-8')
        except UnicodeDecodeError: continue
        rows=text.splitlines(keepends=True)
        step=lines_per_chunk-overlap
        for start in range(0,len(rows),step):
            selected=rows[start:start+lines_per_chunk]
            content=''.join(selected).strip()
            if not content: continue
            end=start+len(selected)
            name=f'{relative.as_posix()}:L{start+1}-L{end}'
            checksum=hashlib.sha256(content.encode()).hexdigest()
            url=source_url_base.rstrip('/')+'/'+quote(relative.as_posix(),safe='/')+f'#L{start+1}-L{end}'
            yield {'elastic_id':hashlib.sha256(f'{repository}\0{revision}\0{name}\0{checksum}'.encode()).hexdigest(),
                'doc_id':name,'content':content,'content_sha256':checksum,'source_url':url,
                'repository':repository,'revision':revision,'path':relative.as_posix(),
                'line_start':start+1,'line_end':end}
            if end==len(rows): break


def publish(client, index, alias, documents, inference_id=None, batch_size=100):
    properties={
        'doc_id':{'type':'keyword'},'content':{'type':'text'},'content_sha256':{'type':'keyword'},
        'source_url':{'type':'keyword','index':False},'repository':{'type':'keyword'},
        'revision':{'type':'keyword'},'path':{'type':'keyword'},'line_start':{'type':'integer'},
        'line_end':{'type':'integer'},
    }
    if inference_id:
        properties['content']['copy_to']='content_semantic'
        properties['content_semantic']={'type':'semantic_text','inference_id':inference_id}
    path='/'+quote(index,safe='')
    if client.request('HEAD',path,allow=(404,)) is None:
        client.request('PUT',path,{'mappings':{'dynamic':'strict','properties':properties}})
    count=0; batch=[]
    def send(rows):
        body=b''
        for row in rows:
            source=dict(row); elastic_id=source.pop('elastic_id')
            body+=json.dumps({'index':{'_index':index,'_id':elastic_id}},separators=(',',':')).encode()+b'\n'
            body+=json.dumps(source,separators=(',',':'),ensure_ascii=False).encode()+b'\n'
        response=client.request('POST','/_bulk',body,'application/x-ndjson')
        if response.get('errors'): raise RuntimeError('Elasticsearch rejected one or more corpus documents')
        if len(response.get('items',[]))!=len(rows): raise RuntimeError('Elasticsearch bulk response count mismatch')
    for document in documents:
        batch.append(document); count+=1
        if len(batch)==batch_size: send(batch); batch=[]
    if batch: send(batch)
    if not count: raise ValueError('source produced no indexable documents')
    actual=client.request('GET','/'+quote(index,safe='')+'/_count').get('count')
    if actual!=count: raise RuntimeError(f'Elasticsearch document count mismatch: expected {count}, got {actual}')
    old=client.request('GET','/_alias/'+quote(alias,safe=''),allow=(404,))
    actions=[{'remove':{'index':name,'alias':alias}} for name in sorted(old or {})]
    actions.append({'add':{'index':index,'alias':alias}})
    client.request('POST','/_aliases',{'actions':actions})
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='.')
    parser.add_argument('--repository',default='shardserve')
    parser.add_argument('--alias',default=os.environ.get('ELASTIC_INDEX','shardserve-docs-live'))
    parser.add_argument('--inference-id',help='pin a semantic_text inference endpoint; omit for BM25-only')
    parser.add_argument('--source-url-base')
    parser.add_argument('--manifest-output')
    args=parser.parse_args()
    root=Path(args.source).resolve(); revision,files=git_files(root)
    base=args.source_url_base or f'https://github.com/zeeshan8281/{args.repository}/blob/{revision}'
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,200}',args.alias): raise ValueError('invalid index alias')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',args.repository): raise ValueError('invalid repository name')
    parsed=urlparse(base)
    if parsed.scheme!='https' or not parsed.netloc: raise ValueError('source URL base must use HTTPS')
    index=(args.alias.removesuffix('-live')+'-v'+revision[:12]).lower()
    config=ElasticConfig.from_env(); client=ElasticSearch(config)
    count=publish(client,index,args.alias,chunks(root,files,args.repository,revision,base),args.inference_id)
    result={'index':index,'alias':args.alias,'repository':args.repository,'revision':revision,
        'documents':count,'semantic':bool(args.inference_id),'inference_id':args.inference_id}
    if args.manifest_output:
        path=Path(args.manifest_output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
