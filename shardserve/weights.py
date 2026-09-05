"""Explicit source-layout validation and CPU safetensors slicing before GPU transfer."""
import hashlib
import json
from pathlib import Path
from .config import MODEL, REVISION, validate_model


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(cache=None):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json']))
    config = json.loads((root/'config.json').read_text())
    validate_model(config, 2)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('pinned Instruct tokenizer lacks chat template')
    files = {p.name: digest_file(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'shardserve-model.json'}
    manifest = {'model': MODEL, 'revision': REVISION, 'tokenizer_revision': REVISION, 'files':files}
    (root/'shardserve-model.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return root


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'shardserve-model.json').read_text())
    if (manifest['model'], manifest['revision'], manifest['tokenizer_revision']) != (MODEL,REVISION,REVISION):
        raise ValueError('wrong pinned model identity')
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != 'shardserve-model.json'}
    if actual != set(manifest['files']):
        raise ValueError('model file inventory changed')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest_file(root/name) != expected:
            raise ValueError(f'model checksum mismatch: {name}')
    config = json.loads((root/'config.json').read_text())
    validate_model(config, 2)
    return config


def specs(c):
    h, q, k, m = (c[x] for x in ('hidden_size','num_attention_heads','num_key_value_heads','intermediate_size'))
    d = h//q
    out = {'model.embed_tokens.weight': ((c['vocab_size'],h), None), 'model.norm.weight': ((h,),None)}
    for i in range(c['num_hidden_layers']):
        p = f'model.layers.{i}.'
        for norm in ('input_layernorm','post_attention_layernorm'):
            out[p+norm+'.weight'] = ((h,),None)
        for name, width in (('q',q*d),('k',k*d),('v',k*d)):
            out[p+f'self_attn.{name}_proj.weight'] = ((width,h),0)
            out[p+f'self_attn.{name}_proj.bias'] = ((width,),0)
        out[p+'self_attn.o_proj.weight'] = ((h,h),1)
        for name in ('gate','up'):
            out[p+f'mlp.{name}_proj.weight'] = ((m,h),0)
        out[p+'mlp.down_proj.weight'] = ((h,m),1)
    return out


def shard(tensor, axis, rank, world):
    if world not in (1,2) or not 0 <= rank < world:
        raise ValueError('invalid rank/world')
    if axis is None:
        return tensor[:]
    shape = tensor.get_shape() if hasattr(tensor,'get_shape') else tensor.shape
    if shape[axis] % world:
        raise ValueError('uneven partition')
    width = shape[axis]//world
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(rank*width,(rank+1)*width)
    return tensor[tuple(sl)]


def load(root, c, rank, world, device, dtype):
    import torch
    from safetensors import safe_open
    validate_model(c, world, candidate=False)
    expected, weights, seen = specs(c), {}, set()
    tied = None
    for path in sorted(Path(root).glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in seen:
                    raise ValueError(f'duplicate tensor {key}')
                seen.add(key)
                if key == 'lm_head.weight':
                    tied = f.get_tensor(key)
                    continue
                if key not in expected:
                    raise ValueError(f'unexpected tensor {key}')
                shape, axis = expected[key]
                view = f.get_slice(key)
                if tuple(view.get_shape()) != shape:
                    raise ValueError(f'wrong stored axes/shape: {key}')
                weights[key] = shard(view, axis, rank, world).to(device=device,dtype=dtype)
    if set(expected) - set(weights):
        raise ValueError(f'missing tensors: {set(expected)-set(weights)}')
    if tied is not None:
        # Compare source precision on CPU, not a rounded BF16 device copy.
        for path in sorted(Path(root).glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as f:
                if 'model.embed_tokens.weight' in f.keys() and not torch.equal(tied, f.get_tensor('model.embed_tokens.weight')):
                    raise ValueError('tied output differs from embedding')
    return weights
