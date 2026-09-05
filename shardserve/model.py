"""Engine-owned Qwen forward. HF is used only for tokenizer and verification."""
import torch
import torch.nn.functional as F
from .math import rms_norm, build_rope_cache, apply_rope, causal_attention
from .config import validate_model

class Model:
    def __init__(self, config, weights, world=1, reduce=None, context=4096):
        validate_model(config,world,candidate=False)
        self.c, self.w, self.world = config, weights, world
        if world > 1 and reduce is None:
            raise ValueError('TP2 requires a collective reduction')
        self.reduce = reduce or (lambda x:x)
        self.d = config['hidden_size']//config['num_attention_heads']
        self.cos,self.sin = build_rope_cache(self.d,context,config['rope_theta'],weights['model.norm.weight'].device)

    @torch.inference_mode()
    def forward(self, ids, positions, cache=None, tables=None, seqs=None, slots=None, selected=None, diagnostic=False, trace=None):
        w,c = self.w,self.c
        hidden = F.embedding(ids,w['model.embed_tokens.weight'])
        for layer in range(c['num_hidden_layers']):
            p = f'model.layers.{layer}.'
            x = rms_norm(hidden,w[p+'input_layernorm.weight'],c['rms_norm_eps'])
            q,k,v = [F.linear(x,w[p+f'self_attn.{name}_proj.weight'],w[p+f'self_attn.{name}_proj.bias']).reshape(len(ids),-1,self.d) for name in ('q','k','v')]
            q,k = apply_rope(q,k,self.cos,self.sin,positions)
            if cache is None:
                if not diagnostic:
                    raise ValueError('cache-off execution is diagnostic only')
                group = q.shape[1]//k.shape[1]
                attention = causal_attention(q,k.repeat_interleave(group,1),v.repeat_interleave(group,1),self.d**-0.5)
            else:
                cache.write(layer,k,v,slots)
                if diagnostic:
                    attention = cache.diagnostic(layer,q,tables,seqs,positions)
                else:
                    from .kernel import ragged_attention_direct
                    attention = ragged_attention_direct(q,cache.k[layer],cache.v[layer],tables,seqs,positions,self.d**-0.5)
                attention = attention.flatten(1)
            # Only the partial projection is reduced; replicated residual is added once.
            hidden = hidden + self.reduce(F.linear(attention,w[p+'self_attn.o_proj.weight']))
            x = rms_norm(hidden,w[p+'post_attention_layernorm.weight'],c['rms_norm_eps'])
            gate = F.silu(F.linear(x,w[p+'mlp.gate_proj.weight']))
            up = F.linear(x,w[p+'mlp.up_proj.weight'])
            hidden = hidden + self.reduce(F.linear(gate*up,w[p+'mlp.down_proj.weight']))
            if trace is not None:
                trace.append(hidden.detach().float().cpu())
        hidden = rms_norm(hidden,w['model.norm.weight'],c['rms_norm_eps'])
        if selected is not None:
            hidden = hidden.index_select(0,selected)
        # Reuse tied BF16 weight; do not materialize a replicated FP32 vocabulary matrix.
        return F.linear(hidden,w['model.embed_tokens.weight']).float()
