from dataclasses import dataclass, asdict
import math

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"

@dataclass(frozen=True)
class Config:
    world_size: int = 2
    context: int = 4096
    max_live: int = 8
    queue_size: int = 64
    output_buffer: int = 256
    token_budget: int = 256
    prefill_chunk: int = 128
    kv_blocks: int = 2048
    graphs: bool = False
    watchdog: float = 60.0
    startup_timeout: float = 600.0
    diagnostic: bool = False

    def __post_init__(self):
        if type(self.world_size) is not int or self.world_size not in (1, 2):
            raise ValueError("only TP1 and TP2 are supported")
        for name in ('context', 'max_live', 'queue_size', 'output_buffer', 'token_budget', 'prefill_chunk', 'kv_blocks'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.context > 4096 or self.max_live > 8 or self.token_budget < self.max_live:
            raise ValueError("context<=4096, max_live<=8, token_budget>=max_live required")
        if any(not math.isfinite(v) or v <= 0 for v in (self.watchdog, self.startup_timeout)):
            raise ValueError("timeouts must be finite and positive")
        if self.graphs and self.diagnostic:
            raise ValueError("graphs require the CUDA kernel")

    def dict(self):
        return asdict(self)


def validate_model(c, world, candidate=True):
    if world not in (1, 2):
        raise ValueError("unsupported TP degree")
    h, q, k, m = (c[x] for x in ('hidden_size','num_attention_heads','num_key_value_heads','intermediate_size'))
    if any(type(x) is not int or x <= 0 for x in (h,q,k,m,c['num_hidden_layers'],c['vocab_size'])):
        raise ValueError("invalid model dimensions")
    if h % q or q % k or any(x % world for x in (h,q,k,m)) or (h//q) % 2:
        raise ValueError("non-partitionable dimensions")
    if c['model_type'] != 'qwen2' or c.get('hidden_act') != 'silu' or c.get('rope_scaling') or c.get('use_sliding_window'):
        raise ValueError("only Qwen2 SwiGLU with standard RoPE/full attention supported")
    if not c.get('tie_word_embeddings'):
        raise ValueError("v1 requires tied embedding/output")
    if candidate and (h,q,k,m,c['num_hidden_layers'],c['vocab_size']) != (2048,16,2,11008,36,151936):
        raise ValueError("checkpoint does not match pinned candidate geometry")
