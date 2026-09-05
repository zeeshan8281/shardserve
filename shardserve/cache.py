"""One physical K/V pool per rank; logical reservations are all-or-nothing.
Full request capacity is reserved at admission: no recompute preemption in v1.
"""
import math

class Blocks:
    def __init__(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('positive block count required')
        self.count = count
        self.free = list(range(count))
        self.tables = {}

    def reserve(self, requests):
        if any(key in self.tables for key in requests):
            raise ValueError('duplicate reservation')
        if any(type(n) is not int or n < 1 for n in requests.values()):
            raise ValueError('invalid reservation')
        sizes = {key: math.ceil(n/16) for key,n in requests.items()}
        if sum(sizes.values()) > len(self.free):
            raise MemoryError('KV capacity exhausted')
        for key,n in sizes.items():
            self.tables[key] = [self.free.pop() for _ in range(n)]
        self.check()

    def release(self, key):
        if key not in self.tables:
            raise ValueError('double/foreign release')
        self.free.extend(self.tables.pop(key))
        self.check()

    def check(self):
        owned = [b for table in self.tables.values() for b in table]
        assert sorted(owned+self.free) == list(range(self.count)), 'leak/duplicate ownership'


class Cache:
    def __init__(self, c, world, blocks, device, dtype):
        import torch
        self.blocks = Blocks(blocks)
        self.shape = (c['num_hidden_layers'],blocks+8,16,c['num_key_value_heads']//world,c['hidden_size']//c['num_attention_heads'])
        # Last eight blocks are exclusive graph-padding scratch, never logical KV.
        self.k = torch.zeros(self.shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.bytes = 2*self.k.numel()*self.k.element_size()

    def write(self, layer, k, v, slots):
        self.k[layer].flatten(0,1).index_copy_(0,slots,k)
        self.v[layer].flatten(0,1).index_copy_(0,slots,v)

    def diagnostic(self, layer, q, tables, seqs, positions):
        import torch
        result = []
        for query, seq, pos in zip(q,seqs.tolist(),positions.tolist()):
            indices = torch.arange(pos+1, device=q.device)
            slots = tables[seq, indices//16].long()*16+indices%16
            k = self.k[layer].flatten(0,1)[slots].repeat_interleave(q.shape[1]//self.shape[3],1)
            v = self.v[layer].flatten(0,1)[slots].repeat_interleave(q.shape[1]//self.shape[3],1)
            scores = torch.einsum('hd,thd->ht',query.float(),k.float())/q.shape[-1]**0.5
            result.append(torch.einsum('ht,thd->hd',scores.softmax(-1).to(v.dtype),v))
        return torch.stack(result)
