import os
import unittest
import torch

@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count()>=2 and os.environ.get('SHARDSERVE_MODEL'), 'requires two CUDA GPUs and SHARDSERVE_MODEL pinned local checkpoint')
class GPUAcceptance(unittest.TestCase):
    def test_graphs_and_fault_lifecycle(self):
        from shardserve.gpu_checks import compare,fault
        root=os.environ['SHARDSERVE_MODEL']
        compare(root,'artifacts/gpu-graphs.json')
        fault(root,'artifacts/gpu-faults.json')
