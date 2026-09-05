# Implementation and evidence plan

The PRD remains the release contract. This repository is an implementation under verification, not a completed GPU engine release.

## P0 audit

Source inspected: `zeeshan8281/cloud-inference-from-scratch`, exact commit `174715839aa256a2010b21a796da716cae1a46f4`, cloned to `/tmp/shardserve-reference`. The original working service was not touched. Source is MIT; its copyright and permission notice are preserved in `LICENSE`.

Inspected `model.py`, `weights.py`, `attention.py`, `kernel.py`, `cache.py`, scheduler interfaces and allocator/ragged tests. Reused RMSNorm/RoPE/causal attention math in `shardserve/math.py` and the direct paged ragged Triton kernel in `shardserve/kernel.py`. The kernel wrapper now accepts BF16; that extension needs CUDA validation. Scheduler tests and allocation concepts informed new focused tests. The original loader stages a full CPU state dictionary and is single-GPU; the new loader uses safetensors slices before device transfer. No API, UI, deployment, quantization, prefix cache or original service infrastructure was imported.

Selected model: `Qwen/Qwen2.5-3B-Instruct` revision `14d7620ba47cf51be0b176e14e27e38a34d4ff88`. Actual pinned config/tokenizer/generation/index files were downloaded and checksummed. `evidence/checkpoint-metadata.json` records the config and chat-template test. No full weights have been downloaded here. Default Instruct generation settings are stochastic; ShardServe explicitly overrides those with greedy argmax. EOS IDs come from pinned generation config.

## Runtime decisions

* Safetensors layout is `[out,in]`: Q/K/V and gate/up split output rows, attention O and MLP down split input columns. Each rank has 8 query heads and 1 KV head at TP2. All geometry is validated. Normalization, embeddings and output weights are replicated. The output shares the embedding tensor and does not create an FP32 vocabulary copy. The BF16 embedding alone costs 622,329,856 bytes per rank; TP2 does not halve all model memory.
* One parent scheduler owns state. Rank 0 receives numbered plans and broadcasts them. Plans contain ordered IDs, positions, lengths, offsets, reserves, releases and eager/decode mode. All ranks reduce allocation success before execution, broadcast rank-0 token choices, and compare logical live-state summaries after execution.
* FIFO admission reserves each request's full prompt-plus-output capacity. Existing decode runs before prefill chunks. An oversized request is rejected; otherwise admission waits for capacity. No prefix sharing or recompute preemption. Conservative fixed KV sizing includes eight private padding blocks; CUDA allocation failure aborts the group. A future memory-budget tuner must be driven by actual per-rank measurements, not an assumed division by TP degree.
* Terminal state is immutable. Cancellation during forward becomes effective when that iteration returns. KV release occurs at the next boundary. Full output buffers fail the slow request. Terminal request objects leave the scheduler; a bounded 100,000-ID tombstone ledger prevents retry duplication until group restart. Each request can have one consumer.
* The parent detects process exit or watchdog expiry and terminates all ranks. Default watchdog is 60 seconds, with up to 4 seconds kill/join cleanup; startup separately allows 600 seconds. Fresh groups have fresh rendezvous and caches. No live-stream replay.
* CUDA decode graphs use independent fixed buffers for buckets 1/2/4/8, padded with distinct non-request scratch blocks. Prefill and unsupported buckets use eager execution. Capture is opt-in until BF16 correctness and NCCL graph compatibility are actually established. Per-rank counters/memory and optional Chrome traces are exposed.

## Phase order and gates

| Phase | Local implementation/checks | Remaining mandatory evidence |
|---|---|---|
| P0 | Source/license audit, checkpoint metadata, environment manifest | Full checkpoint files; allocated GPU/Databricks identity |
| P1 | Custom sharded model, source-axis reassembly, real tiny FP32 TP algebra and HF comparison | Real BF16 TP1/TP2, layer/logit errors, committed numerical envelope, exact greedy corpus |
| P2 | Paged pool, distributed plan path, mixed/chunked CPU regressions | Real Triton/NCCL concurrent workload, capacity agreement/memory on all ranks |
| P3 | Boundary cancellation, deadlines, backpressure, bounded supervisor, fault harness | Observed kill/collective shutdown timings and restart reference output |
| P4 | Bounded graph capture/replay implementation and harness | Actual graph/eager equality, both-rank traces, no stale reads during transitions |
| P5 | Versioned Delta preparation, attempt sealing, insert-only single-writer commit, local recovery checks | Actual classic GPU job, durable Delta recovery, platform filesystem and MLflow upload verification |
| P6 | Deterministic workload and bounded HTTP comparison driver | All five matched comparisons, at least three repeats, raw GPU records, profiler traces, actual cost |

All GPU/platform gates are **UNVERIFIED/BLOCKED** here. There is no committed BF16 envelope: inventing one would bypass P1. The correctness command emits raw calibration records and fails the gate unless an evidence-backed envelope is supplied. CPU parallel checks use real tensor/model operations and synchronized host reductions, not NCCL. Persistence and HTTP stubs test only their control boundaries.

## Known verification gaps

GPU harnesses have not been executed. Full-model reference checks currently include a small corpus and must be expanded/committed with measured BF16 stability before performance conclusions. HF/eager cached/direct-kernel comparisons report fixed-prefix errors; near ties do not waive greedy differences. GPU hardware differences, memory admission peaks, graph/NCCL compatibility, and Databricks FUSE/fsync semantics must be validated in preflight. There is no native managed serving deployment or managed TP-server support claim.
