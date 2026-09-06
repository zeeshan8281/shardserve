# Implementation and evidence plan

The PRD remains the release contract. The GPU core and the split Modal GPU/Databricks serverless data plane have live evidence; performance acceptance is incomplete.

## P0 audit

Source inspected: `zeeshan8281/cloud-inference-from-scratch`, exact commit `174715839aa256a2010b21a796da716cae1a46f4`, cloned to `/tmp/shardserve-reference`. The original working service was not touched. Source is MIT; its copyright and permission notice are preserved in `LICENSE`.

Inspected `model.py`, `weights.py`, `attention.py`, `kernel.py`, `cache.py`, scheduler interfaces and allocator/ragged tests. Reused RMSNorm/RoPE/causal attention math in `shardserve/math.py` and the direct paged ragged Triton kernel in `shardserve/kernel.py`. The BF16 extension passed the two-L4 preflight and full-model checks. Scheduler tests and allocation concepts informed new focused tests. The original loader stages a full CPU state dictionary and is single-GPU; the new loader uses safetensors slices before device transfer. No API, UI, deployment, quantization, prefix cache or original service infrastructure was imported.

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
| P0 | Source/license audit, checkpoint metadata, environment manifest, full checkpoint execution and Modal GPU identity | Databricks identity |
| P1 | Custom sharded model; FP32 algebra; repeated BF16 TP1/TP2 layer/logit checks; committed envelope; exact seven-prompt greedy corpus | Complete on tested L4 hardware |
| P2 | Paged pool, distributed plan path, mixed/chunked regressions, real Triton/NCCL concurrent workload and per-rank state/memory | Complete on tested L4 hardware |
| P3 | Cancellation/deadline/backpressure supervisor plus before/after-all-reduce and external-kill evidence | Complete on tested L4 hardware |
| P4 | Graph/eager equality, 1/2/4/8 transitions, cancellation, 29 graph replays | Preserve both-rank profiler traces in a performance run |
| P5 | Versioned Delta preparation, attempt sealing, insert-only single-writer commit, Databricks serverless Delta recovery, Unity Catalog volume persistence, corruption rejection and MLflow upload | Actual classic GPU job on a workspace that permits classic compute |
| P6 | Deterministic workload and bounded HTTP comparison driver | All five matched comparisons, at least three repeats, raw GPU records, profiler traces, actual cost |

GPU evidence was collected on 2026-09-06 using two co-located Modal NVIDIA L4 GPUs with 23,034 MiB each. Raw calibration and verified reports, hardware identities, graph results and fault timings are committed under `evidence/gpu`. Model weights remain in a Modal Volume and were not copied into this repository. Databricks job `655087961621636` and Modal app `ap-ApmndI2GsQAOjZetqibt7X` passed the split data-plane flow; evidence is under `evidence/databricks`. The connected Free Edition workspace cannot create classic compute, and serverless GPU run `584842402884687` reported exhausted A10 quota. P6 still needs a separate bounded performance allocation.

## Known verification gaps

The seven-prompt full-model corpus passed exact greedy equality on the tested L4s; its numerical envelope is hardware-specific. Both-rank profiler traces, the full matched benchmark matrix, actual billed cost, live vLLM adapter behavior and classic Databricks GPU execution remain unverified. There is no native managed serving deployment or managed TP-server support claim.
