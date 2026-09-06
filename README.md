# ShardServe

Custom Qwen tensor-parallel inference with coordinated continuous batching, paged KV storage, and a bounded Databricks batch adapter.

**Status: GPU core and Databricks Delta/MLflow integration verified; classic Databricks GPU and performance acceptance remain.** CPU checks pass locally. Full-model BF16 TP1/TP2 correctness, NCCL, the Triton paged-attention kernel, CUDA Graphs, cancellation, and injected rank failures passed on two Modal NVIDIA L4 GPUs. Delta recovery, corruption rejection, duplicate replay, Unity Catalog volume persistence, and MLflow upload passed in Databricks serverless. See [phase evidence and limitations](docs/implementation.md).

The runtime owns weight partitioning, forward execution, allocation, iteration plans and sampling. It does not call Hugging Face `generate()`, vLLM or a managed model endpoint. Hugging Face supplies tokenization/download tooling and verification-only reference forward execution. MIT RoPE/RMSNorm math and the paged Triton kernel are adapted from [cloud-inference-from-scratch at 1747158](https://github.com/zeeshan8281/cloud-inference-from-scratch/tree/174715839aa256a2010b21a796da716cae1a46f4); see `LICENSE`.

## Local setup and checks

Run commands from this repository root. Local CPU checks were exercised with Python 3.9.6, Torch 2.8.0, Transformers 4.57.6, safetensors 0.7.0 and huggingface-hub 0.36.2. The pinned Linux CUDA stack was exercised on Modal L4 hardware.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m shardserve env --output artifacts/environment.json
.venv/bin/python -m shardserve check --output artifacts/cpu-tests.txt
```

On an authorized CUDA Linux node, install `pip install -e '.[cuda]'`. Databricks tracking additionally needs `pip install -e '.[databricks]'`; Spark/Delta come from the actual classic Databricks runtime. Do not assume its preinstalled Torch stack matches this project. Save `pip freeze`, runtime identity, GPU inventory and preflight results.

`check` runs real tiny FP32 custom operations, source-axis reassembly, a Hugging Face reference comparison, concurrent two-rank CPU partition algebra, cache-boundary checks and control/recovery tests. HTTP tests bind an ephemeral loopback port. GPU acceptance is explicitly skipped unless two CUDA GPUs and `SHARDSERVE_MODEL` are available. CPU host reductions are not NCCL validation; HTTP and persistence stubs are not inference evidence.

## GPU preflight and pinned model

Only TP1 and TP2 are supported, on one node. Default runtime precision is BF16. TP2 owns 8 query heads and 1 KV head per rank. Full checkpoint weights have not been downloaded or executed on the development Mac.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m shardserve.preflight --output artifacts/gpu-preflight.json
python -m shardserve fetch-model --cache models
```

`fetch-model` prints the snapshot directory. Use that exact path as `MODEL_SNAPSHOT` below. The model is `Qwen/Qwen2.5-3B-Instruct` at `14d7620ba47cf51be0b176e14e27e38a34d4ff88`; an unavailable revision fails. Downloaded files are checksummed and verified before group startup. The Instruct chat template is applied to text prompts; token-ID inputs are used unchanged.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m shardserve.verify --model MODEL_SNAPSHOT --output artifacts/tp1-calibration.json
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m shardserve.verify --model MODEL_SNAPSHOT --output artifacts/tp2-calibration.json
```

These commands deliberately return a nonzero status while no numerical envelope exists, but write raw error records after successful execution. The committed [BF16 envelope](evidence/gpu/bf16-envelope.json) was derived 25% above three repeated TP1/TP2 measurements. The expanded seven-prompt corpus then passed exact greedy equality in both verified reruns. New hardware must recalibrate rather than silently loosen this envelope.

```bash
SHARDSERVE_TRACE_DIR=artifacts/traces python -m shardserve.gpu_checks graphs \
  --model MODEL_SNAPSHOT --output artifacts/gpu-graphs.json
python -m shardserve.gpu_checks fault \
  --model MODEL_SNAPSHOT --output artifacts/gpu-faults.json
```

Graphs are opt-in, with buckets 1/2/4/8, private padding slots and eager fallback. Trace output is bounded to eight active iterations per rank. The graph check passed TP1, TP2 eager and TP2 graph output equality, batch-size transitions, cancellation, and 29 graph replays on Modal. The fault harness passed deliberate exits before/after all-reduce and an external kill after stream output; shutdown took at most 1.7 seconds. Evidence is under [`evidence/gpu`](evidence/gpu). Do not set `SHARDSERVE_TEST_CRASH` during normal execution; it is a fault-test hook.

## Private API

```bash
python -m shardserve serve --model MODEL_SNAPSHOT --world-size 2 --port 8080
curl http://127.0.0.1:8080/health
curl -N http://127.0.0.1:8080/stream \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"example-1","prompt":"Explain tensor parallelism briefly.","max_new_tokens":32}'
```

`POST /generate` blocks for the terminal result. `POST /stream` emits ordered token events and one terminal event. `DELETE /requests/ID` requests cancellation at an iteration boundary. `/health` and `/metrics` expose bounded state and per-rank memory/counters. Requests accept exactly one of `prompt` or `token_ids`, plus an output limit and optional Unix-time `deadline`. Unsupported sampling fields, duplicate IDs, invalid tokens and excessive context are rejected. IDs cannot be reused until group restart. Body limit is 64 KiB; prompt content is not logged.

The default address is loopback. Binding elsewhere requires `SHARDSERVE_API_TOKEN`; clients pass `Authorization: Bearer TOKEN`. Public hosting, TLS termination and production exposure are out of scope. A slow/disconnected stream is terminated without indefinitely blocking other requests. Context is capped at 4096, live requests at eight. KV admission reserves the full request capacity; no preemption or prefix reuse is enabled.

## Workloads and measurements

```bash
python -m shardserve workload --output artifacts/workload.json \
  --prompt-tokens 128 --output-tokens 32 --count 100 --seed 9281
python -m shardserve benchmark --workload artifacts/workload.json \
  --urls http://127.0.0.1:8080 --hardware hardware.json \
  --concurrency 4 --output artifacts/tp2-eager
```

`hardware.json` must describe actual reserved hardware, not an assumed two-GPU allocation. The driver performs equal warmup per target and three measured repeats. It stores all raw request records, failures, TTFT, per-request mean TPOT, latency, useful output throughput and goodput under predeclared thresholds. It reports p50/p95 with at least 100 requests per repeat. `--rate REQUESTS_PER_SECOND` selects offered-load experiments; overload drops are reported separately from finite-batch work. `--ttft-slo` and `--tpot-slo` set thresholds before running.

Run the 128/1024/3072 prompt × 32/128 output × concurrency 1/4/8 matrix, distinct and `--repeated-prefix`, with prefix reuse disabled. Preserve every configuration, including losses and capacity failures. For two TP1 replicas, start one server per GPU on different ports and pass both `--urls`; the dispatcher chooses the least-inflight replica. Compare against TP2 on those same two physical GPUs. Both must disclose any additional GPUs reserved by the node.

The driver has a `--backend vllm` HTTP adapter targeting pinned vLLM 0.10.1 `/v1/completions` with identical token IDs, greedy sampling and output caps. That adapter is unverified against a live vLLM server. It requests logprobs to count stream tokens, which adds measurement overhead and must be disclosed; do not claim a fair performance comparison until that overhead and stopping behavior are validated. Install vLLM in a separate environment to avoid altering the custom engine stack. No benchmark numbers are published yet.

## Durable batch jobs

For local protocol checks, use a JSON token workload, an actual hardware description and SQLite:

```bash
python -m shardserve prepare --workload artifacts/workload.json \
  --hardware hardware.json --directory artifacts/runs
python -m shardserve batch artifacts/runs/RUN_ID \
  --model MODEL_SNAPSHOT --results artifacts/results.sqlite
python -m shardserve resume artifacts/runs/RUN_ID \
  --model MODEL_SNAPSHOT --results artifacts/results.sqlite
python -m shardserve validate artifacts/runs/RUN_ID --results artifacts/results.sqlite
python -m shardserve upload artifacts/runs/RUN_ID --experiment /Shared/ShardServe
```

`RUN_ID` is printed by preparation. Resume reuses its persisted manifest and rejects changed code. Successful staging is validated before insert-only commit. Attempts remain observable; final failures are committed by the coordinator and skipped on resume. SQLite checks do not satisfy the Databricks gate.

The classic GPU job protocol and the executed serverless Delta/MLflow evidence are documented in [Databricks execution](docs/databricks.md). The connected Free Edition workspace cannot create classic compute, and its serverless A10 request reported exhausted GPU quota. No native managed serving endpoint exists, and managed TP custom-server compatibility is not claimed.
