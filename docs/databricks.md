# Classic Databricks GPU job protocol

Status: the Delta/MLflow adapter executed successfully on Databricks serverless on 2026-09-06. Run `418965407352167` verified an immutable Delta snapshot, Unity Catalog volume persistence, insert-only duplicate replay, corrupt-stage rejection, one authoritative terminal row, and MLflow run `2dc0319ddf724ad3af6be3188f99a89d`. This is a Free Edition workspace: classic allocation was rejected with `Only serverless compute is supported in the workspace`, and serverless GPU run `584842402884687` reported `RESOURCE_EXHAUSTED: GPU quota exhausted for GPU_1xA10`. The classic GPU protocol below remains unexecuted.

Use one authorized classic single-node GPU job, one rank group and one committer. The current Databricks GPU guide supports single-node GPU compute, but that is not validation of this Torch/Triton/NCCL stack. GPU scheduling of Spark tasks is not TP. Do not use the AI Runtime adapter or managed serving instructions as substitutes.

Before allocation record workspace cloud/region, node type, runtime/image identity, full reserved GPU count, allowed TP rank mapping, maximum job duration, currency/maximum total spend and shutdown owner. Use a bounded job timeout and termination after the authorized job. A node with 4/8 reserved GPUs must be charged as 4/8 even though only two ranks execute. Put actual values in `hardware.json`; unknown cost is null/unavailable, never zero. Credentials stay in Databricks secret scopes or supported CLI authentication, never manifests.

Install the pinned environment from README on the authorized node. Confirm `python -m shardserve env` plus a clean two-rank `shardserve.preflight` exit before loading weights. Validate durable-volume file creation, fsync, sealed-file checksums and restart visibility. A failed filesystem check blocks this adapter; do not fall back to ephemeral driver storage for authoritative artifacts.

Prepare a fixed Delta snapshot outside the model loop. Input columns are `request_id STRING`, `token_ids ARRAY<INT>`, `max_new_tokens INT`. Normalize text once with the pinned Instruct chat template before writing that snapshot. Preparation rejects duplicate IDs and snapshots exceeding the v1 bound of 100,000 rows, then writes immutable input shards of up to 64 rows. The GPU task never builds a model inside a Spark row/task.

```bash
python -m shardserve.databricks prepare \
  --input-table catalog.schema.inputs --version 7 \
  --hardware hardware.json --directory /Volumes/catalog/schema/artifacts/shardserve

python -m shardserve.databricks run \
  --directory /Volumes/catalog/schema/artifacts/shardserve/RUN_ID \
  --results-table catalog.schema.results --model MODEL_SNAPSHOT

python -m shardserve.databricks resume \
  --directory /Volumes/catalog/schema/artifacts/shardserve/RUN_ID \
  --results-table catalog.schema.results --model MODEL_SNAPSHOT

python -m shardserve.databricks validate \
  --directory /Volumes/catalog/schema/artifacts/shardserve/RUN_ID \
  --results-table catalog.schema.results

python -m shardserve upload /Volumes/catalog/schema/artifacts/shardserve/RUN_ID \
  --experiment /Shared/ShardServe
```

`RUN_ID` and `MODEL_SNAPSHOT` above are the actual outputs of prepare and fetch-model. Code content digest, runtime configuration, generation/model/tokenizer identity, input snapshot/digest, hardware identity and maximum attempts are part of the run identity. Changed code/settings require new preparation. A resume checks the persisted identity and skips all terminal keys, including final failures.

GPU tasks write issued-attempt markers before work and sealed immutable output files afterward. An interrupted issued attempt remains observable and consumes an attempt. Staging checksums, keys, status and token accounting are validated before a success can be committed. Corrupt files are retained with immutable validation-error records and cannot create successful rows. Only the coordinator finalizes exhausted requests as failed. Default maximum is two attempts; no exactly-once execution claim is made.

The authoritative table uses a stable `(run_id, request_id)` key. Under the required single-committer job protocol, Delta `MERGE ... WHEN NOT MATCHED INSERT` preserves the first result and never overwrites it. Concurrent committers are unsupported: use job concurrency one, a single commit task, and the artifact lock. A hard-killed task may leave `committer.lock`; verify that its job/process is dead before removing that one lock and resuming. Never remove a live lock. No streaming checkpoint or transaction-app/version pair is used.

The local SQLite sink exercises insert-only recovery semantics but is not evidence for Delta transactions. Databricks acceptance must repeat interruption before staging, after staging, and after commit-before-ack, inject corrupt output, resume, and verify exactly one authoritative terminal row for every valid input. Preserve all attempts/retries, driver/GPU startup, input-to-commit makespan and actual billed shape in the report.

MLflow upload is separate and retryable: run identity and durable results do not depend on tracking success. `tracking-upload.json` records upload status or error and the retry command. Do not publish successful MLflow logging until the command succeeds on the actual workspace.

The executed serverless evidence is under `evidence/databricks`. It independently queries `workspace.shardserve.results` through the serverless SQL warehouse and confirms one row and one distinct request for run `79fece2b99c9f2fc89393a8b8537f856b7428ee335858fbb8abfbd965998b80a`; it also preserves the serverless GPU quota error. This validates Delta, durable files and MLflow, but does not substitute for the classic GPU task above.

Sources checked 2026-09-06:
- https://docs.databricks.com/aws/en/compute/gpu
- https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
- https://docs.databricks.com/aws/en/structured-streaming/delta-lake
- https://mlflow.org/docs/latest/ml/tracking/tracking-api/
