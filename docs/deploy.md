# Deploy ShardServe IR

This is the shortest deployment path for the retrieval-aware API. Model weights remain in the existing Modal Volume, the search index remains in Elastic Cloud, and Databricks keeps the durable batch/evaluation plane. Nothing here downloads model weights or Elasticsearch data locally.

## 1. Create the Elastic deployment

Create an Elastic Cloud Hosted deployment or trial and copy its Elasticsearch HTTPS endpoint. In Kibana, create an API key that can create and manage `shardserve-docs-*` indices while publishing the corpus.

Export the values only in the current shell:

```bash
export ELASTICSEARCH_URL='https://YOUR-DEPLOYMENT.es.REGION.PROVIDER.elastic.cloud'
export ELASTIC_API_KEY='YOUR-PUBLISHER-API-KEY'
export ELASTIC_INDEX='shardserve-docs-live'
```

Do not put these values in `.env`, Git, shell history, screenshots, or evidence files.

## 2. Publish the repository corpus

Commit the exact source revision first. The publisher refuses a dirty tree because the Git revision is part of every document identity.

For BM25 plus semantic retrieval on an Elastic trial:

```bash
python3 -m deploy.elastic.bootstrap \
  --repository shardserve \
  --inference-id .elser-2-elasticsearch \
  --manifest-output evidence/elastic/index-publication.json
```

This creates an immutable index such as `shardserve-docs-v0123456789ab`, checks every bulk response and the final document count, then atomically moves `shardserve-docs-live` to it. It indexes only tracked `.py`, `.md`, `.json`, `.toml`, `.yml`, and `.yaml` files. Individual files are capped at 512 KB and the total source corpus at 20 MB.

Semantic publication uses batches of eight so a Serverless inference endpoint can process them within the bounded request timeout. Use `--batch-size` only when the deployment has measured capacity for a different value.

If the selected Elastic deployment does not provide the pinned semantic endpoint, omit `--inference-id` and deploy with `ELASTIC_SEARCH_MODE=lexical`. Do not configure hybrid mode against a BM25-only index.

After publication, create a separate runtime API key with read access to `shardserve-docs-*`. Replace the shell value with that narrower key before creating the Modal secret.

## 3. Create the Modal service secret

Generate a long application token and place the runtime credentials in Modal:

```bash
export SHARDSERVE_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export ELASTIC_SEARCH_MODE='hybrid'

modal secret create shardserve-service \
  SHARDSERVE_API_TOKEN="$SHARDSERVE_API_TOKEN" \
  ELASTICSEARCH_URL="$ELASTICSEARCH_URL" \
  ELASTIC_API_KEY="$ELASTIC_API_KEY" \
  ELASTIC_INDEX="$ELASTIC_INDEX" \
  ELASTIC_SEARCH_MODE="$ELASTIC_SEARCH_MODE"
```

Keep `SHARDSERVE_API_TOKEN` in a password manager. It is required by `/health`, `/generate`, `/stream`, `/answer`, and cancellation routes.

## 4. Deploy the two-L4 service

```bash
modal deploy deploy/modal/service.py
```

Modal prints the persistent HTTPS URL. The first request can be slow while the container starts and the pinned Qwen snapshot is loaded from the existing `shardserve-models` Volume. The service scales to zero after five idle minutes.

Set the returned URL locally:

```bash
export SHARDSERVE_URL='https://YOUR-WORKSPACE--shardserve-ir-api.modal.run'
```

## 5. Verify health and ask a question

```bash
curl "$SHARDSERVE_URL/health" \
  -H "Authorization: Bearer $SHARDSERVE_API_TOKEN"

python3 -m shardserve ask \
  --url "$SHARDSERVE_URL" \
  --repository shardserve \
  --question 'Why does the scheduler reserve KV capacity before admission?' \
  --debug
```

The answer response contains source IDs and URLs, the concrete Elastic index names, retrieval and prompt hashes, token usage, TP degree, TTFT, and the pinned model revision. `--json` prints the complete response.

Also verify rejection before sharing the URL:

```bash
curl -i "$SHARDSERVE_URL/health"
```

The response must be `401 Unauthorized`.

## 6. Verify the existing Databricks plane

The current CLI login is enough for the demonstration workspace:

```bash
databricks auth profiles
databricks jobs get 655087961621636 -p shardserve -o json >/dev/null
```

Run the existing prepare → Modal → finalize workflow when publishing durable GPU evidence. Retrieval evaluation tables and automatic retrieval-pack commits are a later milestone; the first deploy uses Databricks’ already verified input, result, Volume, and MLflow path.

## 7. Before sharing the demo

- Run `python3 -m unittest discover -s tests`.
- Confirm the source tree is clean and the deployed commit matches the indexed revision.
- Keep request limits at eight live sequences and a 4,096-token context.
- Share the application token only with intended reviewers; rotate it after the demo.
- Do not promise prefix-cache performance yet. The current `/answer` response reports `prefix_cache_enabled: false` and `reused_prefix_tokens: 0` truthfully.
- Watch Modal GPU usage. Two L4 GPUs are allocated while the container is warm.
