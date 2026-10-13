# Nexus

Nexus is a TypeScript-first research project for building and progressively training an original language model. The model runtime does not call third-party AI inference APIs.

## Current milestone

The repository includes two in-repository CPU language-model baselines: a byte-level bigram model and a context-window neural MLP with learned byte embeddings, a tanh hidden layer, and a softmax next-byte head. The web endpoint serves only a configured, checksum-verified Transformer checkpoint. If no checkpoint is configured, inference returns an explicit `503 MODEL_NOT_CONFIGURED` response rather than generating misleading output from the tiny bigram baseline.

- Deterministic model initialization, UTF-8 byte tokenizer, and CPU training.
- Versioned JSON checkpoints with checksum verification.
- Responsive model playground at `/` and input-validated `POST /api/generate`.
- No third-party AI inference APIs or provider keys.
- PostgreSQL schema for dataset metadata, training runs, model versions, and evaluation metadata.
- Git-based Vercel builds are intentionally ignored; deployments must be explicitly initiated after verification.

## Requirements and commands

Node.js 22 or newer.

```sh
npm install
npm run check
npm test
npm run build
npm run dev
npm run train -- --input data/corpus.txt --epochs 30 --learning-rate 0.08 --output artifacts/bigram-v1.json
npm run generate -- --checkpoint artifacts/bigram-v1.json --prompt "Nexus " --tokens 160
npm run train:context -- --input data/corpus.txt --epochs 30 --learning-rate 0.03 --output artifacts/context-mlp-v1.json
npm run generate:context -- --checkpoint artifacts/context-mlp-v1.json --prompt "Nexus " --tokens 160
npm run prepare:dataset -- --input data/corpus.txt --output artifacts/dataset-manifest.json --seed 17
npm run evaluate:manifest -- --manifest artifacts/dataset-manifest.json --epochs 10 --seed 17
npm run train:transformer:manifest -- --manifest artifacts/dataset-manifest.json --epochs 1 --output artifacts/transformer-manifest-v1.json\nnpm run train:sweep -- --manifest artifacts/dataset-manifest.json --output-dir artifacts/sweep --workers 2 --epochs 1
npm run evaluate -- --input data/corpus.txt --epochs 20
```

Manifest-driven Transformer training uses only examples marked `train`. It writes a normal Transformer checkpoint plus a `.sha256` checksum file and `.provenance.json` sidecar recording the exact manifest hash, source/prepared hashes, split counts, training-example IDs, model configuration, seed, learning rate, epochs, and training loss. The same in-memory checkpoint is scored on validation and test splits before serialization; negative log-likelihood, perplexity, token counts, and example counts are recorded in `heldOutEvaluation`. Validation/test examples are never used for gradient updates, and test metrics are not used for model selection. The training command is CPU-expensive; start with one epoch and a small model configuration. Example: `npm run train:transformer:manifest -- --manifest artifacts/dataset-manifest.json --epochs 1 --context-size 3 --embedding-size 4 --heads 2 --feed-forward-size 8 --learning-rate 0.0001 --output artifacts/transformer-v1.json`. The checkpoint remains compatible with `npm run generate:transformer`.

To serve a trained manifest checkpoint, set `NEXUS_TRANSFORMER_CHECKPOINT` to the absolute path of the checkpoint file and place the trainer-generated `.sha256` sidecar beside it. The Node.js API verifies the checkpoint byte count and SHA-256 before loading it, caches the verified model for the process lifetime, and fails the request rather than silently falling back if a configured artifact is invalid. Example: train with `npm run train:transformer:manifest -- --manifest artifacts/dataset-manifest.json --output artifacts/transformer.json`, then set `NEXUS_TRANSFORMER_CHECKPOINT=/absolute/path/to/transformer.json`. The artifact must be provisioned into the runtime separately; it is not committed to Git. This wiring does not imply that the model is capable: quality still depends on a sufficiently large licensed corpus and held-out evaluation. Never ingest private conversations or personal data without an explicit, appropriate data policy and consent.

## Design limits

The bigram baseline predicts one byte from the previous byte. The context MLP conditions on a fixed window of preceding bytes through learned embeddings and a hidden layer, and is trained with next-token cross-entropy. It is still a small experimental model—not a transformer, not a general-purpose assistant, and not evidence of semantic understanding. The seed corpus is far too small to produce useful language. The legacy evaluation CLI reports held-out negative log-likelihood and perplexity using an ordered 80/10/10 byte split. The manifest evaluation CLI instead trains both baselines only on train examples from the prepared manifest and reports validation/test metrics on their separate line-level splits. Each line is treated as an independent sequence. The dataset preparation CLI normalizes line-oriented text, removes exact normalized duplicates, hashes source and prepared content, and assigns stable per-example splits using a recorded seed. These evaluation protocols are distinct; compare model runs only when they use the same manifest, seed, and training settings. Both remain diagnostic on small corpora, not reliable estimates of generalization.

## Decoder architecture

The current experimental model uses a configurable stack of independently parameterized causal decoder blocks (1–8 layers, default 2), with multi-head causal self-attention, residual connections, RMSNorm, and SwiGLU feed-forward networks. Every position is updated in each block, so deeper layers can build representations over earlier contextualized states. The reverse-mode autodiff path traverses the same stack and is covered by finite-difference gradient tests. Older checkpoints without a `layerCount` field load as one-layer models; newly saved checkpoints record the configured depth.

Manifest-driven BPE training now learns up to 767 merges (a maximum vocabulary of 1,024 IDs including the byte/EOS base), and the model supports context windows up to 256 tokens. The sweep includes longer-context and different-depth candidates, while the default corpus run remains bounded to protect CPU time. The TypeScript scalar autodiff implementation remains CPU-oriented and is intended for small research runs. Increasing depth improves architectural capacity but also increases training cost substantially. Start with the small concurrent sweep, inspect generation samples and validation loss, and scale only when measured runtime and memory permit. This is not a claim that the model already matches the capabilities of a large pretrained language model.

## Resumable training and optimizer controls

Manifest-driven training now supports AdamW (decoupled weight decay), Adam, or SGD, with cosine or constant learning-rate schedules. AdamW is the default for new manifest runs; the default weight decay is 0.01. The training checkpoint preserves model weights plus Adam/AdamW first and second moments and the optimizer step, while a `.resume.json` sidecar binds completed training-example IDs to the exact manifest hash and training configuration. Checkpoints are written atomically after each completed training example, so an interrupted run can resume from the most recent completed example without reusing validation/test data.

Example:

```sh
npm run train:transformer:manifest -- --manifest artifacts/dataset-manifest.json --output artifacts/transformer.json --epochs 1
# Resume using the exact same manifest and training settings:
npm run train:transformer:manifest -- --manifest artifacts/dataset-manifest.json --resume-from artifacts/transformer.json --output artifacts/transformer.json --epochs 1
```

The resume path intentionally rejects mismatched manifest hashes, model settings, optimizer, schedule, learning rate, or sequence settings. Checkpoint format v2 stores optimizer state while continuing to load legacy v1 checkpoints. The current model is still a single-block CPU reference Transformer; these training improvements improve experiment reliability, not model scale or guaranteed language quality.

## Concurrent training sweeps

`npm run train:sweep -- --manifest artifacts/dataset-manifest.json --output-dir artifacts/sweep --workers 2 --epochs 1` launches four isolated candidate runs with a bounded concurrency pool. Each candidate gets its own checkpoint, checksum, and provenance sidecar. The sweep writes `sweep-summary.json` and ranks successful runs using validation negative log-likelihood only; test metrics are never used for ranking, and no candidate is automatically promoted to the serving endpoint. `--workers` accepts 1–8, `--epochs` accepts 1–1000, and `--seed` controls deterministic candidate seeds. The default two-worker setting enables parallel experiments without immediately saturating every CPU core. This is experiment orchestration, not a claim that the underlying single-block reference Transformer is production-scale; larger models remain constrained by the current CPU-oriented scalar backpropagation implementation and the quality/size of the licensed corpus.

## Database

Apply SQL files in `database/migrations` to the Neon PostgreSQL project in order. Migration `0002_experiment_metrics.sql` adds dataset split manifests and initial/validation/test metrics to experiment metadata. These migrations are versioned but have not been applied to the live database. Set `DATABASE_URL` to enable optional training-run metadata persistence; without it, training runs locally without database access. Do not apply migrations to the live database until the schema change is explicitly approved; test migrations against a disposable Neon branch first. The current trainer records running/succeeded status and final loss using the existing `training_runs` table. Checkpoints and large training artifacts should live in versioned file/object storage; PostgreSQL stores metadata and checksums rather than multi-megabyte weight arrays as ordinary rows.

## First production release

See [`docs/production-readiness.md`](docs/production-readiness.md) for the release gates and current blockers. The checked-in seed corpus is a smoke-test fixture, not a production training corpus; production promotion requires a fixed held-out benchmark and a pinned, checksum-verified model artifact. The live endpoint remains unchanged until those gates pass.

## Workflow

Develop on feature branches. TypeScript checks, tests, and production builds run in GitHub Actions. Vercel's Git build-ignore setting prevents automatic deployments from routine commits; production deployment is a separate, explicit action after verification.


## First-response corpus

The bounded first-response training workflow uses the pinned [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b), released under Apache-2.0. The acquisition script downloads the pinned `ready.messages.jsonl.gz` archive in one request (avoiding dataset-viewer rate limits), verifies the upstream revision and declared license before and after retrieval, and records compressed/uncompressed SHA-256 hashes, filtering decisions, split hashes, source attribution, and provenance in an artifact manifest. The source license is included as `OASST1-LICENSE`; downloaded conversations are not committed to this repository.

The workflow keeps English user/assistant parent-child pairs, excludes deleted and synthetic messages, negative review results, obvious contact information, and high toxicity or personal-information label scores where available. It deduplicates exact normalized pairs and assigns complete conversation trees to deterministic train/validation/test splits to prevent turn-level leakage. The initial run deterministically samples 500 complete trees from the pinned archive to keep CPU training bounded. Increase the sample only after measured runtime and response quality justify it. Samples are serialized as `User: ... Assistant: ...`. The current trainer learns the whole sequence rather than masking loss to assistant-only tokens, so this is a fast first-response baseline—not a claim of polished conversational behavior. Apache-2.0 license and attribution notices must accompany redistributed copies. See the [OASST1 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1) and [OpenAssistant license guidance](https://github.com/LAION-AI/Open-Assistant/blob/main/docs/docs/faq.md).
