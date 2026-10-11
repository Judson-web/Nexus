# Nexus

Nexus is a TypeScript-first research project for building and progressively training an original language model. The model runtime does not call third-party AI inference APIs.

## Current milestone

The repository includes two in-repository CPU language-model baselines: a byte-level bigram model and a context-window neural MLP with learned byte embeddings, a tanh hidden layer, and a softmax next-byte head. The current web endpoint still uses the tiny bigram baseline; the context MLP is a research/training milestone and is not yet promoted to serving.

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
npm run evaluate -- --input data/corpus.txt --epochs 20
```

The web runtime's seed corpus is deliberately tiny and trained on cold start per server instance. Outputs are expected to be incoherent. It is only there to prove the application-to-model inference path without an external AI provider. Meaningful training requires a much larger licensed, curated dataset, evaluation splits, and measurable benchmarks. Never ingest private conversations or personal data without an explicit, appropriate data policy and consent.

## Design limits

The bigram baseline predicts one byte from the previous byte. The context MLP conditions on a fixed window of preceding bytes through learned embeddings and a hidden layer, and is trained with next-token cross-entropy. It is still a small experimental model—not a transformer, not a general-purpose assistant, and not evidence of semantic understanding. The seed corpus is far too small to produce useful language. The evaluation CLI reports held-out negative log-likelihood and perplexity for both model families using a deterministic ordered 80/10/10 byte split. The dataset preparation CLI normalizes line-oriented text, removes exact normalized duplicates, hashes source and prepared content, and assigns stable per-example train/validation/test splits using a recorded seed. The generated manifest is reproducible and records split counts; use it to inspect corpus quality before training. The current evaluator remains a small-corpus diagnostic, not a reliable estimate of generalization.

## Database

Apply SQL files in `database/migrations` to the Neon PostgreSQL project in order. Migration `0002_experiment_metrics.sql` adds dataset split manifests and initial/validation/test metrics to experiment metadata. These migrations are versioned but have not been applied to the live database. Set `DATABASE_URL` to enable optional training-run metadata persistence; without it, training runs locally without database access. Do not apply migrations to the live database until the schema change is explicitly approved; test migrations against a disposable Neon branch first. The current trainer records running/succeeded status and final loss using the existing `training_runs` table. Checkpoints and large training artifacts should live in versioned file/object storage; PostgreSQL stores metadata and checksums rather than multi-megabyte weight arrays as ordinary rows.

## First production release

See [`docs/production-readiness.md`](docs/production-readiness.md) for the release gates and current blockers. The checked-in seed corpus is a smoke-test fixture, not a production training corpus; production promotion requires a fixed held-out benchmark and a pinned, checksum-verified model artifact. The live endpoint remains unchanged until those gates pass.

## Workflow

Develop on feature branches. TypeScript checks, tests, and production builds run in GitHub Actions. Vercel's Git build-ignore setting prevents automatic deployments from routine commits; production deployment is a separate, explicit action after verification.
