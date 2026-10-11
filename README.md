# Nexus

Nexus is a TypeScript-first research project for building and progressively training an original language model. The model runtime does not call third-party AI inference APIs.

## Current milestone

The repository includes a small byte-level bigram neural language model, a responsive Next.js playground, and a Node.js inference endpoint that trains a tiny seed model in memory. This is an end-to-end runtime smoke test, not a capable conversational assistant.

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
```

The web runtime's seed corpus is deliberately tiny and trained on cold start per server instance. Outputs are expected to be incoherent. It is only there to prove the application-to-model inference path without an external AI provider. Meaningful training requires a much larger licensed, curated dataset, evaluation splits, and measurable benchmarks. Never ingest private conversations or personal data without an explicit, appropriate data policy and consent.

## Design limits

The current model predicts local byte-to-byte transitions. It has no long context, semantic understanding, transformer blocks, or useful general-purpose chat capability. Do not interpret the playground as a production AI assistant. The next engineering milestone is a more expressive model architecture with held-out evaluation and persistent, versioned checkpoints.

## Database

Apply SQL files in `database/migrations` to the Neon PostgreSQL project. Checkpoints and large training artifacts should live in versioned file/object storage; PostgreSQL stores metadata and checksums rather than multi-megabyte weight arrays as ordinary rows.

## Workflow

Develop on feature branches. TypeScript checks, tests, and production builds run in GitHub Actions. Vercel's Git build-ignore setting prevents automatic deployments from routine commits; production deployment is a separate, explicit action after verification.
