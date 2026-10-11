# Nexus

Nexus is a TypeScript-first research project for building and progressively training an original language model. The model runtime does not call third-party AI inference APIs.

## Current milestone

This repository starts with a deliberately small, byte-level bigram neural language model. It is a real trainable baseline and engineering testbed, not a claim of modern conversational capability. The next stages can replace or extend the model behind the same training/evaluation boundaries.

- Deterministic initialization and reproducible training.
- UTF-8 byte tokenizer with a dedicated end-of-sequence token.
- CPU-only next-token training and generation.
- Versioned, validated JSON checkpoints with checksums.
- Tests for tokenizer, learning behavior, and checkpoint integrity.
- PostgreSQL migrations for datasets, runs, model versions, and evaluation metadata.
- No third-party inference APIs; no production deployment in routine development.

## Requirements

Node.js 22 or newer.

```sh
npm install
npm run check
npm test
npm run train -- --input data/corpus.txt --epochs 30 --learning-rate 0.08 --output artifacts/bigram-v1.json
npm run generate -- --checkpoint artifacts/bigram-v1.json --prompt "Nexus " --tokens 160
```

The included corpus is only a smoke-test seed. For meaningful training, provide a licensed, curated corpus with enough diverse examples. Never ingest private conversations or personal data without an explicit, appropriate data policy and consent.

## Design limits

The first model learns local byte-to-byte transitions only. It has no long context, semantic understanding, transformer blocks, or useful general-purpose chat ability. The point of this milestone is to prove a transparent end-to-end training and inference path before scaling model complexity.

## Database

Apply SQL files in `database/migrations` to the new Neon PostgreSQL project. Checkpoints and large training artifacts should live in versioned file/object storage; PostgreSQL stores metadata and checksums, not multi-megabyte weight arrays as ordinary rows.

## Workflow

Develop on feature branches. Keep `main` stable. Do not merge or deploy without explicit instruction and verification.
