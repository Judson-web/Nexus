# First production release gate

Nexus is built around an original TypeScript model runtime. Production readiness is an evidence gate, not a successful build alone.

## Current status

- [x] TypeScript checks, unit tests, and build run in CI.
- [x] Reproducible byte-level bigram and context-window MLP baselines.
- [x] Checkpoint checksum verification and deterministic dataset manifest generation.
- [x] Validation/test metrics are reported separately by the held-out evaluator.
- [ ] Connect prepared manifest splits to training and evaluation end-to-end.
- [ ] Establish a licensed, documented, deduplicated training corpus with enough data to measure generalization.
- [ ] Implement and benchmark a causal transformer against the current MLP and bigram baselines.
- [ ] Persist evaluation metrics and dataset/checkpoint provenance end-to-end.
- [x] Load an explicitly configured Transformer checkpoint only after SHA-256 and byte-count verification; cache the verified model per Node.js process.
- [ ] Provision a reviewed checkpoint artifact into the runtime and demonstrate improved held-out metrics before enabling public inference.
- [ ] Add request limits, timeout/memory protections, and production observability.
- [ ] Pass an explicit release review before changing the production endpoint or deploying.

## Why we are not promoting the current model yet

The checked-in seed corpus is only about 510 bytes. That is useful for smoke tests, but not a meaningful language-model training set. The current context MLP and single-block Transformer are research baselines, not conversational assistants. The API serves only an explicitly configured Transformer checkpoint after checksum verification. If no checkpoint is configured, it returns `503 MODEL_NOT_CONFIGURED`; it does not silently serve the bigram baseline. Its evaluation currently uses an ordered 80/10/10 byte split, while the dataset manifest uses stable content-hash splits over normalized lines; those are not interchangeable and must be unified before reporting release metrics.

## Release policy

- Work on a feature branch and require green checks.
- Keep the existing production endpoint unchanged until a candidate has measured improvements on a fixed held-out test set.
- Do not call third-party AI inference APIs.
- Do not apply Neon migrations to the live database without explicit approval.
- Do not claim conversational competence based on a smoke test or training loss alone.
