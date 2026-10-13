# Nexus training architecture research notes

Updated: 2026-10-13

This document records design decisions from recent open research and repositories. Nexus remains a TypeScript-first, in-house model; external training repositories are references, not runtime or inference dependencies.

## Research reviewed

- **Microsoft Research, “Demystifying Data Organization for Enhanced LLM Training” (ACL 2026)**: recommends attention to boundary sharpening, cyclic scheduling, curriculum continuity, and local diversity. Repository: https://github.com/microsoft/data-efficacy/ and paper overview: https://www.microsoft.com/en-us/research/publication/demystifying-data-organization-for-enhanced-llm-training/
- **“Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining” (ICLR 2026)**: formalizes data quality as a scaling factor. Practical implication: don't equate raw corpus size with useful training signal; curate and deduplicate.
- **“A Hitchhiker’s Guide to Scaling Law Estimation” (2025)**: recommends using intermediate checkpoints and multiple smaller runs to estimate scaling behavior instead of drawing conclusions from a single final checkpoint. https://proceedings.mlr.press/v267/choshen25a.html
- **PyTorch TorchTitan**: reference for composable parallelism, reproducible configurations, checkpointing, and observability: https://github.com/pytorch/torchtitan
- **Lightning LitGPT**: reference for readable from-scratch model implementations and explicit pretraining recipes: https://github.com/Lightning-AI/litgpt

## Decisions integrated into Nexus

1. **Parallel, controlled experiments:** one shared, source-preserving corpus artifact feeds independent runs. The current matrix varies context length, learning rate, and embedding/feed-forward capacity. Each candidate writes its own checkpoint, provenance, held-out metrics, and generated samples.
2. **Fail independently:** matrix runs use `fail-fast: false`; one bad configuration must not cancel useful candidates.
3. **Data preparation is shared:** acquire and split sources once per workflow, rather than allowing each experiment to download slightly different corpora.
4. **Promotion is separate from training:** all runs remain candidates. No automatic winner selection or production promotion based on training loss alone.
5. **Quality-aware data work:** source-level provenance and overlapping document windows are implemented. A deterministic `length-cyclic` ordering heuristic now interleaves four token-length bands and is compared against manifest order in the experiment matrix. It is inspired by curriculum-continuity/local-diversity guidance but is not a reproduction of Microsoft's STR/SAW methods. Near-duplicate filtering beyond exact duplicates remains future work.
6. **Intermediate checkpoints and resume are required next:** persist model weights, tokenizer, optimizer moments/step, RNG state, data position, configuration, and corpus fingerprint atomically. A resumed run must be equivalent to uninterrupted training within numerical tolerances.
7. **Optimize the implementation before extreme context expansion:** the current scalar-autodiff TypeScript reference path is a correctness-oriented CPU trainer, not a scalable distributed engine. Measure end-to-end tokens/second and gradient parity before replacing it with a vectorized path. Keep the scalar path as a test oracle.

## Limits and cautions

TorchTitan and LitGPT are Python/PyTorch stacks built for GPU-scale work. Porting them wholesale would conflict with Nexus's TypeScript-first architecture and would not make the current scalar autodiff path faster. Their ideas should be reimplemented selectively and tested. Similarly, curriculum/data-ordering gains reported on larger models are hypotheses to validate on Nexus, not guaranteed improvements at this scale.

## Required run report

Every experiment should report the exact config and seed, corpus fingerprint and split counts, train/validation/test tokens, elapsed time, tokens/second, loss and perplexity by split, truncation/drop counts, checkpoint checksum, and fixed-prompt generation samples. Compare candidates on identical held-out sources and prompts. A candidate is eligible for promotion only after reproducible metrics and a blind human review show a real readability gain.
