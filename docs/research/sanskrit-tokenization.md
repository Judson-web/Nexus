# Sanskrit tokenization research track

## Status

Research-only. This track must not change the default English corpus, training workflow, model-serving path, or production configuration. Do not merge results into the main training path until the experiment is reproducible and reviewed.

## Question

Does adding Sanskrit improve Nexus's training efficiency or English capability, rather than merely reducing token counts on a small parallel text sample?

Token count is a tokenizer property, not a measure of model quality. A lower count can reflect a tokenizer's vocabulary and training corpus; it does not establish better reasoning, generalization, or generation.

## Experimental stages

### Stage 0 — Reproduce tokenization evidence

- Use a small, documented parallel Sanskrit/English evaluation set with an explicitly verified license and source provenance.
- Do not assume that a Sanskrit text is public domain solely because it is ancient: the edition, translation, transcription, and digital dataset may have separate rights.
- Record each source URL, license, attribution requirements, source revision, retrieval date, document ID, and content SHA-256.
- Normalize Unicode consistently, preserve the original text, and report normalization policy.
- Compare the current byte-level/BPE tokenizer against a Sanskrit-aware candidate on the same passages.
- Report total tokens, tokens per character/word, unknown or fallback coverage, and corpus-level distributions. Include transliteration only as a separately labeled condition.
- Keep passage-level train/validation/test splits deterministic and deduplicated before training any tokenizer.

### Stage 1 — Controlled model experiment

Run only after Stage 0 passes the data and implementation checks.

Compare:
1. English-only baseline.
2. English plus a bounded, license-cleared Sanskrit corpus.
3. Where feasible, the same bilingual data with a tokenizer trained on the same training split.

Keep architecture, initialization seed, optimizer, training-token or compute budget, sequence length policy, and evaluation set fixed wherever possible. If token budgets and compute budgets cannot both be held constant, report both comparisons rather than presenting either as definitive.

The tokenizer must be trained on training data only. Validation/test data must not influence tokenizer merges, model updates, checkpoint selection, or hyperparameter decisions. Keep source documents grouped across splits and prevent exact or normalized duplicates crossing split boundaries.

### Metrics

Report all of the following, with the corpus sizes and confidence/variance context:

- Token efficiency on held-out Sanskrit and English passages.
- Validation negative log-likelihood and perplexity, separated by language.
- Held-out English generation quality and task behavior, using fixed prompts and a documented rubric.
- Sanskrit generation quality, reviewed for grammaticality and meaning preservation by a capable evaluator; do not infer quality from perplexity alone.
- Training time, examples/tokens processed, peak memory if available, and total compute.
- Checkpoint, tokenizer, dataset manifest, source revisions, configuration, and evaluation artifact hashes.

A bilingual improvement must not be described as an English improvement unless held-out English metrics and generation checks support that claim.

## Data and licensing gate

Do not automatically ingest a convenient Sanskrit corpus. First review the exact dataset card, source edition, license, attribution/share-alike or non-commercial restrictions, and provenance. Avoid scraping texts or translations without clear rights. Keep the research fixture small and redistributable under its verified terms; if redistribution is not allowed, keep only documented acquisition instructions and hashes.

## Acceptance criteria

- Deterministic tests cover Unicode handling, Devanagari combining marks, Sanskrit punctuation, transliteration as a separate condition, serialization/reload, and stable token IDs.
- No validation/test leakage, no unreviewed data source, and no default-pipeline changes.
- The report includes negative or inconclusive results, not only favorable token-count results.
- Main-pipeline adoption requires reproducible evidence of a useful tradeoff and a separate review.

## Immediate next steps

1. Review the active Nexus training run and retain its English baseline artifacts.
2. Select and approve a small Sanskrit/English dataset with clear licensing.
3. Add a tokenizer audit that measures the current tokenizer on the exact same passages before building a new tokenizer.
4. Add tests and a reproducible report; only then run a bounded model comparison.
