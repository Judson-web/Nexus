# Dataset expansion plan

## Current accepted source

- **Common Pile arXiv abstracts** — the existing bounded acquisition path accepts only rows with the exact CC0 1.0 dedication and expected arXiv metadata, pinned to a reviewed dataset revision. This is a useful seed corpus, not enough data to produce a capable general-purpose model.

## Candidate register

| Candidate | Domain / language | Initial disposition | Gate before ingestion |
|---|---|---|---|
| [Common Pile arXiv abstracts](https://huggingface.co/datasets/common-pile/arxiv_abstracts) | Scientific abstracts, mostly English | **Accepted, bounded only** | Preserve pinned revision, record-level CC0 evidence, authors/URLs, content hashes, and split assignments. |
| [dataseek/ptbr-books-publicos](https://huggingface.co/datasets/dataseek/ptbr-books-publicos) | Portuguese public-domain books | **Review required** | Verify book-level public-domain status and original source/edition URLs; the dataset card alone does not establish every book's provenance. Review content safety/PII and language balance. Do not ingest yet. |
| [fka/prompts.chat](https://huggingface.co/datasets/fka/prompts.chat) | Prompt templates, multilingual/mixed | **Later-stage candidate** | Review current revision, row license, attribution, safety, duplicates, and whether each row contains useful target answers. Prompts alone are not a supervised instruction dataset. Keep separate from next-token pretraining. |
| [Wikimedia / Wikipedia text](https://www.wikimedia.org/) | General reference knowledge | **Deferred** | Text is commonly available under attribution/share-alike terms. Decide how attribution and share-alike obligations will be met for datasets, derivatives, and releases before acquiring any text. |
| Project Gutenberg works | Books, language varies | **Per-work review required** | Check each work's status in the relevant jurisdiction and any additional terms. Do not infer a global public-domain guarantee from the collection brand. |
| Official technical documentation | Programming and computing | **Source-by-source review required** | Record exact repository/page revision and license. Publicly accessible documentation is not automatically licensed for model training or redistribution. |

## Reviewed instruction-tuning source: Smol-SmolTalk

A bounded adapter is available for `HuggingFaceTB/smol-smoltalk`:

```sh
npm run acquire:smol-smoltalk -- --max-train-rows 1500 --max-test-rows 500 --output artifacts/smol-smoltalk
npm run prepare:dataset:source-splits -- \
  --train artifacts/smol-smoltalk/train.txt \
  --validation artifacts/smol-smoltalk/validation.txt \
  --test artifacts/smol-smoltalk/test.txt \
  --source-name "HuggingFaceTB/smol-smoltalk:smol-magpie-ultra-short" \
  --source-url "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk" \
  --license "Apache-2.0" \
  --attribution "Hugging Face Smol Models Research, Smol-SmolTalk / Smol-Magpie-Ultra; arXiv:2502.02737" \
  --output artifacts/smol-smoltalk-manifest.json
```

The acquisition script pins revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` and verifies the Hub revision and Apache-2.0 card license before and after retrieval. It reads bounded pages from the Dataset Viewer `/rows` endpoint and filters for the exact `smol-magpie-ultra-short` source label locally, because `/filter` currently returns HTTP 500 for this dataset. It preserves the upstream test split and derives a deterministic validation partition from selected training rows. The pinned Hub card license and revision are verified; the artifact includes a provenance notice and canonical Apache-2.0 license URL because this revision has no standalone `LICENSE` file.

Do not remove the source filter or point this adapter at the `all` SmolTalk mix. Other source labels include datasets with separate license terms. The adapter is intentionally conservative: only a single-turn user/assistant pair with an optional system instruction is accepted, obvious PII-like values are dropped, mixed-role conversations are excluded, and cross-split exact duplicates cause a hard failure.

The acquisition workflow now also runs a **separate, capped from-scratch SFT smoke experiment** on 500 source-approved train examples, 20 validation examples and 32 test examples. It records response-only held-out loss in checkpoint provenance and uploads two assistant-format continuation samples. These small caps keep the reference scalar-autodiff trainer within a reasonable CPU budget; they are a reproducibility/diagnostic run, not a claim of general capability. The full bounded acquisition artifact (598 train / 28 validation / 218 test in the verified run) stays separate from this smaller experiment split, and no checkpoint is promoted to production.

Use the resulting manifest for an independent experiment first. The TypeScript scalar-autodiff backend is slow; a 1,200-row data cap is a small reproducible comparison set, not a claim that this is the full dataset or enough data for a capable general-purpose model. Measure response-only validation loss and inspect generated conversation samples before considering any mixed-corpus run.

## Ingestion gates (must fail closed)

Every newly approved source must have a source card documenting:

1. Exact dataset, configuration, immutable revision/snapshot, retrieval date, and official upstream documentation.
2. License evidence and attribution obligations at the record/work level where possible; uncertain records are excluded.
3. Stable source IDs, original URLs, authors/creators where provided, normalized content hashes, and a documented transformation history.
4. Explicit inclusion/exclusion rules, size caps, language/domain labels, and content-quality/safety filtering.
5. Document-level train/validation/test assignment **before chunking**, preserved upstream splits where supplied, and exact normalized deduplication across all splits.
6. Redistribution decision: raw text may stay outside Git unless redistribution is clearly permitted. Store manifests and scripts separately from raw text.
7. A source-specific audit artifact and tests proving missing provenance/license fields are rejected.

Do not concatenate corpora blindly. Track examples/tokens by source, language, and domain; use a deliberate sampling budget so a large source cannot silently drown out the existing scientific English corpus. Re-run held-out evaluation on the same fixed manifest when comparing models, use validation for selection, and keep test data out of tuning.

## Recommended sequence

1. Finish the current bounded arXiv training run and inspect generated continuations.
2. Implement a reusable source-review/manifest interface before adding a second raw corpus.
3. Investigate a larger English corpus with strong work-level provenance and compatible licensing; approve only after the source card is complete.
4. Add one source at a time, audit duplicates and split leakage, then compare source-only and mixed-corpus baselines.
5. Treat instruction tuning as a separate future stage with reviewed prompt/answer pairs. Do not mix prompt-only data into pretraining as if it contained answers.

## Tokenization and context-window upgrade

The manifest trainer now learns a deterministic byte-pair vocabulary from the **training split only**, persists its merge table inside the checkpoint, and uses that same tokenizer for generation. Vocabulary size is bounded at 512 IDs, including the 257 base byte/EOS IDs. The corpus-training workflow uses a 32-token context rather than 8 and audits token compression and sequence-length percentiles. This is a research baseline, not a production-grade tokenizer: compare its compression and generated text against the byte-level baseline, and keep the held-out splits out of tokenizer training.

## Success criteria

A successful CI run proves only that acquisition, training, checkpoint integrity, and basic generation checks ran. It does **not** prove coherent language ability. Inspect the sample artifact for coherence, repetition, prompt copying, and malformed text; report held-out loss/perplexity separately from human-readable generation quality. No production deployment or live database change is part of this work.
