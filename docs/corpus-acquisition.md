# Reviewed corpus acquisition

## Goal

Produce the first non-toy Nexus training corpus as a small, reproducible research artifact. This is a data-provenance gate, not a claim that a small corpus will create a capable assistant.

## Candidate source and scope

Start by evaluating a **small, metadata-filtered slice** of Common Pile v0.1 rather than downloading the full release. Common Pile documents 30 source collections and was assembled around public-domain and openly licensed text. Its overview and source descriptions are at:

- Project overview: https://huggingface.co/blog/stellaathena/common-pile
- Paper and source descriptions: https://arxiv.org/abs/2506.05209
- Dataset collection: https://huggingface.co/collections/common-pile/common-pile-v01-68307d37df48e36f02717f21

These references establish a candidate to review; they do **not** automatically establish that every record in every source is suitable for redistribution or training. Do not treat the aggregate dataset name as a per-record license.

For the first run, cap the experiment at a few megabytes of normalized text and a manageable number of documents. Select a narrow, documented source with record-level provenance and an explicit license field. Prefer permissive licenses or verified public-domain records; defer share-alike, non-commercial, no-derivatives, unclear, or missing-license records until their implications have been reviewed for this project's intended use and redistribution.

## Required source review

Before acquiring or committing any text, record:

- Source collection name, exact dataset/configuration and immutable revision or snapshot.
- Official source URL, retrieval date, upstream documentation and applicable license(s).
- License evidence at the record level where available; attribution and notice requirements.
- Original document IDs/URLs and content hashes.
- Inclusion/exclusion rules, maximum size, text extraction/normalization steps, and any filtering.
- A clear statement on whether the corpus itself may be redistributed, or only its manifest and acquisition script may be committed.

If these cannot be established, reject the candidate rather than guessing.

## Split and leakage policy

1. Split at the **document/source-record level before text is broken into training examples**. All chunks or lines derived from one document must remain in the same split.
2. Preserve official upstream train/validation/test assignments when the selected dataset provides them. Never recombine official splits and re-split them.
3. If no official split exists, assign whole documents deterministically using stable document IDs and a recorded seed, then write three immutable input files. Do not split individual lines randomly.
4. Run exact normalized-text duplicate checks across all splits. Also inspect document-ID overlap and near-duplicate risk where practical.
5. Save the manifest, upstream revision, inclusion list, source and prepared hashes, split counts, and preprocessing version alongside every training run.

The existing `prepare:dataset:source-splits` command preserves supplied split assignments and rejects exact normalized examples crossing split boundaries. It is appropriate only after the document-level split has been established; line-level duplicate checks alone cannot detect every document-level leak.

## First experiment

Once the candidate passes review:

1. Acquire only the bounded subset and keep the raw source outside Git unless redistribution rights are clear.
2. Create immutable train/validation/test text files and a source-split manifest.
3. Run the repository's manifest evaluation on the same manifest and seed for the bigram, context MLP, and Transformer baselines.
4. Run a deliberately small one-epoch Transformer training trial first. Record wall-clock time, peak memory if available, training loss, validation/test negative log-likelihood and perplexity, token/example counts, configuration, seed, and checkpoint SHA-256.
5. Compare models only on the same fixed manifest. Use validation for model selection; inspect the test set once for final reporting and do not tune against it.
6. Keep the checkpoint as a research artifact. Do not provision it to production unless data review, reproducibility, quality, serving limits, and release review all pass.

## Explicit non-goals

- No 8 TB download, unbounded crawl, or silent dataset expansion.
- No private conversations, personal data, or user content.
- No third-party inference API.
- No claim of language ability from successful CI, smoke tests, training loss, or one tiny corpus.
- No live database migration or production deployment as part of corpus selection.

## Decision record

Status: **pinned candidate implemented; acquisition and local record audit remain to be run**.

The initial candidate is `common-pile/arxiv_abstracts`, configuration `default`, upstream split `train`, pinned to dataset revision `828e35d1000f94579da8850f5f640c138279bdb5`. The dataset viewer exposes record-level `metadata.license`, authors, and arXiv abstract URL; the acquisition code accepts only rows whose license string exactly matches the CC0 1.0 public-domain dedication and whose source/URL metadata matches the expected arXiv abstract collection. It does not ingest full paper text. The license declaration is evidence from the dataset metadata, not an independent legal opinion.

Run `npm run acquire:common-pile -- --documents 200 --max-rows 5000` to create a bounded, local-only candidate corpus under `artifacts/common-pile-arxiv-abstracts/`. It records selected source IDs, URLs, authors, per-record license strings, text hashes, split assignments, and hashes in a manifest/JSONL audit file. The command fails closed if it cannot retrieve the requested number of eligible unique records or if all three splits are not represented. Do not commit the acquired text by default.

Then create the trainer's source-split manifest using the generated files and explicit attribution:

```sh
npm run prepare:dataset:source-splits -- \
  --train artifacts/common-pile-arxiv-abstracts/train.txt \
  --validation artifacts/common-pile-arxiv-abstracts/validation.txt \
  --test artifacts/common-pile-arxiv-abstracts/test.txt \
  --source-name common-pile/arxiv_abstracts \
  --source-url https://huggingface.co/datasets/common-pile/arxiv_abstracts \
  --license CC0-1.0-record-level-metadata-filtered \
  --attribution "Common Pile project and authors credited per record" \
  --output artifacts/common-pile-arxiv-abstracts/dataset-manifest.json
```

Review the resulting `records.jsonl` and `manifest.json` before training. Confirm source IDs are unique, the source revision remains available, split counts are sensible, and the source/processed hashes are stable. The first run should stay bounded; the selected abstracts are a provenance-first experiment corpus, not a claim of broad language coverage.
