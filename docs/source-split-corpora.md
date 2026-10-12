# Preparing corpora with source-provided splits

Use this command when the corpus already publishes separate training, validation, and test files:

```sh
npm run prepare:dataset:source-splits -- \
  --train data/source/train.txt \
  --validation data/source/valid.txt \
  --test data/source/test.txt \
  --source-name "Dataset name and version" \
  --source-url "https://example.org/dataset" \
  --license "Exact license identifier and version" \
  --attribution "Required attribution text" \
  --output artifacts/dataset-manifest.json
```

The example URL and license above are placeholders; replace them with verified source information. Do not run training until the source's actual license, attribution requirements, redistribution terms, and suitability for your intended use have been reviewed and recorded.

The command preserves the published split assignment rather than re-shuffling combined lines. It normalizes whitespace and Unicode NFC, removes exact duplicates within each split, and **fails closed** if the same normalized line occurs across different splits. That conservative rule avoids silently allowing duplicate examples to leak from train into validation/test. If a source has intentional cross-split duplication, investigate and document it before preparing a manifest rather than bypassing this check.

The manifest records SHA-256 digests of all three raw source files as a labeled aggregate, prepared examples and split assignments, plus source name, URL, license, attribution, and input paths. Keep the original files and their acquisition notes alongside your experiment records; hashes establish file identity, not legal permission or source authenticity.

Then train only on the manifest's train split and evaluate the exact checkpoint on validation and test:

```sh
npm run train:transformer:manifest -- \
  --manifest artifacts/dataset-manifest.json \
  --output artifacts/transformer-manifest-v1.json \
  --epochs 1
```

The current Transformer is a small CPU research implementation. Benchmark it on a small sample first; a valid manifest and successful training run do not imply useful language quality. Do not tune hyperparameters against test results, and do not provision a checkpoint for public inference until quality and runtime have been independently reviewed.
