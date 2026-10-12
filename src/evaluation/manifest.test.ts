import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { prepareDataset } from "../data/dataset.js";
import { evaluateDatasetManifest } from "./manifest.js";

test("manifest evaluator trains only on train examples and reports held-out metrics", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-manifest-eval-"));
  const path = join(directory, "manifest.json");
  try {
    const corpus = Array.from({ length: 120 }, (_, i) =>
      `Unique sentence ${i} about model training and data separation.`).join("\n");
    const manifest = prepareDataset(corpus, { seed: 3 });
    assert.ok(manifest.splits.train > 0);
    assert.ok(manifest.splits.validation > 0);
    assert.ok(manifest.splits.test > 0);
    await writeFile(path, JSON.stringify(manifest), "utf8");
    const report = await evaluateDatasetManifest({ manifestPath: path, epochs: 1, seed: 9 });
    assert.equal(report.dataset.counts.train, manifest.splits.train);
    assert.equal(report.training.trainedExamples, manifest.splits.train);
    assert.match(report.dataset.manifestSha256, /^[a-f0-9]{64}$/);
    assert.equal(report.evaluations.length, 2);
    for (const model of report.evaluations) {
      assert.equal(model.validation.examples, manifest.splits.validation);
      assert.equal(model.test.examples, manifest.splits.test);
      assert.ok(model.validation.tokens > 0 && model.test.tokens > 0);
      assert.ok(Number.isFinite(model.validation.perplexity));
      assert.ok(Number.isFinite(model.test.perplexity));
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("opt-in Transformer trains only on manifest train split and reports held-out metrics", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-transformer-manifest-"));
  const path = join(directory, "manifest.json");
  try {
    const corpus = Array.from({ length: 60 }, (_, i) => `x${i}`).join("\n");
    const manifest = prepareDataset(corpus, { seed: 31 });
    await writeFile(path, JSON.stringify(manifest), "utf8");
    const report = await evaluateDatasetManifest({
      manifestPath: path,
      epochs: 1,
      seed: 9,
      includeTransformer: true,
    });
    assert.equal(report.evaluations.length, 3);
    const transformer = report.evaluations.find((item) => item.modelFamily === "nexus-transformer");
    assert.ok(transformer);
    assert.equal(transformer.validation.examples, manifest.splits.validation);
    assert.equal(transformer.test.examples, manifest.splits.test);
    assert.ok(Number.isFinite(transformer.validation.negativeLogLikelihood));
    assert.ok(Number.isFinite(transformer.test.perplexity));
    assert.equal(report.training.transformer?.trainedExamples, manifest.splits.train);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("manifest evaluator rejects split count tampering", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-manifest-eval-"));
  const path = join(directory, "manifest.json");
  try {
    const manifest = prepareDataset(Array.from({ length: 80 }, (_, i) => `Line ${i} contains unique training data.`).join("\n"));
    await writeFile(path, JSON.stringify({ ...manifest, splits: { ...manifest.splits, test: 999 } }), "utf8");
    await assert.rejects(evaluateDatasetManifest({ manifestPath: path, epochs: 1 }), /count does not match/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("manifest evaluator rejects example text and prepared hash tampering", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-manifest-eval-"));
  const path = join(directory, "manifest.json");
  try {
    const manifest = prepareDataset(Array.from({ length: 120 }, (_, i) => `Unique integrity line ${i} for dataset validation.`).join("\n"), { seed: 3 });
    const changedText = {
      ...manifest,
      examples: manifest.examples.map((example, index) => index === 0 ? { ...example, text: example.text + " altered" } : example),
    };
    await writeFile(path, JSON.stringify(changedText), "utf8");
    await assert.rejects(evaluateDatasetManifest({ manifestPath: path, epochs: 1 }), /text hash does not match/i);

    const changedPreparedHash = { ...manifest, preparedSha256: "0".repeat(64) };
    await writeFile(path, JSON.stringify(changedPreparedHash), "utf8");
    await assert.rejects(evaluateDatasetManifest({ manifestPath: path, epochs: 1 }), /prepared-content hash does not match/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
