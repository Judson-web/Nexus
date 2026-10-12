import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { evaluateTransformerHeldOutFile } from "./transformer-held-out.js";

test("Transformer held-out evaluation trains only on the prefix and scores both suffixes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-transformer-eval-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, ("Nexus learns from examples and predicts the next byte. ").repeat(8), "utf8");
    const report = await evaluateTransformerHeldOutFile(path, {
      epochs: 1,
      config: { contextSize: 3, embeddingSize: 4, headCount: 2, feedForwardSize: 8, seed: 23 },
    });
    assert.equal(report.split.method, "ordered 80/10/10 byte split");
    assert.equal(report.model.family, "nexus-transformer");
    assert.equal(report.evaluation.trainTokens, report.split.trainTokens);
    assert.equal(report.evaluation.validationTokens, report.split.validationTokens);
    assert.equal(report.evaluation.testTokens, report.split.testTokens);
    assert.match(report.dataset.sourceSha256, /^[a-f0-9]{64}$/);
    assert.ok(Number.isFinite(report.evaluation.validationNegativeLogLikelihood));
    assert.ok(Number.isFinite(report.evaluation.validationPerplexity));
    assert.ok(Number.isFinite(report.evaluation.testNegativeLogLikelihood));
    assert.ok(Number.isFinite(report.evaluation.testPerplexity));
    assert.ok(report.evaluation.validationPerplexity >= 1);
    assert.ok(report.evaluation.testPerplexity >= 1);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Transformer held-out evaluation rejects invalid options and short corpora", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-transformer-eval-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, "too short", "utf8");
    await assert.rejects(evaluateTransformerHeldOutFile(path), /at least 40 characters/i);
    await writeFile(path, "A sufficiently long corpus for split validation. ".repeat(3), "utf8");
    await assert.rejects(evaluateTransformerHeldOutFile(path, { epochs: 0 }), /epochs must be an integer/i);
    await assert.rejects(evaluateTransformerHeldOutFile(path, { learningRate: 1 }), /learningRate must be/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
