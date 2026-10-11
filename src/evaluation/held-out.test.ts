import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { evaluateHeldOutFile } from "./held-out.js";

test("held-out evaluation scores every test token with preceding context", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-eval-"));
  const path = join(directory, "corpus.txt");
  try {
    const corpus = ("Nexus builds its own language model. Training needs clean data. ").repeat(20);
    await writeFile(path, corpus, "utf8");
    const report = await evaluateHeldOutFile(path, 1);
    assert.equal(report.split.method, "ordered 80/10/10 byte split");
    assert.equal(report.evaluations.length, 2);
    assert.match(report.dataset.sourceSha256, /^[a-f0-9]{64}$/);
    assert.match(report.dataset.evaluatedContentSha256, /^[a-f0-9]{64}$/);
    for (const result of report.evaluations) {
      assert.equal(result.testTokens, report.split.testTokens);
      assert.equal(result.validationTokens, report.split.validationTokens);
      assert.ok(Number.isFinite(result.validationNegativeLogLikelihood));
      assert.ok(Number.isFinite(result.validationPerplexity));
      assert.ok(Number.isFinite(result.testNegativeLogLikelihood));
      assert.ok(Number.isFinite(result.testPerplexity));
      assert.ok(result.validationPerplexity >= 1);
      assert.ok(result.testPerplexity >= 1);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("held-out evaluation rejects short corpora and invalid epoch counts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-eval-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, "too short", "utf8");
    await assert.rejects(evaluateHeldOutFile(path, 1), /at least 40 characters/i);
    await writeFile(path, "A sufficiently long corpus for this validation test. ".repeat(3), "utf8");
    await assert.rejects(evaluateHeldOutFile(path, 0), /epochs must be between/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
