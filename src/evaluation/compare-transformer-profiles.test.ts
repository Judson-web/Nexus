import test from "node:test";
import assert from "node:assert/strict";
import { compareTransformerProfileReports } from "./compare-transformer-profiles.js";

function report(overrides: Record<string, unknown> = {}) {
  return {
    dataset: { sourceSha256: "a".repeat(64), tokenCount: 120 },
    model: {
      family: "nexus-transformer",
      config: { contextSize: 3, embeddingSize: 4, seed: 101 },
      parameterCount: 500,
      epochs: 4,
      learningRate: 0.0001,
      repeats: 5,
    },
    profile: {
      totalDurationMs: 100,
      meanEpochDurationMs: 25,
      meanExamplesPerSecond: 40,
      medianFinalLoss: 2,
    },
    ...overrides,
  };
}

test("compares compatible reports and calculates signed changes", () => {
  const result = compareTransformerProfileReports(report(), report({
    profile: { totalDurationMs: 80, meanEpochDurationMs: 20, meanExamplesPerSecond: 50, medianFinalLoss: 1.8 },
  }));
  assert.equal(result.compatible, true);
  assert.equal(result.metrics.totalDurationMs.percentChange, -20);
  assert.equal(result.metrics.meanExamplesPerSecond.percentChange, 25);
  assert.ok(Math.abs(result.metrics.medianFinalLoss.percentChange! - (-9.999999999999998)) < 1e-9);
});

test("refuses comparisons across different datasets", () => {
  assert.throws(() => compareTransformerProfileReports(report(), report({
    dataset: { sourceSha256: "b".repeat(64), tokenCount: 120 },
  })), /dataset fingerprint/i);
});

test("refuses comparisons across different model settings", () => {
  assert.throws(() => compareTransformerProfileReports(report(), report({
    model: { ...report().model as object, learningRate: 0.001 },
  })), /model.learningRate differs/i);
});

test("rejects malformed profile metrics", () => {
  assert.throws(() => compareTransformerProfileReports(report(), report({
    profile: { totalDurationMs: Number.NaN, meanEpochDurationMs: 20, meanExamplesPerSecond: 50, medianFinalLoss: 1.8 },
  })), /finite number/i);
});
