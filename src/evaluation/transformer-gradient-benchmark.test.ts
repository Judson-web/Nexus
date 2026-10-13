import assert from "node:assert/strict";
import test from "node:test";
import { benchmarkTransformerGradients } from "./transformer-gradient-benchmark.js";

test("gradient benchmark reports deterministic inputs and numerical parity", () => {
  const options = { repeats: 3, warmups: 0, examples: 3, config: {
    vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, layerCount: 1, seed: 19,
  } };
  const first = benchmarkTransformerGradients(options);
  const second = benchmarkTransformerGradients(options);
  assert.equal(first.input.sha256, second.input.sha256);
  assert.equal(first.input.comparedGradients, first.input.parameterCount * options.examples);
  assert.equal(first.parity.withinTolerance, true);
  assert.ok(first.measurements.scalarMedianMs > 0);
  assert.ok(first.measurements.tapeMedianMs > 0);
  assert.ok(first.measurements.speedup > 0);
});

test("gradient benchmark validates measurement bounds and model dimensions", () => {
  assert.throws(() => benchmarkTransformerGradients({ repeats: 2 }), /repeats/);
  assert.throws(() => benchmarkTransformerGradients({ config: { embeddingSize: 3, headCount: 2 } }), /configuration/);
});
