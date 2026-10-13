import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";
import { transformerLossAndGradients } from "./transformer-backprop.js";
import { transformerTapeLossAndGradients } from "./transformer-tape-backprop.js";

test("typed-array Transformer tape matches scalar reference loss and every parameter gradient", () => {
  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    vocabSize: 8,
    contextSize: 2,
    embeddingSize: 4,
    headCount: 2,
    feedForwardSize: 4,
    layerCount: 1,
    seed: 23,
  };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  for (const context of [[1, 2], [0], [7, 3, 2]]) {
    for (const target of [0, 3, 7]) {
      const reference = transformerLossAndGradients(config, weights, context, target);
      const tape = transformerTapeLossAndGradients(config, weights, context, target);
      assert.ok(Math.abs(tape.loss - reference.loss) < 1e-10, `loss mismatch: tape=${tape.loss}, reference=${reference.loss}`);
      assert.equal(tape.gradients.length, reference.gradients.length);
      for (let i = 0; i < reference.gradients.length; i++) {
        const expected = reference.gradients[i]!;
        const actual = tape.gradients[i]!;
        const tolerance = 2e-9 + 2e-8 * Math.max(Math.abs(expected), Math.abs(actual));
        assert.ok(Math.abs(actual - expected) <= tolerance,
          `gradient mismatch at ${i}: tape=${actual}, reference=${expected}, tolerance=${tolerance}`);
      }
    }
  }
});

test("typed-array Transformer gradients match finite differences", () => {
  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    vocabSize: 8,
    contextSize: 2,
    embeddingSize: 4,
    headCount: 2,
    feedForwardSize: 4,
    layerCount: 1,
    seed: 31,
  };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  const result = transformerTapeLossAndGradients(config, weights, [1, 2], 3);
  const groups = [
    0,
    config.vocabSize * config.embeddingSize,
    config.vocabSize * config.embeddingSize + config.contextSize * config.embeddingSize,
    weights.length - config.vocabSize - config.embeddingSize * config.vocabSize,
    weights.length - 1,
  ];
  const epsilon = 1e-5;
  for (const index of groups) {
    const original = weights[index]!;
    weights[index] = original + epsilon;
    const plus = transformerTapeLossAndGradients(config, weights, [1, 2], 3).loss;
    weights[index] = original - epsilon;
    const minus = transformerTapeLossAndGradients(config, weights, [1, 2], 3).loss;
    weights[index] = original;
    const numerical = (plus - minus) / (2 * epsilon);
    const analytic = result.gradients[index]!;
    const tolerance = 2e-3 + 2e-2 * Math.max(Math.abs(numerical), Math.abs(analytic));
    assert.ok(Math.abs(analytic - numerical) <= tolerance,
      `finite difference mismatch at ${index}: tape=${analytic}, numerical=${numerical}`);
  }
});

test("typed-array Transformer gradient path rejects malformed inputs", () => {
  const config = { ...DEFAULT_TRANSFORMER_CONFIG, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4 };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  assert.throws(() => transformerTapeLossAndGradients(config, weights.slice(1), [1], 2), /weight vector length/i);
  assert.throws(() => transformerTapeLossAndGradients(config, weights, [1], 8), /target token/i);
  assert.throws(() => transformerTapeLossAndGradients(config, weights, [8], 2), /invalid token/i);
});
