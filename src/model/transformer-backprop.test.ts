import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG } from "./transformer-model.js";
import { TransformerModel } from "./transformer-model.js";
import { transformerLossAndGradients } from "./transformer-backprop.js";

test("Transformer analytic gradients match finite differences across parameter groups", () => {
  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    vocabSize: 8,
    contextSize: 2,
    embeddingSize: 4,
    headCount: 2,
    feedForwardSize: 4,
    seed: 23,
  };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  const context = [1, 2];
  const target = 3;
  const result = transformerLossAndGradients(config, weights, context, target);
  const epsilon = 1e-5;
  const groups = [
    0, // token embeddings
    config.vocabSize * config.embeddingSize, // position embeddings
    config.vocabSize * config.embeddingSize + config.contextSize * config.embeddingSize, // query projection
    config.vocabSize * config.embeddingSize + config.contextSize * config.embeddingSize + config.embeddingSize ** 2, // key projection
    config.vocabSize * config.embeddingSize + config.contextSize * config.embeddingSize + 2 * config.embeddingSize ** 2, // value projection
    weights.length - config.vocabSize - config.embeddingSize * config.vocabSize, // language-model head
    weights.length - 1, // final language-model bias
  ];

  for (const index of groups) {
    const original = weights[index]!;
    weights[index] = original + epsilon;
    const plus = transformerLossAndGradients(config, weights, context, target).loss;
    weights[index] = original - epsilon;
    const minus = transformerLossAndGradients(config, weights, context, target).loss;
    weights[index] = original;
    const numerical = (plus - minus) / (2 * epsilon);
    const analytic = result.gradients[index]!;
    const tolerance = 2e-3 + 2e-2 * Math.max(Math.abs(numerical), Math.abs(analytic));
    assert.ok(
      Math.abs(analytic - numerical) <= tolerance,
      `gradient mismatch at parameter ${index}: analytic=${analytic}, numerical=${numerical}`,
    );
  }
});

test("Transformer gradient computation rejects malformed inputs", () => {
  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    vocabSize: 8,
    contextSize: 2,
    embeddingSize: 4,
    headCount: 2,
    feedForwardSize: 4,
  };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  assert.throws(() => transformerLossAndGradients(config, weights.slice(1), [1], 2), /weight vector length/i);
  assert.throws(() => transformerLossAndGradients(config, weights, [1], 8), /target token/i);
  assert.throws(() => transformerLossAndGradients(config, weights, [8], 2), /invalid token/i);
});
