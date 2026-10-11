import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";
import { transformerLossAndGradients } from "./transformer-gradient.js";

const config = { ...DEFAULT_TRANSFORMER_CONFIG, vocabSize: 9, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 5, seed: 37 };

test("Transformer analytical gradients match finite differences across parameter groups", () => {
  const model = new TransformerModel(config);
  const weights = model.toCheckpoint().weights;
  const context = [2, 3];
  const target = 4;
  const result = transformerLossAndGradients(config, weights, context, target);
  const indices = [0, 9, 9 * 4, 9 * 4 + 2, 9 * 4 + 2 * 4 + 1, weights.length - 1];
  const epsilon = 1e-5;
  for (const index of indices) {
    const plus = [...weights];
    const minus = [...weights];
    plus[index] = plus[index]! + epsilon;
    minus[index] = minus[index]! - epsilon;
    const numeric = (
      transformerLossAndGradients(config, plus, context, target).loss -
      transformerLossAndGradients(config, minus, context, target).loss
    ) / (2 * epsilon);
    assert.ok(Math.abs(result.gradients[index]! - numeric) < 2e-4,
      `gradient mismatch at parameter ${index}: analytical=${result.gradients[index]}, numeric=${numeric}`);
  }
});

test("Transformer analytical gradient path returns finite loss and gradients", () => {
  const model = new TransformerModel(config);
  const result = transformerLossAndGradients(config, model.toCheckpoint().weights, [1, 2], 3);
  assert.ok(Number.isFinite(result.loss));
  assert.ok(result.gradients.every(Number.isFinite));
  assert.equal(result.gradients.length, model.toCheckpoint().weights.length);
});

test("Transformer training uses analytical gradients and changes parameters", () => {
  const model = new TransformerModel(config);
  const before = model.toCheckpoint().weights;
  const metrics = model.train([1, 2, 3, 4, 5], 0.0001, 1);
  const after = model.toCheckpoint().weights;
  assert.ok(Number.isFinite(metrics.loss));
  assert.equal(metrics.examples, 4);
  assert.ok(after.some((value, index) => value !== before[index]));
});
