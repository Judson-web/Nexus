import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";

const config = { ...DEFAULT_TRANSFORMER_CONFIG, contextSize: 3, embeddingSize: 4, headCount: 2, feedForwardSize: 8, seed: 19 };
const sample = [...new TextEncoder().encode("tiny transformer training data "), 256];

test("Transformer produces a normalized next-token distribution", () => {
  const model = new TransformerModel(config);
  const probabilities = model.probabilities([116, 105, 110]);
  assert.equal(probabilities.length, config.vocabSize);
  assert.ok(probabilities.every((p) => Number.isFinite(p) && p >= 0));
  assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-9);
});

test("Transformer backpropagation updates parameters and returns finite metrics", () => {
  const model = new TransformerModel(config);
  const before = model.toCheckpoint().weights;
  const metrics = model.train(sample, 0.0001, 1);
  const after = model.toCheckpoint().weights;
  assert.ok(Number.isFinite(metrics.loss));
  assert.ok(Number.isFinite(metrics.gradientNorm) && metrics.gradientNorm >= 0);
  assert.ok(metrics.gradientScale > 0 && metrics.gradientScale <= 1);
  assert.equal(metrics.examples, sample.length - 1);
  assert.ok(after.some((value, index) => value !== before[index]));
});

test("Transformer checkpoint round-trips and rejects malformed weights", () => {
  const model = new TransformerModel(config);
  const restored = TransformerModel.fromCheckpoint(model.toCheckpoint());
  assert.deepEqual(restored.probabilities([116, 105]), model.probabilities([116, 105]));
  assert.throws(
    () => TransformerModel.fromCheckpoint({ ...model.toCheckpoint(), weights: [1] }),
    /weight count mismatch/i,
  );
});

test("Transformer validates head dimensions, tokens, and learning rate", () => {
  assert.throws(() => new TransformerModel({ ...config, embeddingSize: 5 }), /divisible/i);
  const model = new TransformerModel(config);
  assert.throws(() => model.probabilities([999]), /invalid token/i);
  assert.throws(() => model.train([1, 2], 0, 1), /learningRate/i);
});

test("Transformer backpropagation lowers loss on a repeated next-token task", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 7 };
  const model = new TransformerModel(tinyConfig);
  const tokens = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
  const initial = model.toCheckpoint().weights;
  const before = model.probabilities([1])[2]!;
  const metrics = model.train(tokens, 0.01, 20);
  const after = model.probabilities([1])[2]!;
  assert.ok(Number.isFinite(metrics.loss));
  assert.equal(metrics.examples, tokens.length - 1);
  assert.ok(model.toCheckpoint().weights.some((value, index) => value !== initial[index]));
  assert.ok(after > before, "the repeated target token should become more probable");
});


test("Transformer gradient clipping bounds the averaged update norm", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 31 };
  const tokens = [1, 2, 3, 4, 1, 2, 3, 4];
  const clipped = new TransformerModel(tinyConfig);
  const metrics = clipped.train(tokens, 0.01, 1, 0.05);
  assert.ok(metrics.gradientNorm > 0.05);
  assert.ok(Math.abs(metrics.gradientScale - 0.05 / metrics.gradientNorm) < 1e-12);
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.01, 1, 0), /maxGradientNorm/i);
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.01, 1, Number.POSITIVE_INFINITY), /maxGradientNorm/i);
});

test("Transformer leaves gradients unclipped when below the configured threshold", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 37 };
  const metrics = new TransformerModel(tinyConfig).train([1, 2, 1, 2, 1], 0.001, 1, 1e6);
  assert.equal(metrics.gradientScale, 1);
});

test("Transformer Adam optimizer trains deterministically and validates its name", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 43 };
  const tokens = [1, 2, 1, 2, 1, 2, 1, 2];
  const first = new TransformerModel(tinyConfig);
  const second = new TransformerModel(tinyConfig);
  const metrics = first.train(tokens, 0.001, 3, 1, "adam");
  second.train(tokens, 0.001, 3, 1, "adam");
  assert.ok(Number.isFinite(metrics.loss));
  assert.deepEqual(first.toCheckpoint().weights, second.toCheckpoint().weights);
  assert.ok(first.toCheckpoint().weights.some((weight, i) => weight !== new TransformerModel(tinyConfig).toCheckpoint().weights[i]));
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.001, 1, 1, "bad" as "adam"), /optimizer/i);
});
