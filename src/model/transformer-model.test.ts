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
