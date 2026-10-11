import assert from "node:assert/strict";
import test from "node:test";
import { CausalAttentionModel, DEFAULT_CAUSAL_ATTENTION_CONFIG } from "./causal-attention-model.js";

const sample = [...new TextEncoder().encode("banana banana banana banana "), 256];

test("causal attention model trains a next-token predictor", () => {
  const model = new CausalAttentionModel({ ...DEFAULT_CAUSAL_ATTENTION_CONFIG, seed: 7, contextSize: 4, embeddingSize: 8 });
  const first = model.train(sample, 0.025, 1);
  const later = model.train(sample, 0.025, 15);
  assert.ok(Number.isFinite(first.loss));
  assert.ok(Number.isFinite(later.loss));
  assert.ok(later.loss < first.loss, `expected loss to fall from ${first.loss} to below it, got ${later.loss}`);
  assert.equal(later.examples, sample.length - 1);
});

test("causal attention probabilities are normalized and deterministic", () => {
  const model = new CausalAttentionModel({ ...DEFAULT_CAUSAL_ATTENTION_CONFIG, seed: 9, contextSize: 4, embeddingSize: 8 });
  const probabilities = model.probabilities([98, 97, 110]);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
  assert.ok(probabilities.every((value) => Number.isFinite(value) && value >= 0));
  assert.deepEqual(model.generate([98, 97], 8, () => 0.25), model.generate([98, 97], 8, () => 0.25));
});

test("causal attention checkpoints round-trip and reject invalid weights", () => {
  const model = new CausalAttentionModel({ ...DEFAULT_CAUSAL_ATTENTION_CONFIG, seed: 13, contextSize: 4, embeddingSize: 8 });
  model.train(sample, 0.025, 2);
  const checkpoint = model.toCheckpoint();
  const restored = CausalAttentionModel.fromCheckpoint(checkpoint);
  assert.deepEqual(restored.probabilities([98, 97, 110]), model.probabilities([98, 97, 110]));
  assert.throws(() => CausalAttentionModel.fromCheckpoint({ ...checkpoint, output: [1] }), /weight count mismatch/i);
});

test("causal attention rejects invalid token IDs and training parameters", () => {
  const model = new CausalAttentionModel();
  assert.throws(() => model.probabilities([999]), /invalid token/i);
  assert.throws(() => model.train([1, 2], 0, 1), /learningRate/i);
});
