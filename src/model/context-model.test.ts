import assert from "node:assert/strict";
import test from "node:test";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "./context-model.js";

const sample = new TextEncoder().encode("banana banana banana banana ").reduce<number[]>((all, byte) => { all.push(byte); return all; }, []);
sample.push(256);

test("context model trains a neural next-token predictor and lowers loss", () => {
  const model = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: 7, contextSize: 3, embeddingSize: 8, hiddenSize: 16 });
  const first = model.train(sample, 0.04, 1);
  const later = model.train(sample, 0.04, 12);
  assert.ok(Number.isFinite(first.loss));
  assert.ok(Number.isFinite(later.loss));
  assert.ok(later.loss < first.loss, `expected loss to fall from ${first.loss} to below it, got ${later.loss}`);
  assert.equal(later.examples, sample.length);
});

test("context model probabilities are normalized and deterministic with a seeded draw", () => {
  const model = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: 9, contextSize: 3, embeddingSize: 8, hiddenSize: 16 });
  const probabilities = model.probabilities([98, 97, 110]);
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(total - 1) < 1e-9);
  assert.ok(probabilities.every((value) => Number.isFinite(value) && value >= 0));
  assert.deepEqual(model.generate([98, 97], 8, () => 0.25), model.generate([98, 97], 8, () => 0.25));
});

test("context checkpoints round-trip and reject malformed dimensions", () => {
  const model = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: 13, contextSize: 3, embeddingSize: 8, hiddenSize: 16 });
  model.train(sample, 0.03, 2);
  const restored = ContextModel.fromCheckpoint(model.toCheckpoint());
  assert.deepEqual(Array.from(restored.probabilities([98, 97, 110])), Array.from(model.probabilities([98, 97, 110])));
  const invalid = { ...model.toCheckpoint(), w1: [1, 2] };
  assert.throws(() => ContextModel.fromCheckpoint(invalid), /weight count mismatch/i);
});
