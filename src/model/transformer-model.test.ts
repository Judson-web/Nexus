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

test("Transformer reports bounded token-level training progress", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 53 };
  const progress: Array<{ epoch: number; epochs: number; targetsProcessed: number; totalTargets: number }> = [];
  const tokens = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
  new TransformerModel(tinyConfig).train(tokens, 0.001, 2, 1, "adam", (entry) => progress.push(entry));
  assert.ok(progress.length > 0);
  assert.deepEqual(progress.at(-1), { epoch: 2, epochs: 2, targetsProcessed: tokens.length - 1, totalTargets: tokens.length - 1 });
  assert.ok(progress.every((entry) => entry.targetsProcessed > 0 && entry.targetsProcessed <= entry.totalTargets));
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


test("Transformer generation suppresses premature EOS for a bounded minimum continuation", () => {
  const model = new TransformerModel({ ...config, vocabSize: 257 });
  const generated = model.generate([84, 101, 115, 116], 4, () => 0.999999, 4);
  assert.equal(generated.length, 8);
  assert.ok(generated.slice(4).every((token) => token !== 256), "EOS must not terminate the minimum continuation");
  assert.throws(() => model.generate([1], 2, () => 0.5, -1), /minTokens/);
});

test("Transformer generation resumes normal EOS sampling after the minimum continuation", () => {
  const model = new TransformerModel({ ...config, vocabSize: 257 });
  const generated = model.generate([84], 3, () => 0.999999, 1);
  assert.equal(generated.length, 4);
});


test("Transformer supports deterministic strided target sampling and reports sampled target counts", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 71 };
  const tokens = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3];
  const progress: Array<{ epoch: number; epochs: number; targetsProcessed: number; totalTargets: number }> = [];
  const model = new TransformerModel(tinyConfig);
  const metrics = model.train(tokens, 0.001, 1, 1, "adam", (entry) => progress.push(entry), 2);
  assert.equal(metrics.examples, 5);
  assert.ok(Number.isFinite(metrics.loss));
  assert.deepEqual(progress.at(-1), { epoch: 1, epochs: 1, targetsProcessed: 5, totalTargets: 5 });
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.001, 1, 1, "adam", undefined, 0), /targetStride/);
});

test("Transformer AdamW applies decoupled weight decay with a cosine schedule", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 83 };
  const tokens = [1, 2, 1, 2, 1, 2, 1, 2];
  const first = new TransformerModel(tinyConfig);
  const second = new TransformerModel(tinyConfig);
  const metrics = first.train(tokens, 0.001, 3, 1, "adamw", undefined, 1, 0.01, "cosine");
  second.train(tokens, 0.001, 3, 1, "adamw", undefined, 1, 0.01, "cosine");
  assert.ok(Number.isFinite(metrics.loss));
  assert.deepEqual(first.toCheckpoint().weights, second.toCheckpoint().weights);
  assert.ok(first.toCheckpoint().weights.some((weight, index) => weight !== new TransformerModel(tinyConfig).toCheckpoint().weights[index]));
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.001, 1, 1, "adamw", undefined, 1, -0.1), /weightDecay/i);
  assert.throws(() => new TransformerModel(tinyConfig).train(tokens, 0.001, 1, 1, "adam", undefined, 1, 0.01), /only by adamw/i);
});

test("Transformer checkpoint preserves AdamW moments and exact continuation", () => {
  const tinyConfig = { ...config, vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 89 };
  const tokens = [1, 2, 1, 2, 1, 2, 1, 2];
  const uninterrupted = new TransformerModel(tinyConfig);
  uninterrupted.train(tokens, 0.001, 2, 1, "adamw", undefined, 1, 0.01, "cosine");
  const restored = TransformerModel.fromCheckpoint(uninterrupted.toCheckpoint());
  uninterrupted.train(tokens, 0.001, 1, 1, "adamw", undefined, 1, 0.01, "cosine");
  restored.train(tokens, 0.001, 1, 1, "adamw", undefined, 1, 0.01, "cosine");
  assert.deepEqual(restored.toCheckpoint().weights, uninterrupted.toCheckpoint().weights);
  assert.deepEqual(restored.toCheckpoint().optimizerState, uninterrupted.toCheckpoint().optimizerState);
  const malformed = { ...restored.toCheckpoint(), optimizerState: { step: -1, firstMoment: [], secondMoment: [] } };
  assert.throws(() => TransformerModel.fromCheckpoint(malformed), /optimizer state/i);
});

test("Transformer supports independently parameterized decoder depth", () => {
  const shallow = new TransformerModel({ ...config, vocabSize: 8, layerCount: 1 });
  const deep = new TransformerModel({ ...config, vocabSize: 8, layerCount: 3 });
  assert.equal(shallow.config.layerCount, 1);
  assert.equal(deep.config.layerCount, 3);
  assert.ok(deep.toCheckpoint().weights.length > shallow.toCheckpoint().weights.length * 2);
  const before = deep.toCheckpoint().weights;
  const metrics = deep.train([1, 2, 3, 1, 2, 3], 0.001, 1, 1, "adamw");
  const after = deep.toCheckpoint().weights;
  assert.ok(Number.isFinite(metrics.loss));
  assert.ok(after.some((weight, index) => weight !== before[index]));
  assert.deepEqual(TransformerModel.fromCheckpoint(deep.toCheckpoint()).probabilities([1, 2]), deep.probabilities([1, 2]));
});

test("Legacy single-block checkpoints remain inference-compatible", () => {
  const legacyConfig = { vocabSize: 8, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 97 };
  const oldWeightCount = legacyConfig.vocabSize * legacyConfig.embeddingSize +
    legacyConfig.contextSize * legacyConfig.embeddingSize +
    4 * legacyConfig.embeddingSize ** 2 + 5 * legacyConfig.embeddingSize +
    2 * legacyConfig.embeddingSize * legacyConfig.feedForwardSize +
    legacyConfig.feedForwardSize + legacyConfig.vocabSize * legacyConfig.embeddingSize + legacyConfig.vocabSize;
  const legacy = TransformerModel.fromCheckpoint({
    format: "nexus-transformer", version: 1, config: legacyConfig, weights: new Array(oldWeightCount).fill(0),
  });
  const probabilities = legacy.probabilities([1, 2]);
  assert.equal(probabilities.length, legacyConfig.vocabSize);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
  assert.throws(() => legacy.train([1, 2, 3]), /inference-only/i);
  const roundTrip = TransformerModel.fromCheckpoint(legacy.toCheckpoint());
  assert.deepEqual(roundTrip.probabilities([1, 2]), probabilities);
});
