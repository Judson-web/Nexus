import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";
import { checkTransformerGradients } from "./transformer-gradient-check.js";

test("Transformer reverse-mode gradients match finite differences across parameter groups", () => {
  const config = { ...DEFAULT_TRANSFORMER_CONFIG, vocabSize: 7, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 13 };
  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  const groups = [
    0, 7 * 4 + 2 * 4 - 1, // token / position embeddings
    7 * 4 + 2 * 4, // query projection
    7 * 4 + 2 * 4 + 4 * 4, // key projection
    weights.length - 7 * 4 - 7, // language-model head
    weights.length - 1, // final vocabulary bias
  ];
  const result = checkTransformerGradients(config, weights, [1, 3], 2, { indices: groups });
  assert.equal(result.checkedParameters, groups.length);
  assert.deepEqual(result.failures, []);
  assert.ok(result.maxAbsoluteError < 2e-5, `max absolute gradient error: ${result.maxAbsoluteError}`);
});

test("gradient checker rejects duplicate indices and excessive work", () => {
  const config = { ...DEFAULT_TRANSFORMER_CONFIG, vocabSize: 7, contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4 };
  const weights = Float64Array.from(new TransformerModel(config).toCheckpoint().weights);
  assert.throws(() => checkTransformerGradients(config, weights, [1], 2, { indices: [0, 0] }), /unique/i);
  assert.throws(() => checkTransformerGradients(config, weights, [1], 2, { indices: Array.from({ length: 65 }, (_, i) => i) }), /64/i);
});
