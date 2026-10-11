import test from "node:test";
import assert from "node:assert/strict";
import { BigramModel } from "./bigram-model.js";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";

test("training improves loss on a repeated tiny corpus", () => {
  const tokens = new ByteTokenizer().encode("abababababababababababab");
  const model = new BigramModel({ vocabSize: 257, seed: 4, initializationScale: 0.01 });
  const before = model.train(tokens, 0.05, 1).loss;
  const after = model.train(tokens, 0.05, 25).loss;
  assert.ok(after < before, `expected loss to decrease: before=${before}, after=${after}`);
});

test("checkpoint round trip preserves probabilities", () => {
  const model = new BigramModel({ vocabSize: 257, seed: 8, initializationScale: 0.01 });
  const restored = BigramModel.fromCheckpoint(model.toCheckpoint());
  assert.deepEqual(Array.from(restored.probabilities(65)), Array.from(model.probabilities(65)));
});

test("checkpoint rejects unsupported formats", () => {
  assert.throws(() => BigramModel.fromCheckpoint({ format: "unknown", version: 1 }));
});
