import test from "node:test";
import assert from "node:assert/strict";
import { BytePairTokenizer } from "./byte-pair-tokenizer.js";
import { EOS_TOKEN_ID } from "./byte-tokenizer.js";

test("learns merges only from supplied training text and round-trips UTF-8", () => {
  const texts = ["the scientific paper presents a model", "the scientific paper presents results", "Nexus λ 🧬"];
  const tokenizer = BytePairTokenizer.train(texts, 80);
  assert.ok(tokenizer.vocabSize > 257);
  for (const text of texts) assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
  assert.ok(tokenizer.encode("the scientific paper", false).length < new TextEncoder().encode("the scientific paper").length);
});

test("serialized tokenizer reproduces token IDs and decoding", () => {
  const tokenizer = BytePairTokenizer.train(["repeat repeat repeat", "repeat a repeat"], 32);
  const restored = new BytePairTokenizer(tokenizer.toJSON());
  assert.deepEqual(restored.encode("repeat repeat"), tokenizer.encode("repeat repeat"));
  assert.equal(restored.decode(restored.encode("repeat repeat")), "repeat repeat");
});

test("rejects invalid tokens and malformed merges", () => {
  const tokenizer = new BytePairTokenizer();
  assert.throws(() => tokenizer.decode([257]), RangeError);
  assert.throws(() => new BytePairTokenizer({ format: "nexus-byte-pair-tokenizer", version: 1, merges: [{ left: 257, right: 1 }] }), /merge table/i);
  assert.equal(tokenizer.encode("x").at(-1), EOS_TOKEN_ID);
});
