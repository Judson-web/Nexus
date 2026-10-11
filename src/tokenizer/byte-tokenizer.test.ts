import test from "node:test";
import assert from "node:assert/strict";
import { ByteTokenizer, BYTE_VOCAB_SIZE, EOS_TOKEN_ID } from "./byte-tokenizer.js";

test("encodes and decodes UTF-8 including non-ASCII text", () => {
  const tokenizer = new ByteTokenizer();
  const input = "Nexus λ 🧬";
  assert.equal(tokenizer.decode(tokenizer.encode(input)), input);
  assert.equal(BYTE_VOCAB_SIZE, 257);
  assert.equal(tokenizer.encode("x").at(-1), EOS_TOKEN_ID);
});

test("rejects invalid token IDs", () => {
  assert.throws(() => new ByteTokenizer().decode([257]), RangeError);
});
