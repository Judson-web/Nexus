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

test("drops incomplete or invalid UTF-8 suffixes instead of rendering replacement characters", () => {
  const tokenizer = new ByteTokenizer();
  assert.equal(tokenizer.decode([72, 105, 0xf0, 0x9f]), "Hi");
  assert.equal(tokenizer.decode([0xe2, 0x28, 0xa1]), "");
  assert.equal(tokenizer.decode([72, 105, 0xf0, 0x9f, EOS_TOKEN_ID, 65]), "Hi");
});
