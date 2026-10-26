import assert from "node:assert/strict";
import test from "node:test";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";
import { EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";
import { encodeTrainingExample } from "./supervised-example.js";

const pairs = [
  "User: How are you feeling today? Assistant: I am doing well, thank you.",
  "User: Summarize this short note. Assistant: The meeting moved to Tuesday.",
  "User: Explain the result. Assistant: The value increased by two.",
];
const tokenizer = BytePairTokenizer.train(pairs, 120);

test("instruction pairs encode prompt context separately and target assistant text only", () => {
  const text = pairs[0]!;
  const separator = " Assistant: ";
  const split = text.indexOf(separator);
  const prefix = text.slice(0, split + separator.length);
  const response = text.slice(split + separator.length);
  const encoded = encodeTrainingExample(text, tokenizer, 32);
  assert.deepEqual(encoded.tokens.slice(0, encoded.lossStartIndex), tokenizer.encode(prefix, false));
  assert.deepEqual(encoded.tokens.slice(encoded.lossStartIndex, -1), tokenizer.encode(response, false));
  assert.equal(encoded.tokens.at(-1), EOS_TOKEN_ID);
  assert.equal(encoded.lossPolicy, "assistant-only-for-user-assistant-pairs-v1");
  assert.equal(encoded.truncated, false);
});

test("long instruction prompts keep only the suffix available to the model's context window", () => {
  const text = "User: " + "earlier context ".repeat(100) + " Assistant: Keep this final answer.";
  const marker = text.indexOf(" Assistant: ");
  const prefix = text.slice(0, marker + " Assistant: ".length);
  const encoded = encodeTrainingExample(text, tokenizer, 8, 32);
  assert.equal(encoded.lossStartIndex, 8);
  assert.equal(encoded.tokens.length, tokenizer.encode("Keep this final answer.", false).length + 9);
  assert.deepEqual(encoded.tokens.slice(0, 8), tokenizer.encode(prefix, false).slice(-8));
});

test("overlong responses are truncated without adding a false end-of-turn token", () => {
  const text = "User: Say many words. Assistant: " + "long answer ".repeat(80);
  const encoded = encodeTrainingExample(text, tokenizer, 8, 24);
  assert.equal(encoded.tokens.length, 24);
  assert.equal(encoded.lossStartIndex, 8);
  assert.equal(encoded.truncated, true);
  assert.notEqual(encoded.tokens.at(-1), EOS_TOKEN_ID);
});

test("plain-text examples retain the full-sequence objective", () => {
  const encoded = encodeTrainingExample("ordinary pretraining text", tokenizer, 8, 32);
  assert.equal(encoded.lossStartIndex, 1);
  assert.equal(encoded.lossPolicy, "full-sequence-next-token-v1");
  assert.equal(encoded.tokens.at(-1), EOS_TOKEN_ID);
});

test("encoding rejects invalid context and truncation settings", () => {
  assert.throws(() => encodeTrainingExample("text", tokenizer, 0), /contextSize/i);
  assert.throws(() => encodeTrainingExample("text", tokenizer, 8, 8), /maxTokens/i);
});
