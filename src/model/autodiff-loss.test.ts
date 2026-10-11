import assert from "node:assert/strict";
import test from "node:test";
import { Value } from "./autodiff.js";
import { softmaxCrossEntropy } from "./autodiff-loss.js";

test("softmax cross-entropy produces the expected stable loss", () => {
  const logits = [Value.of(2), Value.of(1), Value.of(0)];
  const loss = softmaxCrossEntropy(logits, 0);
  const expected = Math.log(Math.exp(0) + Math.exp(-1) + Math.exp(-2));
  assert.ok(Math.abs(loss.data - expected) < 1e-12);
});

test("softmax cross-entropy gradient is softmax minus one-hot target", () => {
  const logits = [Value.of(0.2), Value.of(-0.4), Value.of(1.1)];
  const loss = softmaxCrossEntropy(logits, 1);
  loss.backward();
  const exp = logits.map((x) => Math.exp(x.data - 1.1));
  const sum = exp.reduce((a, b) => a + b, 0);
  for (let i = 0; i < logits.length; i++) {
    const expected = exp[i]! / sum - (i === 1 ? 1 : 0);
    assert.ok(Math.abs(logits[i]!.grad - expected) < 1e-12);
  }
});

test("softmax cross-entropy remains stable for large logits", () => {
  const logits = [Value.of(10000), Value.of(9999), Value.of(9998)];
  const loss = softmaxCrossEntropy(logits, 0);
  assert.ok(Number.isFinite(loss.data));
  loss.backward();
  assert.ok(logits.every((logit) => Number.isFinite(logit.grad)));
});

test("softmax cross-entropy validates targets and dimensions", () => {
  assert.throws(() => softmaxCrossEntropy([Value.of(1)], 0), /at least two/i);
  assert.throws(() => softmaxCrossEntropy([Value.of(1), Value.of(2)], 2), /target index/i);
});
