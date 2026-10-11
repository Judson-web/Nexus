import test from "node:test";
import assert from "node:assert/strict";
import { prepareDataset } from "./dataset.js";

test("dataset preparation normalizes, deduplicates, hashes, and assigns stable splits", () => {
  const source = "\uFEFFAlpha  line\r\n\r\nBeta line\nAlpha line\nGamma line\nDelta line\nEpsilon line\nZeta line\nEta line\nTheta line\nIota line\nKappa line";
  const first = prepareDataset(source, { seed: 9 });
  const second = prepareDataset(source, { seed: 9 });
  assert.deepEqual(first, second);
  assert.equal(first.duplicatesRemoved, 1);
  assert.equal(first.emptyLinesRemoved, 1);
  assert.equal(first.acceptedExamples, 10);
  assert.equal(first.splits.train + first.splits.validation + first.splits.test, first.acceptedExamples);
  assert.match(first.sourceSha256, /^[a-f0-9]{64}$/);
  assert.match(first.preparedSha256, /^[a-f0-9]{64}$/);
  assert.ok(first.examples.every((example) => /^[a-f0-9]{64}$/.test(example.sha256)));
});

test("split assignment does not depend on source line ordering", () => {
  const a = prepareDataset("alpha\nbeta\ngamma\ndelta\nepsilon", { seed: 4 });
  const b = prepareDataset("epsilon\ndelta\ngamma\nbeta\nalpha", { seed: 4 });
  assert.equal(a.preparedSha256, b.preparedSha256);
  assert.deepEqual(a.examples.map((item) => [item.sha256, item.split]), b.examples.map((item) => [item.sha256, item.split]));
  assert.notEqual(a.sourceSha256, b.sourceSha256);
});

test("dataset preparation rejects invalid configuration", () => {
  assert.throws(() => prepareDataset("abc", { seed: -1 }), /seed must be/i);
  assert.throws(() => prepareDataset("abc", { trainRatio: 0.7, validationRatio: 0.1, testRatio: 0.1 }), /sum to 1/i);
  assert.throws(() => prepareDataset("abc", { minimumCharacters: 0 }), /minimumCharacters/i);
});
