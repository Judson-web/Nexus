import test from "node:test";
import assert from "node:assert/strict";
import { orderTrainingExamples } from "./data-ordering.js";

const examples = [
  { id: "a", text: "tiny" },
  { id: "b", text: "a much longer example" },
  { id: "c", text: "medium" },
  { id: "d", text: "the longest example in this little dataset" },
  { id: "e", text: "short" },
  { id: "f", text: "another medium sized example" },
];

test("length-cyclic ordering is deterministic and preserves each example exactly once", () => {
  const length = (text: string) => text.length;
  const first = orderTrainingExamples(examples, length, "length-cyclic");
  const second = orderTrainingExamples(examples, length, "length-cyclic");
  assert.deepEqual(first, second);
  assert.equal(first.length, examples.length);
  assert.deepEqual(new Set(first.map((example) => example.id)), new Set(examples.map((example) => example.id)));
  assert.notDeepEqual(first, examples);
});

test("manifest order preserves source order and invalid strategies fail closed", () => {
  assert.deepEqual(orderTrainingExamples(examples, (text) => text.length, "manifest"), examples);
  assert.throws(() => orderTrainingExamples(examples, () => -1, "length-cyclic"), /non-negative/i);
  assert.throws(() => orderTrainingExamples(examples, (text) => text.length, "unknown" as "manifest"), /data order/i);
});
