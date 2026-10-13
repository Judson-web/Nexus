import test from "node:test";
import assert from "node:assert/strict";
import { makeOverlappingTokenWindows } from "./token-windows.js";

test("overlapping windows preserve every next-token target exactly once", () => {
  const tokens = Array.from({ length: 23 }, (_, i) => i);
  const windows = makeOverlappingTokenWindows(tokens, 8, 3);
  assert.ok(windows.length > 1);
  const targets = windows.flatMap((window) => window.tokens.slice(window.targetStartIndex).map((token) => token));
  assert.deepEqual(targets, tokens.slice(1));
  assert.deepEqual(windows[1]?.tokens.slice(0, 3), [5, 6, 7]);
  assert.equal(windows[1]?.targetStartIndex, 3);
});

test("overlapping windows validate dimensions and handle short inputs", () => {
  assert.deepEqual(makeOverlappingTokenWindows([1], 8, 3), []);
  assert.throws(() => makeOverlappingTokenWindows([1, 2], 1, 0), /maxTokensPerWindow/i);
  assert.throws(() => makeOverlappingTokenWindows([1, 2], 8, 8), /contextOverlap/i);
});
