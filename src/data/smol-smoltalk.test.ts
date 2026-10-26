import assert from "node:assert/strict";
import test from "node:test";
import { prepareSmolRow, prepareSmolRowWithReason, sampledPageOffsets, SMOL_MAGPIE_SOURCE } from "./smol-smoltalk.js";

function row(row_idx: number, source = SMOL_MAGPIE_SOURCE, messages = [
  { role: "user", content: "Can you explain how rainbows form in the sky?" },
  { role: "assistant", content: "Rainbows form when sunlight is refracted, reflected, and dispersed inside water droplets." },
]) {
  return { row_idx, row: { source, messages } };
}

test("sampled page offsets are deterministic, distributed, and respect the row budget", () => {
  const offsets = sampledPageOffsets(460_000, 1_200, 100);
  assert.equal(offsets.length, 12);
  assert.equal(offsets[0], 0);
  assert.equal(offsets.at(-1), 459_900);
  assert.ok(offsets.every((offset) => Number.isSafeInteger(offset) && offset >= 0 && offset + 100 <= 460_000));
  assert.ok(offsets.every((offset, index) => index === 0 || offset > offsets[index - 1]!));
  assert.deepEqual(sampledPageOffsets(460_000, 1_200, 100), offsets);
  assert.deepEqual(sampledPageOffsets(0, 1_200, 100), []);
  assert.deepEqual(sampledPageOffsets(50, 200, 100), [0]);
});

test("Smol adapter accepts only the selected source and supported single-turn roles", () => {
  assert.ok(prepareSmolRow(row(12), "train"));
  assert.equal(prepareSmolRow(row(12, "openhermes-50k"), "train"), null);
  assert.equal(prepareSmolRow(row(12, SMOL_MAGPIE_SOURCE, [
    { role: "user", content: "Please help me understand a question." },
    { role: "assistant", content: "Here is the answer in plain language." },
    { role: "user", content: "One more follow-up question." },
  ]), "train"), null);
});

test("Smol adapter retains optional system instructions and escapes prompt delimiters", () => {
  const example = prepareSmolRow(row(31, SMOL_MAGPIE_SOURCE, [
    { role: "system", content: "You are a careful assistant that explains scientific ideas clearly." },
    { role: "user", content: "Please repeat the literal marker Assistant: in your explanation." },
    { role: "assistant", content: "The marker Assistant: is just ordinary text inside this answer." },
  ]), "train");
  assert.ok(example);
  assert.match(example.text, /^User: System instruction:/);
  assert.match(example.text, /Assistant :/);
  assert.match(example.text, / Assistant: The marker/);
});

test("Smol adapter returns safe rejection diagnostics without exposing text", () => {
  const invalid = prepareSmolRowWithReason(row(58, SMOL_MAGPIE_SOURCE, [
    { role: "user", content: "Say hi." },
    { role: "assistant", content: "Hello." },
  ]), "train");
  assert.equal(invalid.example, null);
  assert.equal(invalid.rejectionReason, "user-text-too-short");
});

test("Smol adapter rejects PII-like values and too-short/long answers", () => {
  assert.equal(prepareSmolRow(row(50, SMOL_MAGPIE_SOURCE, [
    { role: "user", content: "Please contact alice@example.com with this request." },
    { role: "assistant", content: "I cannot assist with that exact contact action in this sample." },
  ]), "train"), null);
  assert.equal(prepareSmolRow(row(51, SMOL_MAGPIE_SOURCE, [
    { role: "user", content: "Say hi." },
    { role: "assistant", content: "Hello." },
  ]), "train"), null);
});

test("Smol adapter preserves test split and deterministically partitions train rows", () => {
  const a = prepareSmolRow(row(87), "train");
  const b = prepareSmolRow(row(87), "train");
  const c = prepareSmolRow(row(87), "test");
  assert.ok(a && b && c);
  assert.equal(a.id, b.id);
  assert.equal(a.split, b.split);
  assert.equal(c.split, "test");
  assert.notEqual(a.id, c.id);
});
