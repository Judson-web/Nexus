import test from "node:test";
import assert from "node:assert/strict";
import { COMMON_PILE_ARXIV_ABSTRACTS, reviewRows, splitForSourceId, type CommonPileRow } from "./common-pile.js";

function row(id: string, text = `This is a sufficiently long scientific abstract about a reproducible experimental result in a research field, with methods and findings.`): CommonPileRow {
  return {
    id,
    text,
    source: "arxiv-abstracts",
    metadata: {
      license: COMMON_PILE_ARXIV_ABSTRACTS.license,
      full_text_license: "None",
      authors: "Research Author",
      url: `https://arxiv.org/abs/${id}`,
    },
  };
}

test("Common Pile selection accepts only explicit CC0 abstract records with arXiv provenance", () => {
  const accepted = row("2401.00001");
  const wrongLicense = { ...row("2401.00002"), metadata: { ...row("2401.00002").metadata, license: "CC-BY-NC" } };
  const wrongSource = { ...row("2401.00003"), source: "unknown-source" };
  const wrongUrl = { ...row("2401.00004"), metadata: { ...row("2401.00004").metadata, url: "https://example.com/paper" } };
  const result = reviewRows([accepted, wrongLicense, wrongSource, wrongUrl], 3);
  assert.equal(result.accepted.length, 1);
  assert.equal(result.rejectedLicenseOrMetadata, 3);
  assert.equal(result.accepted[0]?.license, COMMON_PILE_ARXIV_ABSTRACTS.license);
  assert.equal(result.accepted[0]?.sourceUrl, "https://arxiv.org/abs/2401.00001");
});

test("document split is deterministic and independent of row order", () => {
  assert.equal(splitForSourceId("2401.00001"), splitForSourceId("2401.00001"));
  assert.throws(() => splitForSourceId("  "), /must not be empty/i);
  const ids = Array.from({ length: 300 }, (_, i) => `2401.${String(i).padStart(5, "0")}`);
  const rows = ids.map((id) => row(id, `Unique abstract for ${id}: this scientific record describes a research method, an experimental observation, and a reproducible finding in its field.`));
  const first = reviewRows(rows, 200).accepted.map((item) => [item.id, item.split] as [string, string]);
  const second = reviewRows([...rows].reverse(), 200).accepted.map((item) => [item.id, item.split]);
  assert.deepEqual(new Map(first), new Map(second));
  assert.deepEqual(new Set(first.map(([, split]) => split)), new Set(["train", "validation", "test"]));
});

test("exact duplicate abstracts are kept in only one split", () => {
  const first = row("2401.00001", "The same normalized abstract is deliberately repeated for this duplicate test with enough characters.");
  const duplicate = row("2401.00002", "The same normalized abstract is deliberately repeated for this duplicate test with enough characters.");
  const result = reviewRows([first, duplicate], 3);
  assert.equal(result.accepted.length, 1);
  assert.equal(result.duplicateDocuments, 1);
});

test("sample size is bounded", () => {
  assert.throws(() => reviewRows([], 0), /between 3 and 5000/i);
});
