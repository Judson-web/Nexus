import { readFile } from "node:fs/promises";
import { validateDatasetManifest } from "../evaluation/manifest.js";
import { ByteTokenizer, EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";

function percentiles(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))] ?? 0;
  return { min: sorted[0] ?? 0, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: sorted.at(-1) ?? 0 };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== "--manifest") throw new Error("Usage: npm run audit:tokenizer -- --manifest <path>");
  const raw = await readFile(args[1]!, "utf8");
  const manifest = validateDatasetManifest(JSON.parse(raw) as unknown);
  const tokenizer = new ByteTokenizer();
  const summaries = Object.fromEntries((["train", "validation", "test"] as const).map((split) => {
    const examples = manifest.examples.filter((example) => example.split === split);
    const byteLengths = examples.map((example) => tokenizer.encode(example.text, false).length);
    const codePointLengths = examples.map((example) => Array.from(example.text).length);
    const utf8Bytes = examples.reduce((sum, example) => sum + Buffer.byteLength(example.text, "utf8"), 0);
    const encodedBytes = examples.reduce((sum, example) => sum + tokenizer.encode(example.text, false).length, 0);
    const eosTargets = examples.length;
    return [split, {
      examples: examples.length,
      utf8Bytes,
      encodedBytes,
      byteCoverage: utf8Bytes === 0 ? 1 : encodedBytes / utf8Bytes,
      unicodeCodePoints: codePointLengths.reduce((sum, n) => sum + n, 0),
      bytesPerCodePoint: codePointLengths.reduce((sum, n) => sum + n, 0) === 0 ? 0 : encodedBytes / codePointLengths.reduce((sum, n) => sum + n, 0),
      sequenceBytes: percentiles(byteLengths),
      nextTokenTargetsIncludingEos: encodedBytes + eosTargets,
    }];
  }));
  const report = {
    format: "nexus-tokenizer-audit",
    version: 1,
    tokenizer: { type: "UTF-8 byte-level", vocabularySize: 257, eosTokenId: EOS_TOKEN_ID, merges: 0, learnedVocabulary: false },
    dataset: { acceptedExamples: manifest.acceptedExamples, splits: manifest.splits, sourceSha256: manifest.sourceSha256, preparedSha256: manifest.preparedSha256 },
    splits: summaries,
    interpretation: "Byte coverage should be exactly 1.0 for valid UTF-8 text, but byte tokenization uses more sequence positions than subword tokenization. Byte-length percentiles show the context-window mismatch; they do not measure semantic quality.",
  };
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
