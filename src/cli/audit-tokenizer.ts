import { readFile } from "node:fs/promises";
import { validateDatasetManifest } from "../evaluation/manifest.js";
import { EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";

function percentiles(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))] ?? 0;
  return { min: sorted[0] ?? 0, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: sorted.at(-1) ?? 0 };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    const value = args[i + 1];
    if (!key || !value || !key.startsWith("--") || options.has(key.slice(2))) throw new Error("Usage: audit-tokenizer.ts --manifest <path> [--max-tokens-per-example <16-4096>]");
    options.set(key.slice(2), value);
  }
  for (const key of options.keys()) if (!["manifest", "max-tokens-per-example"].includes(key)) throw new Error(`Unknown option: --${key}`);
  const manifestPath = options.get("manifest");
  if (!manifestPath) throw new Error("Missing required --manifest");
  const trainingLimit = Number(options.get("max-tokens-per-example") ?? "256");
  if (!Number.isSafeInteger(trainingLimit) || trainingLimit < 16 || trainingLimit > 4096) throw new RangeError("max-tokens-per-example must be an integer between 16 and 4096.");
  const raw = await readFile(manifestPath, "utf8");
  const manifest = validateDatasetManifest(JSON.parse(raw) as unknown);
  const tokenizer = BytePairTokenizer.train(manifest.examples.filter((example) => example.split === "train").map((example) => example.text), 255);
  const summaries = Object.fromEntries((["train", "validation", "test"] as const).map((split) => {
    const examples = manifest.examples.filter((example) => example.split === split);
    const tokenLengths = examples.map((example) => tokenizer.encode(example.text, false).length);
    const codePointLengths = examples.map((example) => Array.from(example.text).length);
    const targetLengthsIncludingEos = tokenLengths.map((length) => length + 1);
    const truncatedTokens = targetLengthsIncludingEos.map((length) => Math.max(0, length - trainingLimit));
    const utf8Bytes = examples.reduce((sum, example) => sum + Buffer.byteLength(example.text, "utf8"), 0);
    const encodedTokens = examples.reduce((sum, example) => sum + tokenizer.encode(example.text, false).length, 0);
    const eosTargets = examples.length;
    const totalCodePoints = codePointLengths.reduce((sum, n) => sum + n, 0);
    return [split, {
      examples: examples.length,
      utf8Bytes,
      encodedTokens,
      tokenToByteRatio: utf8Bytes === 0 ? 0 : encodedTokens / utf8Bytes,
      unicodeCodePoints: totalCodePoints,
      tokensPerCodePoint: totalCodePoints === 0 ? 0 : encodedTokens / totalCodePoints,
      sequenceTokens: percentiles(tokenLengths),
      trainingMaxTokensPerExample: trainingLimit,
      examplesExceedingTrainingLimit: truncatedTokens.filter((count) => count > 0).length,
      estimatedTruncatedTokens: truncatedTokens.reduce((sum, count) => sum + count, 0),
      nextTokenTargetsIncludingEos: encodedTokens + eosTargets,
    }];
  }));
  const report = {
    format: "nexus-tokenizer-audit",
    version: 2,
    tokenizer: { type: "train-only byte-pair", vocabularySize: tokenizer.vocabSize, eosTokenId: EOS_TOKEN_ID, merges: tokenizer.toJSON().merges.length, learnedVocabulary: true, trainingSplitOnly: true },
    dataset: { acceptedExamples: manifest.acceptedExamples, splits: manifest.splits, sourceSha256: manifest.sourceSha256, preparedSha256: manifest.preparedSha256 },
    trainingLimit,
    splits: summaries,
    interpretation: `The tokenizer is learned from training examples only. Truncation estimates use the same ${trainingLimit}-token cap passed to the trainer. Compression does not itself establish semantic quality; these diagnostics do not change training data.`,
  };
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
