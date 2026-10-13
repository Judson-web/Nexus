import { readFile } from "node:fs/promises";
import { validateDatasetManifest } from "../evaluation/manifest.js";
import { EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";

const DEFAULT_MAX_TOKENS_PER_EXAMPLE = 512;

function percentiles(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (fraction: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))] ?? 0;
  return { min: sorted[0] ?? 0, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: sorted.at(-1) ?? 0 };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    const value = args[i + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) throw new Error("Usage: npm run audit:tokenizer -- --manifest <path> [--max-tokens-per-example <count>]");
    const name = key.slice(2);
    if (!["manifest", "max-tokens-per-example"].includes(name) || options.has(name)) throw new Error(`Unknown or duplicate option: ${key}`);
    options.set(name, value);
    i += 1;
  }
  const manifestPath = options.get("manifest");
  if (!manifestPath) throw new Error("Missing required option --manifest");
  const maxTokensPerExample = Number(options.get("max-tokens-per-example") ?? DEFAULT_MAX_TOKENS_PER_EXAMPLE);
  if (!Number.isSafeInteger(maxTokensPerExample) || maxTokensPerExample < 16 || maxTokensPerExample > 4096) throw new RangeError("max-tokens-per-example must be an integer between 16 and 4096.");
  const raw = await readFile(manifestPath, "utf8");
  const manifest = validateDatasetManifest(JSON.parse(raw) as unknown);
  const tokenizer = BytePairTokenizer.train(manifest.examples.filter((example) => example.split === "train").map((example) => example.text), 255);
  const summaries = Object.fromEntries((["train", "validation", "test"] as const).map((split) => {
    const examples = manifest.examples.filter((example) => example.split === split);
    const tokenLengths = examples.map((example) => tokenizer.encode(example.text, false).length);
    const codePointLengths = examples.map((example) => Array.from(example.text).length);
    const targetLengthsIncludingEos = tokenLengths.map((length) => length + 1);
    const truncatedTokens = targetLengthsIncludingEos.map((length) => Math.max(0, length - maxTokensPerExample));
    const utf8Bytes = examples.reduce((sum, example) => sum + Buffer.byteLength(example.text, "utf8"), 0);
    const encodedTokens = examples.reduce((sum, example) => sum + tokenizer.encode(example.text, false).length, 0);
    const eosTargets = examples.length;
    return [split, {
      examples: examples.length,
      utf8Bytes,
      encodedTokens,
      tokenToByteRatio: utf8Bytes === 0 ? 0 : encodedTokens / utf8Bytes,
      unicodeCodePoints: codePointLengths.reduce((sum, n) => sum + n, 0),
      tokensPerCodePoint: codePointLengths.reduce((sum, n) => sum + n, 0) === 0 ? 0 : encodedTokens / codePointLengths.reduce((sum, n) => sum + n, 0),
      sequenceTokens: percentiles(tokenLengths),
      trainingMaxTokensPerExample: maxTokensPerExample,
      examplesExceedingTrainingLimit: truncatedTokens.filter((count) => count > 0).length,
      estimatedTruncatedTokens: truncatedTokens.reduce((sum, count) => sum + count, 0),
      nextTokenTargetsIncludingEos: encodedTokens + eosTargets,
    }];
  }));
  const report = {
    format: "nexus-tokenizer-audit",
    version: 1,
    tokenizer: { type: "train-only byte-pair", vocabularySize: tokenizer.vocabSize, eosTokenId: EOS_TOKEN_ID, merges: tokenizer.toJSON().merges.length, learnedVocabulary: true, trainingSplitOnly: true },
    dataset: { acceptedExamples: manifest.acceptedExamples, splits: manifest.splits, sourceSha256: manifest.sourceSha256, preparedSha256: manifest.preparedSha256 },
    splits: summaries,
    interpretation: "The tokenizer is learned from training examples only. Compare token-to-byte ratio, sequence-token percentiles, and estimated truncation at the configured training token cap. Compression does not itself establish semantic quality; these diagnostics do not change training data.",
  };
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
