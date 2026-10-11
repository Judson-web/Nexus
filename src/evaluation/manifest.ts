import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import type { DatasetManifest, PreparedExample, DatasetSplit } from "../data/dataset.js";
import { BigramModel } from "../model/bigram-model.js";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "../model/context-model.js";
import { ByteTokenizer, EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";

export interface ManifestEvaluationOptions {
  readonly manifestPath: string;
  readonly epochs?: number;
  readonly seed?: number;
}

interface Score {
  readonly examples: number;
  readonly tokens: number;
  readonly negativeLogLikelihood: number;
  readonly perplexity: number;
}

function validateManifest(value: unknown): DatasetManifest {
  if (!value || typeof value !== "object") throw new Error("Dataset manifest must be a JSON object.");
  const manifest = value as Partial<DatasetManifest>;
  const isSha256 = (digest: unknown): digest is string =>
    typeof digest === "string" && /^[a-f0-9]{64}$/.test(digest);
  if (manifest.formatVersion !== 1 || !Array.isArray(manifest.examples) ||
      !manifest.splits || typeof manifest.splits !== "object" ||
      !isSha256(manifest.sourceSha256) || !isSha256(manifest.preparedSha256) ||
      !Number.isSafeInteger(manifest.splitSeed) || manifest.splitSeed! < 0 ||
      !manifest.ratios || ![manifest.ratios.train, manifest.ratios.validation, manifest.ratios.test]
        .every((ratio) => Number.isFinite(ratio) && ratio! > 0) ||
      Math.abs(manifest.ratios.train + manifest.ratios.validation + manifest.ratios.test - 1) > 1e-9) {
    throw new Error("Unsupported or malformed dataset manifest.");
  }
  const ids = new Set<string>();
  const splitCounts: Record<DatasetSplit, number> = { train: 0, validation: 0, test: 0 };
  for (const example of manifest.examples) {
    if (!example || typeof example.text !== "string" || example.text.length === 0 ||
        !isSha256(example.sha256) || typeof example.id !== "string" ||
        !["train", "validation", "test"].includes(example.split)) {
      throw new Error("Dataset manifest contains a malformed example.");
    }
    const actualHash = createHash("sha256").update(example.text, "utf8").digest("hex");
    if (actualHash !== example.sha256) throw new Error("Dataset manifest example text hash does not match.");
    if (example.id !== example.sha256.slice(0, 24)) throw new Error("Dataset manifest example ID does not match its hash.");
    if (ids.has(example.sha256)) throw new Error("Dataset manifest contains duplicate examples.");
    ids.add(example.sha256);
    const split = example.split as DatasetSplit;
    splitCounts[split] += 1;
  }
  for (const split of ["train", "validation", "test"] as const) {
    if (!Number.isInteger(manifest.splits[split]) || manifest.splits[split] !== splitCounts[split]) {
      throw new Error(`Dataset manifest ${split} count does not match its examples.`);
    }
  }
  if (manifest.examples.length === 0 || splitCounts.train < 1 ||
      splitCounts.validation < 1 || splitCounts.test < 1) {
    throw new Error("Manifest must contain at least one example in every split.");
  }
  if (manifest.acceptedExamples !== manifest.examples.length) {
    throw new Error("Dataset manifest accepted-example count does not match its examples.");
  }
  const preparedPayload = [...manifest.examples]
    .sort((a, b) => a.sha256.localeCompare(b.sha256))
    .map((example) => example.sha256 + ":" + example.split)
    .join("\n");
  const actualPreparedHash = createHash("sha256").update(preparedPayload, "utf8").digest("hex");
  if (actualPreparedHash !== manifest.preparedSha256) {
    throw new Error("Dataset manifest prepared-content hash does not match its examples.");
  }
  return manifest as DatasetManifest;
}

function scoreExamples(
  examples: readonly PreparedExample[],
  probabilities: (context: number[]) => Float64Array,
  contextSize: number,
  tokenizer: ByteTokenizer,
): Score {
  let total = 0;
  let tokens = 0;
  for (const example of examples) {
    const encoded = [...tokenizer.encode(example.text, false), EOS_TOKEN_ID];
    for (let i = 0; i < encoded.length; i += 1) {
      const context = encoded.slice(Math.max(0, i - contextSize), i);
      const distribution = probabilities(context);
      total -= Math.log(Math.max(distribution[encoded[i]!]!, Number.MIN_VALUE));
      tokens += 1;
    }
  }
  if (!tokens) throw new Error("Cannot score an empty dataset split.");
  const nll = total / tokens;
  return { examples: examples.length, tokens, negativeLogLikelihood: nll, perplexity: Math.exp(Math.min(nll, 700)) };
}

export async function evaluateDatasetManifest(options: ManifestEvaluationOptions) {
  const epochs = options.epochs ?? 10;
  const seed = options.seed ?? 17;
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 10000) throw new RangeError("epochs must be between 1 and 10000.");
  if (!Number.isSafeInteger(seed)) throw new RangeError("seed must be a safe integer.");
  const raw = await readFile(options.manifestPath, "utf8");
  const manifest = validateManifest(JSON.parse(raw) as unknown);
  const train = manifest.examples.filter((example) => example.split === "train");
  const validation = manifest.examples.filter((example) => example.split === "validation");
  const test = manifest.examples.filter((example) => example.split === "test");
  const tokenizer = new ByteTokenizer();

  const bigram = new BigramModel({ vocabSize: 257, seed, initializationScale: 0.01 });
  const context = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed });
  for (const example of train) {
    const tokens = [...tokenizer.encode(example.text, false), EOS_TOKEN_ID];
    bigram.train(tokens, 0.08, epochs);
    context.train(tokens, 0.03, epochs);
  }

  const evaluate = (modelFamily: string, probabilities: (context: number[]) => Float64Array, contextSize: number) => ({
    modelFamily,
    validation: scoreExamples(validation, probabilities, contextSize, tokenizer),
    test: scoreExamples(test, probabilities, contextSize, tokenizer),
  });
  return {
    dataset: {
      manifestSha256: createHash("sha256").update(raw, "utf8").digest("hex"),
      sourceSha256: manifest.sourceSha256,
      preparedSha256: manifest.preparedSha256,
      splitSeed: manifest.splitSeed,
      counts: manifest.splits,
    },
    training: { epochsPerExample: epochs, seed, trainedExamples: train.length, leakagePolicy: "Each line is an independent sequence; validation/test examples are never used for gradient updates." },
    evaluations: [
      evaluate("nexus-bigram", (ctx) => bigram.probabilities(ctx.length ? ctx[ctx.length - 1]! : 0), 1),
      evaluate("nexus-context-mlp", (ctx) => context.probabilities(ctx), context.config.contextSize),
    ],
    warning: "This evaluates the exact content-hash splits in the manifest, separately from the legacy ordered byte-split evaluator. Small corpora and tiny models do not establish general language ability; do not tune against the test split.",
  };
}

export function splitExamples(examples: readonly PreparedExample[], split: DatasetSplit): PreparedExample[] {
  return examples.filter((example) => example.split === split);
}
