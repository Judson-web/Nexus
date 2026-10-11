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
  if (manifest.formatVersion !== 1 || !Array.isArray(manifest.examples) ||
      !manifest.splits || typeof manifest.sourceSha256 !== "string" ||
      typeof manifest.preparedSha256 !== "string") {
    throw new Error("Unsupported or malformed dataset manifest.");
  }
  const ids = new Set<string>();
  for (const example of manifest.examples) {
    if (!example || typeof example.text !== "string" || !/^[a-f0-9]{64}$/.test(example.sha256) ||
        !["train", "validation", "test"].includes(example.split)) {
      throw new Error("Dataset manifest contains a malformed example.");
    }
    if (ids.has(example.sha256)) throw new Error("Dataset manifest contains duplicate examples.");
    ids.add(example.sha256);
  }
  for (const split of ["train", "validation", "test"] as const) {
    if (!Number.isInteger(manifest.splits[split]) ||
        manifest.splits[split] !== manifest.examples.filter((example) => example.split === split).length) {
      throw new Error(`Dataset manifest ${split} count does not match its examples.`);
    }
  }
  if (manifest.examples.length === 0 || manifest.splits.train < 1 ||
      manifest.splits.validation < 1 || manifest.splits.test < 1) {
    throw new Error("Manifest must contain at least one example in every split.");
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
