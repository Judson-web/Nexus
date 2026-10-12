import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel, type TransformerConfig } from "../model/transformer-model.js";

export interface TransformerHeldOutMetrics {
  readonly modelFamily: "nexus-transformer";
  readonly trainTokens: number;
  readonly validationTokens: number;
  readonly testTokens: number;
  readonly validationNegativeLogLikelihood: number;
  readonly validationPerplexity: number;
  readonly testNegativeLogLikelihood: number;
  readonly testPerplexity: number;
}

export interface TransformerHeldOutOptions {
  readonly epochs?: number;
  readonly learningRate?: number;
  readonly config?: Partial<TransformerConfig>;
}

function scoreRange(
  model: TransformerModel,
  tokens: readonly number[],
  contextSize: number,
  start: number,
  end: number,
): { tokenCount: number; negativeLogLikelihood: number; perplexity: number } {
  if (start < 1 || end > tokens.length || start >= end) {
    throw new Error("Held-out split is too small to evaluate.");
  }
  let totalLoss = 0;
  for (let index = start; index < end; index += 1) {
    const context = tokens.slice(Math.max(0, index - contextSize), index);
    const probabilities = model.probabilities(context);
    totalLoss -= Math.log(Math.max(Number.MIN_VALUE, probabilities[tokens[index]!]!));
  }
  const negativeLogLikelihood = totalLoss / (end - start);
  return {
    tokenCount: end - start,
    negativeLogLikelihood,
    perplexity: Math.exp(Math.min(negativeLogLikelihood, 700)),
  };
}

/** Train only on the first 80% and score untouched validation/test suffixes. */
export async function evaluateTransformerHeldOutFile(
  path = "data/corpus.txt",
  options: TransformerHeldOutOptions = {},
) {
  const epochs = options.epochs ?? 1;
  const learningRate = options.learningRate ?? 0.0001;
  if (!Number.isSafeInteger(epochs) || epochs < 1 || epochs > 1000) {
    throw new RangeError("epochs must be an integer between 1 and 1000.");
  }
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.02) {
    throw new RangeError("learningRate must be in (0, 0.02].");
  }

  const source = await readFile(path, "utf8");
  const normalized = source.trim();
  if (normalized.length < 40) {
    throw new Error("Corpus needs at least 40 characters for a meaningful deterministic split.");
  }
  const tokens = new ByteTokenizer().encode(normalized, false);
  const trainEnd = Math.floor(tokens.length * 0.8);
  const validationEnd = Math.floor(tokens.length * 0.9);
  if (trainEnd < 2 || validationEnd - trainEnd < 2 || tokens.length - validationEnd < 2) {
    throw new Error("Corpus split is too small.");
  }

  const config: TransformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...options.config };
  const model = new TransformerModel(config);
  const trainingMetrics = model.train(tokens.slice(0, trainEnd), learningRate, epochs);
  const validation = scoreRange(model, tokens, config.contextSize, trainEnd, validationEnd);
  const test = scoreRange(model, tokens, config.contextSize, validationEnd, tokens.length);

  return {
    dataset: {
      sourceSha256: createHash("sha256").update(source, "utf8").digest("hex"),
      evaluatedContentSha256: createHash("sha256").update(normalized, "utf8").digest("hex"),
    },
    split: {
      method: "ordered 80/10/10 byte split",
      totalTokens: tokens.length,
      trainTokens: trainEnd,
      validationTokens: validationEnd - trainEnd,
      testTokens: tokens.length - validationEnd,
    },
    model: { family: "nexus-transformer", config, epochs, learningRate },
    training: trainingMetrics,
    evaluation: {
      modelFamily: "nexus-transformer" as const,
      trainTokens: trainEnd,
      validationTokens: validation.tokenCount,
      testTokens: test.tokenCount,
      validationNegativeLogLikelihood: validation.negativeLogLikelihood,
      validationPerplexity: validation.perplexity,
      testNegativeLogLikelihood: test.negativeLogLikelihood,
      testPerplexity: test.perplexity,
    } satisfies TransformerHeldOutMetrics,
    warning: "This small-corpus benchmark is diagnostic only, not evidence of general language ability. Keep the test split untouched during model tuning; use a large, licensed, deduplicated corpus and a fixed test set for meaningful comparisons.",
  };
}
