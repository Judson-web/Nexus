import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { BigramModel } from "../model/bigram-model.js";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "../model/context-model.js";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";

export interface HeldOutMetrics {
  readonly modelFamily: string;
  readonly trainTokens: number;
  readonly validationTokens: number;
  readonly testTokens: number;
  readonly validationNegativeLogLikelihood: number;
  readonly validationPerplexity: number;
  readonly testNegativeLogLikelihood: number;
  readonly testPerplexity: number;
}

function measure(
  probabilities: (context: number[]) => Float64Array,
  tokens: readonly number[],
  contextSize: number,
  start: number,
  end: number,
): { readonly tokenCount: number; readonly negativeLogLikelihood: number; readonly perplexity: number } {
  if (start < contextSize || end > tokens.length || start >= end) throw new Error("Held-out split is too small to evaluate.");
  let total = 0;
  for (let i = start; i < end; i += 1) {
    const context = tokens.slice(i - contextSize, i);
    const distribution = probabilities(context);
    const probability = distribution[tokens[i]!]!;
    total -= Math.log(Math.max(probability, Number.MIN_VALUE));
  }
  const nll = total / (end - start);
  return { tokenCount: end - start, negativeLogLikelihood: nll, perplexity: Math.exp(Math.min(nll, 700)) };
}

export async function evaluateHeldOutFile(path = "data/corpus.txt", epochs = 20) {
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 10000) throw new RangeError("epochs must be between 1 and 10000.");
  const source = await readFile(path, "utf8");
  if (source.trim().length < 40) throw new Error("Corpus needs at least 40 characters for a meaningful deterministic split.");
  const tokenizer = new ByteTokenizer();
  const tokens = tokenizer.encode(source.trim(), false);
  const trainEnd = Math.floor(tokens.length * 0.8);
  const validationEnd = Math.floor(tokens.length * 0.9);
  const trainTokens = tokens.slice(0, trainEnd);
  if (trainTokens.length < 10 || validationEnd - trainEnd < 5 || tokens.length - validationEnd < 5) {
    throw new Error("Corpus split is too small.");
  }

  const bigram = new BigramModel({ vocabSize: 257, seed: 17, initializationScale: 0.01 });
  bigram.train([...trainTokens, 256], 0.08, epochs);
  const context = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: 41 });
  context.train([...trainTokens, 256], 0.03, epochs);

  const evaluate = (modelFamily: string, probabilities: (ctx: number[]) => Float64Array, contextSize: number): HeldOutMetrics => {
    const validation = measure(probabilities, tokens, contextSize, trainEnd, validationEnd);
    const test = measure(probabilities, tokens, contextSize, validationEnd, tokens.length);
    return {
      modelFamily,
      trainTokens: trainTokens.length,
      validationTokens: validation.tokenCount,
      testTokens: test.tokenCount,
      validationNegativeLogLikelihood: validation.negativeLogLikelihood,
      validationPerplexity: validation.perplexity,
      testNegativeLogLikelihood: test.negativeLogLikelihood,
      testPerplexity: test.perplexity,
    };
  };

  return {
    dataset: {
      sourceSha256: createHash("sha256").update(source, "utf8").digest("hex"),
      evaluatedContentSha256: createHash("sha256").update(source.trim(), "utf8").digest("hex"),
    },
    split: { method: "ordered 80/10/10 byte split", totalTokens: tokens.length, trainTokens: trainTokens.length, validationTokens: validationEnd - trainEnd, testTokens: tokens.length - validationEnd },
    evaluations: [
      evaluate("nexus-bigram", (ctx) => bigram.probabilities(ctx[ctx.length - 1]!), 1),
      evaluate("nexus-context-mlp", (ctx) => context.probabilities(ctx), context.config.contextSize),
    ],
    warning: "This small corpus benchmark is diagnostic only; it is not evidence of general language ability. Validation metrics can guide experiments; do not tune repeatedly against the test split. Use a large, licensed, deduplicated corpus and a fixed held-out test set before comparing model quality.",
  };
}
