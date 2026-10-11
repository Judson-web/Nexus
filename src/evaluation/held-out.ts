import { readFile } from "node:fs/promises";
import { BigramModel } from "../model/bigram-model.js";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "../model/context-model.js";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";

export interface HeldOutMetrics {
  readonly modelFamily: string;
  readonly trainTokens: number;
  readonly testTokens: number;
  readonly negativeLogLikelihood: number;
  readonly perplexity: number;
}

function meanNegativeLogLikelihood(
  probabilities: (context: number[]) => Float64Array,
  tokens: readonly number[],
  contextSize: number,
  start: number,
  end: number,
): HeldOutMetrics {
  if (start < contextSize || end > tokens.length || start >= end) throw new Error("Held-out split is too small to evaluate.");
  let total = 0;
  let count = 0;
  for (let i = start; i < end; i += 1) {
    const context = tokens.slice(i - contextSize, i);
    const distribution = probabilities(context);
    const probability = distribution[tokens[i]!]!;
    total -= Math.log(Math.max(probability, Number.MIN_VALUE));
    count += 1;
  }
  const nll = total / count;
  return { modelFamily: "", trainTokens: 0, testTokens: count, negativeLogLikelihood: nll, perplexity: Math.exp(Math.min(nll, 700)) };
}

export async function evaluateHeldOutFile(path = "data/corpus.txt", epochs = 20) {
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 10000) throw new RangeError("epochs must be between 1 and 10000.");
  const source = (await readFile(path, "utf8")).trim();
  if (source.length < 40) throw new Error("Corpus needs at least 40 characters for a meaningful deterministic split.");
  const tokenizer = new ByteTokenizer();
  const tokens = tokenizer.encode(source, false);
  const trainEnd = Math.floor(tokens.length * 0.8);
  const testStart = Math.floor(tokens.length * 0.9);
  const trainTokens = tokens.slice(0, trainEnd);
  const testTokens = tokens.slice(testStart);
  if (trainTokens.length < 10 || testTokens.length < 5) throw new Error("Corpus split is too small.");
  const bigram = new BigramModel({ vocabSize: 257, seed: 17, initializationScale: 0.01 });
  bigram.train([...trainTokens, 256], 0.08, epochs);
  const context = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: 41 });
  context.train([...trainTokens, 256], 0.03, epochs);
  const bigramRaw = meanNegativeLogLikelihood((ctx) => bigram.probabilities(ctx[ctx.length - 1]!), tokens, 1, testStart, tokens.length);
  const contextRaw = meanNegativeLogLikelihood((ctx) => context.probabilities(ctx), tokens, context.config.contextSize, testStart, tokens.length);
  return {
    split: { method: "ordered 80/10/10 byte split", totalTokens: tokens.length, trainTokens: trainTokens.length, validationTokens: testStart - trainEnd, testTokens: testTokens.length },
    evaluations: [
      { ...bigramRaw, modelFamily: "nexus-bigram", trainTokens: trainTokens.length },
      { ...contextRaw, modelFamily: "nexus-context-mlp", trainTokens: trainTokens.length },
    ],
    warning: "This small corpus benchmark is diagnostic only; it is not evidence of general language ability. Use a large, deduplicated corpus and a fixed held-out test set before comparing model quality.",
  };
}
