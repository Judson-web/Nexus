import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel, type TransformerConfig } from "../model/transformer-model.js";
import { transformerLossAndGradients } from "../model/transformer-backprop.js";
import { transformerTapeLossAndGradients } from "../model/transformer-tape-backprop.js";

export interface TransformerGradientBenchmarkOptions {
  readonly repeats?: number;
  readonly warmups?: number;
  readonly examples?: number;
  readonly config?: Partial<TransformerConfig>;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!;
}

/** Deterministic, same-weights comparison; reports speed and numerical parity separately. */
export function benchmarkTransformerGradients(options: TransformerGradientBenchmarkOptions = {}) {
  const repeats = options.repeats ?? 7;
  const warmups = options.warmups ?? 2;
  const exampleCount = options.examples ?? 8;
  for (const [name, value, min, max] of [
    ["repeats", repeats, 3, 30],
    ["warmups", warmups, 0, 10],
    ["examples", exampleCount, 1, 32],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < min || value > max) {
      throw new RangeError(`${name} must be an integer in [${min}, ${max}].`);
    }
  }
  const config: TransformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...options.config };
  if (config.vocabSize < 8 || config.contextSize < 1 || config.embeddingSize < 2 ||
      !Number.isInteger(config.headCount) || config.headCount < 1 ||
      config.embeddingSize % config.headCount !== 0 || config.feedForwardSize < 1) {
    throw new RangeError("Invalid benchmark Transformer configuration.");
  }

  const model = new TransformerModel(config);
  const weights = Float64Array.from(model.toCheckpoint().weights);
  const examples = Array.from({ length: exampleCount }, (_, index) => ({
    context: Array.from({ length: Math.min(config.contextSize, 1 + (index % config.contextSize)) },
      (_, position) => (index * 7 + position * 3 + 1) % config.vocabSize),
    target: (index * 5 + 2) % config.vocabSize,
  }));
  const inputDigest = createHash("sha256")
    .update(JSON.stringify({ config, weights: Array.from(weights), examples }))
    .digest("hex");

  const evaluate = (fn: typeof transformerLossAndGradients) => examples.map((example) =>
    fn(config, weights, example.context, example.target));
  for (let index = 0; index < warmups; index += 1) {
    evaluate(transformerLossAndGradients);
    evaluate(transformerTapeLossAndGradients);
  }

  const scalarTimes: number[] = [];
  const tapeTimes: number[] = [];
  let scalarResults = evaluate(transformerLossAndGradients);
  let tapeResults = evaluate(transformerTapeLossAndGradients);
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    let started = performance.now();
    scalarResults = evaluate(transformerLossAndGradients);
    scalarTimes.push(performance.now() - started);
    started = performance.now();
    tapeResults = evaluate(transformerTapeLossAndGradients);
    tapeTimes.push(performance.now() - started);
  }

  let maxAbsoluteGradientDifference = 0;
  let maxRelativeGradientDifference = 0;
  let maxAbsoluteLossDifference = 0;
  let comparedGradients = 0;
  for (let example = 0; example < examples.length; example += 1) {
    const scalar = scalarResults[example]!;
    const tape = tapeResults[example]!;
    maxAbsoluteLossDifference = Math.max(maxAbsoluteLossDifference, Math.abs(scalar.loss - tape.loss));
    for (let index = 0; index < scalar.gradients.length; index += 1) {
      const expected = scalar.gradients[index]!;
      const actual = tape.gradients[index]!;
      const difference = Math.abs(actual - expected);
      maxAbsoluteGradientDifference = Math.max(maxAbsoluteGradientDifference, difference);
      maxRelativeGradientDifference = Math.max(maxRelativeGradientDifference,
        difference / Math.max(Math.abs(expected), Math.abs(actual), 1e-12));
      comparedGradients += 1;
    }
  }
  const scalarMedianMs = median(scalarTimes);
  const tapeMedianMs = median(tapeTimes);
  return {
    format: "nexus-transformer-gradient-benchmark",
    version: 1,
    input: { sha256: inputDigest, examples: exampleCount, comparedGradients, parameterCount: weights.length, config },
    runtime: { nodeVersion: process.version, platform: process.platform, architecture: process.arch },
    measurements: {
      repeats,
      warmups,
      scalarMedianMs,
      tapeMedianMs,
      speedup: scalarMedianMs / tapeMedianMs,
      scalarRunsMs: scalarTimes,
      tapeRunsMs: tapeTimes,
    },
    parity: {
      maxAbsoluteLossDifference,
      maxAbsoluteGradientDifference,
      maxRelativeGradientDifference,
      withinTolerance: maxAbsoluteLossDifference <= 1e-10 &&
        maxAbsoluteGradientDifference <= 2e-9 + 2e-8 * Math.max(1, maxAbsoluteGradientDifference),
    },
    interpretation: "This is a small deterministic CPU microbenchmark, not a training-throughput or model-quality claim. Use repeated runs on the same machine; promote the tape path only after parity and a repeatable end-to-end training win.",
  };
}
