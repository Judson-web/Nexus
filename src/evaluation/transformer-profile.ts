import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel, type TransformerConfig } from "../model/transformer-model.js";

export interface TransformerProfileOptions {
  readonly epochs?: number;
  readonly learningRate?: number;
  readonly config?: Partial<TransformerConfig>;
}

export async function profileTransformerTraining(
  path = "data/corpus.txt",
  options: TransformerProfileOptions = {},
) {
  const epochs = options.epochs ?? 2;
  const learningRate = options.learningRate ?? 0.0001;
  if (!Number.isSafeInteger(epochs) || epochs < 1 || epochs > 100) {
    throw new RangeError("epochs must be an integer between 1 and 100.");
  }
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.02) {
    throw new RangeError("learningRate must be in (0, 0.02].");
  }

  const source = await readFile(path, "utf8");
  const tokens = new ByteTokenizer().encode(source.trim(), false);
  if (tokens.length < 2) throw new Error("Training corpus must contain at least two encoded tokens.");

  const config: TransformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...options.config };
  const model = new TransformerModel(config);
  const parameterCount = model.toCheckpoint().weights.length;
  const epochMetrics: { epoch: number; loss: number; durationMs: number; examplesPerSecond: number }[] = [];

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const started = performance.now();
    const metrics = model.train(tokens, learningRate, 1);
    const durationMs = performance.now() - started;
    epochMetrics.push({
      epoch,
      loss: metrics.loss,
      durationMs,
      examplesPerSecond: metrics.examples / Math.max(durationMs / 1000, Number.EPSILON),
    });
  }

  const totalDurationMs = epochMetrics.reduce((sum, item) => sum + item.durationMs, 0);
  return {
    dataset: {
      sourceSha256: createHash("sha256").update(source, "utf8").digest("hex"),
      tokenCount: tokens.length,
    },
    model: { family: "nexus-transformer", config, parameterCount, epochs, learningRate },
    profile: {
      totalDurationMs,
      meanEpochDurationMs: totalDurationMs / epochs,
      meanExamplesPerSecond: epochMetrics.reduce((sum, item) => sum + item.examplesPerSecond, 0) / epochs,
      epochs: epochMetrics,
    },
    warning: "Wall-clock results depend on hardware, runtime, and corpus size. Compare repeated runs on the same machine; this profiler measures the current CPU reference implementation and does not establish production throughput.",
  };
}
