import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { cpus, platform, arch } from "node:os";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel, type TransformerConfig } from "../model/transformer-model.js";

export interface TransformerProfileOptions {
  readonly epochs?: number;
  readonly learningRate?: number;
  readonly repeats?: number;
  readonly targetStride?: number;
  readonly optimizer?: "sgd" | "adam";
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
  const targetStride = options.targetStride ?? 1;
  if (!Number.isSafeInteger(targetStride) || targetStride < 1 || targetStride > 32) {
    throw new RangeError("targetStride must be an integer between 1 and 32.");
  }
  const optimizer = options.optimizer ?? "sgd";
  if (optimizer !== "sgd" && optimizer !== "adam") {
    throw new RangeError("optimizer must be sgd or adam.");
  }
  const repeats = options.repeats ?? 1;
  if (!Number.isSafeInteger(repeats) || repeats < 1 || repeats > 20) {
    throw new RangeError("repeats must be an integer between 1 and 20.");
  }

  const source = await readFile(path, "utf8");
  const tokens = new ByteTokenizer().encode(source.trim(), false);
  if (tokens.length < 2) throw new Error("Training corpus must contain at least two encoded tokens.");

  const config: TransformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...options.config };
  const initialModel = new TransformerModel(config);
  const parameterCount = initialModel.toCheckpoint().weights.length;
  const runMetrics: {
    run: number;
    totalDurationMs: number;
    meanEpochDurationMs: number;
    meanExamplesPerSecond: number;
    meanTargetsPerSecond: number;
    finalLoss: number;
    epochs: { epoch: number; loss: number; durationMs: number; examplesPerSecond: number; targetsProcessed: number; targetsPerSecond: number }[];
  }[] = [];

  for (let run = 1; run <= repeats; run += 1) {
    const model = new TransformerModel(config);
    const epochMetrics: { epoch: number; loss: number; durationMs: number; examplesPerSecond: number; targetsProcessed: number; targetsPerSecond: number }[] = [];
    for (let epoch = 1; epoch <= epochs; epoch += 1) {
      const started = performance.now();
      const metrics = model.train(tokens, learningRate, 1, 1, optimizer, undefined, targetStride);
      const durationMs = performance.now() - started;
      epochMetrics.push({
        epoch,
        loss: metrics.loss,
        durationMs,
        examplesPerSecond: metrics.examples / Math.max(durationMs / 1000, Number.EPSILON),
        targetsProcessed: metrics.examples,
        targetsPerSecond: metrics.examples / Math.max(durationMs / 1000, Number.EPSILON),
      });
    }
    const totalDurationMs = epochMetrics.reduce((sum, item) => sum + item.durationMs, 0);
    runMetrics.push({
      run,
      totalDurationMs,
      meanEpochDurationMs: totalDurationMs / epochs,
      meanExamplesPerSecond: epochMetrics.reduce((sum, item) => sum + item.examplesPerSecond, 0) / epochs,
      meanTargetsPerSecond: epochMetrics.reduce((sum, item) => sum + item.targetsPerSecond, 0) / epochs,
      finalLoss: epochMetrics[epochMetrics.length - 1]!.loss,
      epochs: epochMetrics,
    });
  }

  const median = (values: number[]): number => {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[middle - 1]! + sorted[middle]!) / 2
      : sorted[middle]!;
  };
  const totalDurationMs = median(runMetrics.map((run) => run.totalDurationMs));
  const meanEpochDurationMs = median(runMetrics.map((run) => run.meanEpochDurationMs));
  const meanExamplesPerSecond = median(runMetrics.map((run) => run.meanExamplesPerSecond));
  const meanTargetsPerSecond = median(runMetrics.map((run) => run.meanTargetsPerSecond));

  return {
    dataset: {
      sourceSha256: createHash("sha256").update(source, "utf8").digest("hex"),
      tokenCount: tokens.length,
    },
    runtime: {
      nodeVersion: process.version,
      platform: platform(),
      architecture: arch(),
      cpuModel: cpus()[0]?.model ?? "unknown",
      logicalCpuCount: cpus().length,
    },
    model: { family: "nexus-transformer", config, parameterCount, epochs, learningRate, optimizer, repeats, targetStride },
    profile: {
      totalDurationMs,
      meanEpochDurationMs,
      meanExamplesPerSecond,
      meanTargetsPerSecond,
      medianFinalLoss: median(runMetrics.map((run) => run.finalLoss)),
      runs: runMetrics,
    },
    warning: "Timing summaries are medians across fresh, identically seeded runs. Wall-clock results depend on hardware, runtime, and corpus size. Compare revisions on the same machine with the same corpus and settings; this profiler does not establish production throughput.",
  };
}
