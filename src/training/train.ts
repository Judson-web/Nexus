import { readFile } from "node:fs/promises";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { BigramModel, type TrainingMetrics } from "../model/bigram-model.js";
import { saveCheckpoint } from "./checkpoint.js";

export interface TrainOptions {
  readonly inputPath: string;
  readonly outputPath: string;
  readonly epochs: number;
  readonly learningRate: number;
  readonly seed: number;
}

export interface TrainResult extends TrainingMetrics {
  readonly initialLoss: number;
  readonly finalLoss: number;
  readonly checkpointPath: string;
  readonly checksum: string;
  readonly checkpointBytes: number;
}

export async function trainFromFile(options: TrainOptions): Promise<TrainResult> {
  const text = (await readFile(options.inputPath, "utf8")).trim();
  if (text.length === 0) throw new Error("Training corpus is empty.");
  const tokenizer = new ByteTokenizer();
  const tokens = tokenizer.encode(text);
  const model = new BigramModel({ vocabSize: 257, seed: options.seed, initializationScale: 0.01 });
  const initialLoss = model.train(tokens, options.learningRate, 1).loss;
  const metrics = model.train(tokens, options.learningRate, options.epochs);
  const manifest = await saveCheckpoint(options.outputPath, model);
  return {
    ...metrics,
    initialLoss,
    finalLoss: metrics.loss,
    checkpointPath: options.outputPath,
    checksum: manifest.sha256,
    checkpointBytes: manifest.bytes,
  };
}
