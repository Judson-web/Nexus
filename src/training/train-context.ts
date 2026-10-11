import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "../model/context-model.js";
import { failTrainingRun, finishTrainingRun, registerModelVersion, startTrainingRun } from "./experiment-store.js";

export interface ContextTrainOptions {
  readonly inputPath: string;
  readonly outputPath: string;
  readonly epochs: number;
  readonly learningRate: number;
  readonly seed: number;
}

export async function trainContextFromFile(options: ContextTrainOptions) {
  const text = (await readFile(options.inputPath, "utf8")).trim();
  if (!text) throw new Error("Training corpus is empty.");
  const tokens = new ByteTokenizer().encode(text);
  const model = new ContextModel({ ...DEFAULT_CONTEXT_CONFIG, seed: options.seed });
  const runId = await startTrainingRun({
    modelFamily: "nexus-context-mlp",
    seed: options.seed,
    config: { ...model.config, epochs: options.epochs, learningRate: options.learningRate, inputPath: options.inputPath },
  });
  try {
    const initialLoss = model.train(tokens, options.learningRate, 1).loss;
    const metrics = model.train(tokens, options.learningRate, options.epochs);
    const checkpoint = model.toCheckpoint();
    const bytes = Buffer.from(JSON.stringify(checkpoint), "utf8");
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    await mkdir(dirname(options.outputPath), { recursive: true });
    const temporary = options.outputPath + ".tmp";
    await writeFile(temporary, bytes, { flag: "w" });
    await rename(temporary, options.outputPath);
    await writeFile(options.outputPath + ".sha256", sha256 + "  " + bytes.length + "\n", { flag: "w" });
    await finishTrainingRun(runId, { finalLoss: metrics.loss, initialLoss });
    await registerModelVersion({
      trainingRunId: runId,
      modelFamily: "nexus-context-mlp",
      version: sha256.slice(0, 16),
      checkpointUri: options.outputPath,
      checkpointSha256: sha256,
      config: { ...model.config, epochs: options.epochs, learningRate: options.learningRate },
      parameterCount: checkpoint.embedding.length + checkpoint.w1.length + checkpoint.b1.length + checkpoint.w2.length + checkpoint.b2.length,
    });
    return { ...metrics, initialLoss, finalLoss: metrics.loss, checkpointPath: options.outputPath, checksum: sha256, checkpointBytes: bytes.length, modelFamily: "nexus-context-mlp", parameterCount: checkpoint.embedding.length + checkpoint.w1.length + checkpoint.b1.length + checkpoint.w2.length + checkpoint.b2.length };
  } catch (error) {
    await failTrainingRun(runId, error);
    throw error;
  }
}

export async function loadContextCheckpoint(path: string): Promise<ContextModel> {
  const bytes = await readFile(path);
  const expected = (await readFile(`${path}.sha256`, "utf8")).trim().split(/\s+/)[0];
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (!expected || expected !== actual) throw new Error("Context checkpoint checksum mismatch; refusing to load.");
  return ContextModel.fromCheckpoint(JSON.parse(bytes.toString("utf8")) as unknown);
}
