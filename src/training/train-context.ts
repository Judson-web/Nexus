import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { ContextModel, DEFAULT_CONTEXT_CONFIG } from "../model/context-model.js";

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
  const initialLoss = model.train(tokens, options.learningRate, 1).loss;
  const metrics = model.train(tokens, options.learningRate, options.epochs);
  const bytes = Buffer.from(JSON.stringify(model.toCheckpoint()), "utf8");
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  await mkdir(dirname(options.outputPath), { recursive: true });
  const temporary = `${options.outputPath}.tmp`;
  await writeFile(temporary, bytes, { flag: "w" });
  await rename(temporary, options.outputPath);
  await writeFile(`${options.outputPath}.sha256`, `${sha256}  ${bytes.length}\n`, { flag: "w" });
  return { ...metrics, initialLoss, finalLoss: metrics.loss, checkpointPath: options.outputPath, checksum: sha256, checkpointBytes: bytes.length, modelFamily: "nexus-context-mlp", parameterCount: model.toCheckpoint().embedding.length + model.toCheckpoint().w1.length + model.toCheckpoint().b1.length + model.toCheckpoint().w2.length + model.toCheckpoint().b2.length };
}

export async function loadContextCheckpoint(path: string): Promise<ContextModel> {
  const bytes = await readFile(path);
  const expected = (await readFile(`${path}.sha256`, "utf8")).trim().split(/\s+/)[0];
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (!expected || expected !== actual) throw new Error("Context checkpoint checksum mismatch; refusing to load.");
  return ContextModel.fromCheckpoint(JSON.parse(bytes.toString("utf8")) as unknown);
}
