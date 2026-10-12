import { readFile, writeFile } from "node:fs/promises";
import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "../model/transformer-model.js";

function optionsFromArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (!arg.startsWith("--")) throw new Error(`Unexpected argument: ${arg}`);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${arg.slice(2)}`);
    options.set(arg.slice(2), value);
    i++;
  }
  return options;
}

function numeric(options: Map<string, string>, key: string, fallback: number): number {
  const value = options.get(key);
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`--${key} must be numeric.`);
  return parsed;
}

function required(options: Map<string, string>, key: string): string {
  const value = options.get(key);
  if (!value) throw new Error(`Required option --${key} is missing.`);
  return value;
}

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);
  const options = optionsFromArgs(args);
  const tokenizer = new ByteTokenizer();

  if (command === "train") {
    const inputPath = options.get("input") ?? "data/corpus.txt";
    const outputPath = options.get("output") ?? "artifacts/transformer-v1.json";
    const corpus = await readFile(inputPath, "utf8");
    const tokens = tokenizer.encode(corpus);
    const config = {
      ...DEFAULT_TRANSFORMER_CONFIG,
      contextSize: numeric(options, "context-size", DEFAULT_TRANSFORMER_CONFIG.contextSize),
      embeddingSize: numeric(options, "embedding-size", DEFAULT_TRANSFORMER_CONFIG.embeddingSize),
      headCount: numeric(options, "heads", DEFAULT_TRANSFORMER_CONFIG.headCount),
      feedForwardSize: numeric(options, "feed-forward-size", DEFAULT_TRANSFORMER_CONFIG.feedForwardSize),
      seed: numeric(options, "seed", DEFAULT_TRANSFORMER_CONFIG.seed),
    };
    const model = new TransformerModel(config);
    const metrics = model.train(tokens, numeric(options, "learning-rate", 0.0001), numeric(options, "epochs", 1));
    await writeFile(outputPath, JSON.stringify(model.toCheckpoint(), null, 2), "utf8");
    console.log(JSON.stringify({ inputPath, outputPath, tokenCount: tokens.length, ...metrics }, null, 2));
    return;
  }

  if (command === "generate") {
    const checkpoint = JSON.parse(await readFile(required(options, "checkpoint"), "utf8")) as Parameters<typeof TransformerModel.fromCheckpoint>[0];
    const model = TransformerModel.fromCheckpoint(checkpoint);
    const generationTokenizer = checkpoint.tokenizer ? new BytePairTokenizer(checkpoint.tokenizer) : tokenizer;
    if (generationTokenizer instanceof BytePairTokenizer && generationTokenizer.vocabSize !== model.config.vocabSize) {
      throw new Error("Checkpoint tokenizer vocabulary does not match model configuration.");
    }
    const prompt = required(options, "prompt");
    const generated = model.generate(generationTokenizer.encode(prompt, false), numeric(options, "tokens", 80));
    console.log(generationTokenizer.decode(generated));
    return;
  }

  throw new Error("Usage: npm run train:transformer -- [--input path --epochs n --output path] | npm run generate:transformer -- --checkpoint path --prompt text [--tokens n]");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
