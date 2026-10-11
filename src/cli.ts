import { ByteTokenizer } from "./tokenizer/byte-tokenizer.js";
import { trainFromFile } from "./training/train.js";
import { loadCheckpoint } from "./training/checkpoint.js";

function optionsFromArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i]!;
    if (!arg.startsWith("--")) throw new Error(`Unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    options.set(key, value);
    i += 1;
  }
  return options;
}

function required(options: Map<string, string>, key: string): string {
  const value = options.get(key);
  if (!value) throw new Error(`Required option --${key} is missing.`);
  return value;
}

function numeric(options: Map<string, string>, key: string, fallback: number): number {
  const value = options.get(key);
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`--${key} must be numeric.`);
  return parsed;
}

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);
  const options = optionsFromArgs(args);
  if (command === "train") {
    const result = await trainFromFile({
      inputPath: options.get("input") ?? "data/corpus.txt",
      outputPath: options.get("output") ?? "artifacts/bigram-v1.json",
      epochs: numeric(options, "epochs", 20),
      learningRate: numeric(options, "learning-rate", 0.08),
      seed: numeric(options, "seed", 17),
    });
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  if (command === "generate") {
    const model = await loadCheckpoint(required(options, "checkpoint"));
    const tokenizer = new ByteTokenizer();
    const prompt = required(options, "prompt");
    const maxTokens = numeric(options, "tokens", 120);
    const generated = model.generate(tokenizer.encode(prompt, false), maxTokens);
    console.log(tokenizer.decode(generated));
    return;
  }
  throw new Error("Usage: npm run train -- [--input path --epochs n --learning-rate n --output path] | npm run generate -- --checkpoint path --prompt text [--tokens n]");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
