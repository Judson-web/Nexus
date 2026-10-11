import { ByteTokenizer } from "../tokenizer/byte-tokenizer.js";
import { loadContextCheckpoint, trainContextFromFile } from "./train-context.js";

function parse(args: string[]): Map<string, string> {
  const result = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    result.set(key.slice(2), value);
    i += 1;
  }
  return result;
}

function numberOption(options: Map<string, string>, key: string, fallback: number): number {
  const raw = options.get(key);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} must be numeric.`);
  return value;
}

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);
  const options = parse(args);
  if (command === "train") {
    console.log(JSON.stringify(await trainContextFromFile({
      inputPath: options.get("input") ?? "data/corpus.txt",
      outputPath: options.get("output") ?? "artifacts/context-mlp-v1.json",
      epochs: numberOption(options, "epochs", 20),
      learningRate: numberOption(options, "learning-rate", 0.03),
      seed: numberOption(options, "seed", 41),
    }), null, 2));
    return;
  }
  if (command === "generate") {
    const checkpoint = options.get("checkpoint");
    const prompt = options.get("prompt");
    if (!checkpoint || prompt === undefined) throw new Error("Usage: train-context generate --checkpoint path --prompt text [--tokens n]");
    const model = await loadContextCheckpoint(checkpoint);
    const tokenizer = new ByteTokenizer();
    const output = model.generate(tokenizer.encode(prompt, false), numberOption(options, "tokens", 120));
    console.log(tokenizer.decode(output));
    return;
  }
  throw new Error("Usage: npm run train:context -- [--input path --epochs n --learning-rate n --output path] | npm run generate:context -- --checkpoint path --prompt text [--tokens n]");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
