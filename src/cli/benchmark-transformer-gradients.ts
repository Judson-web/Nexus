import { benchmarkTransformerGradients } from "../evaluation/transformer-gradient-benchmark.js";

const args = process.argv.slice(2);
const options = new Map<string, string>();
for (let index = 0; index < args.length; index += 1) {
  const key = args[index]!;
  if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
  if (options.has(key.slice(2))) throw new Error(`Duplicate option: ${key}`);
  options.set(key.slice(2), value);
  index += 1;
}
const allowed = new Set(["repeats", "warmups", "examples", "vocab-size", "context-size", "embedding-size", "heads", "feed-forward-size", "seed"]);
for (const key of options.keys()) if (!allowed.has(key)) throw new Error(`Unknown option: --${key}`);
function numberOption(name: string, fallback: number): number {
  const raw = options.get(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be numeric.`);
  return value;
}
try {
  console.log(JSON.stringify(benchmarkTransformerGradients({
    repeats: numberOption("repeats", 7),
    warmups: numberOption("warmups", 2),
    examples: numberOption("examples", 8),
    config: {
      vocabSize: numberOption("vocab-size", 32),
      contextSize: numberOption("context-size", 4),
      embeddingSize: numberOption("embedding-size", 4),
      headCount: numberOption("heads", 2),
      feedForwardSize: numberOption("feed-forward-size", 8),
      seed: numberOption("seed", 101),
    },
  }), null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
