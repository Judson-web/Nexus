import { profileTransformerTraining } from "../evaluation/transformer-profile.js";

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

const allowed = new Set(["input", "epochs", "repeats", "learning-rate", "context-size", "embedding-size", "heads", "feed-forward-size", "seed", "target-stride"]);
for (const key of options.keys()) {
  if (!allowed.has(key)) throw new Error(`Unknown option: --${key}`);
}
function numberOption(name: string, fallback: number): number {
  const raw = options.get(name);
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) throw new Error(`--${name} must be numeric.`);
  return parsed;
}

try {
  const report = await profileTransformerTraining(options.get("input") ?? "data/corpus.txt", {
    epochs: numberOption("epochs", 2),
    repeats: numberOption("repeats", 1),
    targetStride: numberOption("target-stride", 1),
    learningRate: numberOption("learning-rate", 0.0001),
    config: {
      contextSize: numberOption("context-size", 3),
      embeddingSize: numberOption("embedding-size", 4),
      headCount: numberOption("heads", 2),
      feedForwardSize: numberOption("feed-forward-size", 8),
      seed: numberOption("seed", 101),
    },
  });
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
