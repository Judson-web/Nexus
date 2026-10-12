import { evaluateDatasetManifest } from "../evaluation/manifest.js";

const args = process.argv.slice(2);
const options = new Map<string, string>();
let includeTransformer = false;
for (let i = 0; i < args.length; i += 1) {
  const key = args[i]!;
  if (key === "--include-transformer") {
    if (includeTransformer) throw new Error("Duplicate option: --include-transformer");
    includeTransformer = true;
    continue;
  }
  if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
  const value = args[i + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
  const name = key.slice(2);
  if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
  if (!["manifest", "epochs", "seed"].includes(name)) throw new Error(`Unknown option: ${key}`);
  options.set(name, value);
  i += 1;
}

try {
  const manifest = options.get("manifest") ?? "artifacts/dataset-manifest.json";
  const report = await evaluateDatasetManifest({
    manifestPath: manifest,
    epochs: Number(options.get("epochs") ?? "10"),
    seed: Number(options.get("seed") ?? "17"),
    includeTransformer,
  });
  console.log(JSON.stringify({ manifest, ...report }, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
