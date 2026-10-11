import { evaluateDatasetManifest } from "../evaluation/manifest.js";

const args = process.argv.slice(2);
const options = new Map<string, string>();
for (let i = 0; i < args.length; i += 1) {
  const key = args[i]!;
  if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
  const value = args[i + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
  options.set(key.slice(2), value);
  i += 1;
}

try {
  const manifest = options.get("manifest") ?? "artifacts/dataset-manifest.json";
  const report = await evaluateDatasetManifest({
    manifestPath: manifest,
    epochs: Number(options.get("epochs") ?? "10"),
    seed: Number(options.get("seed") ?? "17"),
  });
  console.log(JSON.stringify({ manifest, ...report }, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
