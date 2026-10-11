import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { prepareDataset } from "../data/dataset.js";

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
  const input = options.get("input") ?? "data/corpus.txt";
  const output = options.get("output") ?? "artifacts/dataset-manifest.json";
  const seed = Number(options.get("seed") ?? "17");
  const source = await readFile(input, "utf8");
  const manifest = prepareDataset(source, { seed });
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(manifest, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({
    input,
    output,
    sourceSha256: manifest.sourceSha256,
    preparedSha256: manifest.preparedSha256,
    acceptedExamples: manifest.acceptedExamples,
    duplicatesRemoved: manifest.duplicatesRemoved,
    emptyLinesRemoved: manifest.emptyLinesRemoved,
    splits: manifest.splits,
    seed: manifest.splitSeed,
  }, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
