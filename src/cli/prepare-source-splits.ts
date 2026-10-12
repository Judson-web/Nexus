import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { DatasetSplit, PreparedExample, DatasetManifest } from "../data/dataset.js";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    const name = key.slice(2);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    options.set(name, value);
    i += 1;
  }
  return options;
}

/**
 * Build a manifest from a dataset that already defines train/validation/test.
 * Never reshuffles official splits: this avoids accidental leakage from
 * recombining benchmark splits and then hashing lines into new partitions.
 */
async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const required = ["train", "validation", "test", "source-name", "source-url", "license", "attribution"];
  for (const key of required) {
    if (!options.get(key)?.trim()) throw new Error(`Missing required option --${key}`);
  }
  const inputPaths = {
    train: options.get("train")!,
    validation: options.get("validation")!,
    test: options.get("test")!,
  };
  const raw = {
    train: await readFile(inputPaths.train, "utf8"),
    validation: await readFile(inputPaths.validation, "utf8"),
    test: await readFile(inputPaths.test, "utf8"),
  };
  const unique = new Map<string, { text: string; split: DatasetSplit }>();
  const examples: PreparedExample[] = [];
  const counts: Record<DatasetSplit, number> = { train: 0, validation: 0, test: 0 };
  let sourceLines = 0;
  let emptyLinesRemoved = 0;
  let duplicatesRemoved = 0;

  for (const split of ["train", "validation", "test"] as const) {
    const lines = raw[split].replace(/^\uFEFF/, "").split(/\r?\n/);
    sourceLines += lines.length;
    for (const line of lines) {
      const text = line.normalize("NFC").replace(/\s+/gu, " ").trim();
      if (!text) {
        emptyLinesRemoved += 1;
        continue;
      }
      const digest = sha256(text);
      const existing = unique.get(digest);
      if (existing) {
        if (existing.split !== split) {
          throw new Error(`Exact duplicate crosses ${existing.split}/${split} split boundary (sha256 ${digest}). Refusing possible evaluation leakage.`);
        }
        duplicatesRemoved += 1;
        continue;
      }
      unique.set(digest, { text, split });
      examples.push({ id: digest.slice(0, 24), text, sha256: digest, split });
      counts[split] += 1;
    }
  }

  if (Object.values(counts).some((count) => count === 0)) {
    throw new Error("Train, validation, and test files must each contain at least one unique non-empty line.");
  }
  examples.sort((a, b) => a.sha256.localeCompare(b.sha256));
  const preparedPayload = examples.map((example) => example.sha256 + ":" + example.split).join("\n");
  const total = examples.length;
  const manifest: DatasetManifest & { sourceMetadata: Record<string, string> } = {
    formatVersion: 1,
    sourceSha256: sha256(["train", raw.train, "validation", raw.validation, "test", raw.test].join("\n")),
    preparedSha256: sha256(preparedPayload),
    splitSeed: 0,
    ratios: { train: counts.train / total, validation: counts.validation / total, test: counts.test / total },
    sourceLines,
    acceptedExamples: total,
    duplicatesRemoved,
    emptyLinesRemoved,
    splits: counts,
    examples,
    sourceMetadata: {
      name: options.get("source-name")!,
      url: options.get("source-url")!,
      license: options.get("license")!,
      attribution: options.get("attribution")!,
      splitPolicy: "preserved-source-provided-splits",
      trainPath: inputPaths.train,
      validationPath: inputPaths.validation,
      testPath: inputPaths.test,
    },
  };

  const output = options.get("output") ?? "artifacts/dataset-manifest.json";
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(manifest, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({
    output,
    sourceSha256: manifest.sourceSha256,
    preparedSha256: manifest.preparedSha256,
    acceptedExamples: total,
    duplicatesRemoved,
    emptyLinesRemoved,
    splits: counts,
    sourceMetadata: manifest.sourceMetadata,
  }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
