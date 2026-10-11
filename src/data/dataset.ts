import { createHash } from "node:crypto";

export type DatasetSplit = "train" | "validation" | "test";

export interface PreparedExample {
  readonly id: string;
  readonly text: string;
  readonly sha256: string;
  readonly split: DatasetSplit;
}

export interface DatasetManifest {
  readonly formatVersion: 1;
  readonly sourceSha256: string;
  readonly preparedSha256: string;
  readonly splitSeed: number;
  readonly ratios: { readonly train: number; readonly validation: number; readonly test: number };
  readonly sourceLines: number;
  readonly acceptedExamples: number;
  readonly duplicatesRemoved: number;
  readonly emptyLinesRemoved: number;
  readonly splits: Record<DatasetSplit, number>;
  readonly examples: readonly PreparedExample[];
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function stableBucket(hash: string, seed: number): number {
  const prefix = Number.parseInt(sha256(seed + ":" + hash).slice(0, 8), 16);
  return prefix / 0x1_0000_0000;
}

export interface PrepareDatasetOptions {
  readonly seed?: number;
  readonly trainRatio?: number;
  readonly validationRatio?: number;
  readonly testRatio?: number;
  readonly minimumCharacters?: number;
}

/**
 * Cleans line-oriented text corpora, removes exact normalized duplicates, and
 * assigns each example to a stable split based on its content hash and seed.
 * Split assignment is independent of input ordering, preventing accidental
 * split drift when a corpus is reordered.
 */
export function prepareDataset(source: string, options: PrepareDatasetOptions = {}): DatasetManifest {
  const seed = options.seed ?? 17;
  const trainRatio = options.trainRatio ?? 0.8;
  const validationRatio = options.validationRatio ?? 0.1;
  const testRatio = options.testRatio ?? 0.1;
  const minimumCharacters = options.minimumCharacters ?? 1;

  if (!Number.isSafeInteger(seed) || seed < 0) throw new RangeError("seed must be a non-negative safe integer.");
  if (![trainRatio, validationRatio, testRatio].every((ratio) => Number.isFinite(ratio) && ratio > 0)) {
    throw new RangeError("All split ratios must be finite positive numbers.");
  }
  if (Math.abs(trainRatio + validationRatio + testRatio - 1) > 1e-9) {
    throw new RangeError("Split ratios must sum to 1.");
  }
  if (!Number.isInteger(minimumCharacters) || minimumCharacters < 1) {
    throw new RangeError("minimumCharacters must be a positive integer.");
  }

  const lines = source.replace(/^\uFEFF/, "").split(/\r?\n/);
  const unique = new Map<string, string>();
  let emptyLinesRemoved = 0;
  let duplicatesRemoved = 0;
  for (const line of lines) {
    const normalized = line.normalize("NFC").replace(/\s+/gu, " ").trim();
    if (normalized.length < minimumCharacters) {
      emptyLinesRemoved += 1;
      continue;
    }
    const digest = sha256(normalized);
    if (unique.has(digest)) {
      duplicatesRemoved += 1;
      continue;
    }
    unique.set(digest, normalized);
  }

  const boundaries = { train: trainRatio, validation: trainRatio + validationRatio };
  const examples = [...unique.entries()]
    .map(([digest, text]) => {
      const bucket = stableBucket(digest, seed);
      const split: DatasetSplit = bucket < boundaries.train
        ? "train"
        : bucket < boundaries.validation ? "validation" : "test";
      return { id: digest.slice(0, 24), text, sha256: digest, split };
    })
    .sort((a, b) => a.sha256.localeCompare(b.sha256));
  const splits: Record<DatasetSplit, number> = { train: 0, validation: 0, test: 0 };
  for (const example of examples) splits[example.split] += 1;

  const preparedPayload = examples.map(({ sha256: digest, split }) => digest + ":" + split).join("\n");
  return {
    formatVersion: 1,
    sourceSha256: sha256(source),
    preparedSha256: sha256(preparedPayload),
    splitSeed: seed,
    ratios: { train: trainRatio, validation: validationRatio, test: testRatio },
    sourceLines: lines.length,
    acceptedExamples: examples.length,
    duplicatesRemoved,
    emptyLinesRemoved,
    splits,
    examples,
  };
}
