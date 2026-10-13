import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const DATASET = "Salesforce/wikitext";
const CONFIG = "wikitext-103-raw-v1";
const SPLITS = ["train", "validation", "test"] as const;
type Split = typeof SPLITS[number];

interface DatasetRow { readonly text?: unknown }
interface RowsResponse {
  readonly rows?: readonly { readonly row?: DatasetRow }[];
  readonly num_rows_total?: number;
  readonly error?: string;
}
interface DatasetMetadata { readonly sha?: unknown; readonly cardData?: { readonly license?: unknown } }

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const name = key.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    if (!["train-rows", "validation-rows", "test-rows", "max-scan-per-split", "timeout-ms", "output"].includes(name)) {
      throw new Error(`Unknown option: ${key}`);
    }
    options.set(name, value);
    i += 1;
  }
  return options;
}

function intOption(options: Map<string, string>, key: string, fallback: number, min: number, max: number): number {
  const raw = options.get(key);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`--${key} must be an integer from ${min} to ${max}.`);
  }
  return value;
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function normalize(value: string): string {
  return value.normalize("NFC").replace(/\s+/gu, " ").trim();
}

async function fetchJson<T>(url: URL | string, timeoutMs: number): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { accept: "application/json", "user-agent": "Nexus-WikiText-Acquisition/1.0" },
        signal: AbortSignal.timeout(timeoutMs),
        cache: "no-store",
      });
      if (response.ok) return await response.json() as T;
      lastError = new Error(`HTTP ${response.status} from ${url}`);
      if (response.status < 500 && response.status !== 429) break;
    } catch (error) {
      lastError = error;
    }
    if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** (attempt - 1)));
  }
  throw new Error(`Request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

async function getPinnedRevision(timeoutMs: number): Promise<string> {
  const metadata = await fetchJson<DatasetMetadata>(`https://huggingface.co/api/datasets/${DATASET}`, timeoutMs);
  if (typeof metadata.sha !== "string" || !/^[a-f0-9]{40}$/i.test(metadata.sha)) {
    throw new Error("Hugging Face did not return a valid immutable dataset revision.");
  }
  const license = metadata.cardData?.license;
  if (typeof license !== "string" || !["cc-by-sa-3.0", "cc-by-sa-3.0-de"].includes(license.toLowerCase())) {
    throw new Error(`Unexpected WikiText dataset license metadata: ${String(license)}. Refusing acquisition until reviewed.`);
  }
  return metadata.sha;
}

async function fetchSplit(split: Split, targetRows: number, maxScan: number, timeoutMs: number) {
  const seen = new Set<string>();
  const rows: { text: string; sha256: string; id: string }[] = [];
  let offset = 0;
  while (offset < maxScan && rows.length < targetRows) {
    const length = Math.min(100, maxScan - offset);
    const url = new URL("https://datasets-server.huggingface.co/rows");
    url.searchParams.set("dataset", DATASET);
    url.searchParams.set("config", CONFIG);
    url.searchParams.set("split", split);
    url.searchParams.set("offset", String(offset));
    url.searchParams.set("length", String(length));
    const payload = await fetchJson<RowsResponse>(url, timeoutMs);
    if (!Array.isArray(payload.rows)) throw new Error(`Dataset viewer returned malformed rows for ${split}: ${payload.error ?? "unknown error"}`);
    for (const item of payload.rows) {
      const raw = item.row?.text;
      if (typeof raw !== "string") throw new Error(`Unexpected row schema in ${split}; no output has been written.`);
      const text = normalize(raw);
      if (text.length < 40) continue;
      const digest = sha256(text);
      if (seen.has(digest)) continue;
      seen.add(digest);
      rows.push({ text, sha256: digest, id: `${split}-${digest.slice(0, 20)}` });
      if (rows.length >= targetRows) break;
    }
    offset += payload.rows.length;
    if (payload.rows.length < length) break;
  }
  if (rows.length < targetRows) {
    throw new Error(`Only ${rows.length}/${targetRows} eligible ${split} rows found after scanning ${offset}; no output was written.`);
  }
  return { rows, scanned: offset };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const timeoutMs = intOption(options, "timeout-ms", 120000, 1000, 120000);
  const target = {
    train: intOption(options, "train-rows", 20000, 100, 100000),
    validation: intOption(options, "validation-rows", 1000, 50, 10000),
    test: intOption(options, "test-rows", 1000, 50, 10000),
  };
  const maxScan = intOption(options, "max-scan-per-split", 100000, 100, 1000000);
  const outputDirectory = options.get("output") ?? "artifacts/wikitext-103";
  const revision = await getPinnedRevision(timeoutMs);
  const acquired = {} as Record<Split, { rows: { text: string; sha256: string; id: string }[]; scanned: number }>;
  for (const split of SPLITS) acquired[split] = await fetchSplit(split, target[split], maxScan, timeoutMs);
  const verifiedRevision = await getPinnedRevision(timeoutMs);
  if (revision !== verifiedRevision) {
    throw new Error(`WikiText dataset changed during acquisition (${revision} -> ${verifiedRevision}); no output was written.`);
  }

  // Keep the source-provided train/validation/test boundaries. Reject duplicate text
  // across splits, preferring held-out records so no evaluation text leaks into training.
  const globallySeen = new Set<string>();
  const splitRows = { train: [] as typeof acquired.train.rows, validation: [] as typeof acquired.validation.rows, test: [] as typeof acquired.test.rows };
  for (const split of ["test", "validation", "train"] as const) {
    for (const row of acquired[split].rows) {
      if (globallySeen.has(row.sha256)) continue;
      globallySeen.add(row.sha256);
      splitRows[split].push(row);
    }
  }
  for (const split of SPLITS) {
    if (splitRows[split].length < 1) throw new Error(`Split ${split} became empty after cross-split deduplication.`);
  }

  const manifest = {
    format: "nexus-wikitext-corpus",
    formatVersion: 1,
    dataset: DATASET,
    config: CONFIG,
    revision,
    datasetUrl: `https://huggingface.co/datasets/${DATASET}/tree/${revision}/${CONFIG}`,
    license: "CC BY-SA 3.0 (dataset card metadata verified; retain attribution and share-alike obligations)",
    licenseUrl: "https://creativecommons.org/licenses/by-sa/3.0/",
    retrieval: {
      method: "Hugging Face datasets-server rows API; immutable dataset revision checked before and after retrieval",
      retrievedAt: new Date().toISOString(),
      requestedRows: target,
      scannedRows: Object.fromEntries(SPLITS.map((split) => [split, acquired[split].scanned])),
      selectedRows: Object.fromEntries(SPLITS.map((split) => [split, splitRows[split].length])),
      duplicateRowsRemovedAcrossSplits: Object.fromEntries(SPLITS.map((split) => [split, acquired[split].rows.length - splitRows[split].length])),
    },
    splitPolicy: "Preserve source-provided WikiText-103 train/validation/test split; exact normalized-text deduplication prioritizes test then validation then train",
    sourceSha256: Object.fromEntries(SPLITS.map((split) => [split, sha256(splitRows[split].map((row) => row.sha256).join("\n"))])),
    records: Object.fromEntries(SPLITS.map((split) => [split, splitRows[split].map(({ id, sha256: textSha256 }) => ({ id, textSha256 }))])),
    counts: Object.fromEntries(SPLITS.map((split) => [split, splitRows[split].length])),
    warning: "Research dataset acquisition, not legal advice. WikiText is derived from Wikipedia and distributed under CC BY-SA; preserve source attribution and comply with share-alike requirements before redistributing derived corpora or model artifacts.",
  };

  await mkdir(dirname(outputDirectory), { recursive: true });
  const staging = await mkdtemp(`${outputDirectory}.staging-`);
  try {
    for (const split of SPLITS) await writeFile(join(staging, `${split}.txt`), splitRows[split].map((row) => row.text).join("\n") + "\n", "utf8");
    await writeFile(join(staging, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
    await rename(staging, outputDirectory);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({ outputDirectory, dataset: DATASET, config: CONFIG, revision, counts: manifest.counts, scannedRows: manifest.retrieval.scannedRows, sourceSha256: manifest.sourceSha256 }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
