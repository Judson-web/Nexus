import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import {
  COMMON_PILE_ARXIV_ABSTRACTS,
  reviewRows,
  sha256,
  type CommonPileRow,
  type SourceSplit,
} from "../data/common-pile.js";

interface DatasetRowsResponse {
  readonly rows?: readonly { readonly row?: CommonPileRow }[];
  readonly num_rows_total?: number;
  readonly error?: string;
}

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const name = key.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    if (!["documents", "max-rows", "output", "timeout-ms"].includes(name)) {
      throw new Error(`Unknown option: ${key}`);
    }
    options.set(name, value);
    i += 1;
  }
  return options;
}

function integerOption(options: Map<string, string>, name: string, fallback: number, min: number, max: number): number {
  const raw = options.get(name);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`--${name} must be an integer between ${min} and ${max}.`);
  }
  return value;
}

async function fetchPage(offset: number, length: number, timeoutMs: number): Promise<CommonPileRow[]> {
  const url = new URL("https://datasets-server.huggingface.co/rows");
  url.searchParams.set("dataset", COMMON_PILE_ARXIV_ABSTRACTS.dataset);
  url.searchParams.set("config", COMMON_PILE_ARXIV_ABSTRACTS.config);
  url.searchParams.set("split", COMMON_PILE_ARXIV_ABSTRACTS.split);
  url.searchParams.set("revision", COMMON_PILE_ARXIV_ABSTRACTS.revision);
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("length", String(length));
  const response = await fetch(url, {
    headers: { accept: "application/json", "user-agent": "Nexus-Corpus-Acquisition/1.0" },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`Hugging Face dataset viewer returned HTTP ${response.status} at offset ${offset}; no data was written.`);
  }
  const payload = await response.json() as DatasetRowsResponse;
  if (!Array.isArray(payload.rows)) {
    throw new Error(`Dataset viewer response did not contain rows: ${payload.error ?? "unknown response shape"}`);
  }
  return payload.rows.map((item) => {
    if (!item.row || typeof item.row.id !== "string" || typeof item.row.text !== "string"
      || typeof item.row.source !== "string" || !item.row.metadata || typeof item.row.metadata !== "object") {
      throw new Error("Dataset viewer returned a row with an unexpected schema; no data was written.");
    }
    return item.row;
  });
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const targetDocuments = integerOption(options, "documents", 200, 3, 1000);
  const maxRows = integerOption(options, "max-rows", 5000, 100, 20000);
  const timeoutMs = integerOption(options, "timeout-ms", 20000, 1000, 120000);
  const outputDirectory = options.get("output") ?? "artifacts/common-pile-arxiv-abstracts";
  const rows: CommonPileRow[] = [];
  let offset = 0;
  let reviewed = reviewRows(rows, targetDocuments);
  while (offset < maxRows && reviewed.accepted.length < targetDocuments) {
    const pageLength = Math.min(100, maxRows - offset);
    const page = await fetchPage(offset, pageLength, timeoutMs);
    if (page.length === 0) break;
    rows.push(...page);
    offset += page.length;
    reviewed = reviewRows(rows, targetDocuments);
    if (page.length < pageLength) break;
  }

  if (reviewed.accepted.length !== targetDocuments) {
    throw new Error(`Bounded scan found only ${reviewed.accepted.length} of ${targetDocuments} eligible unique CC0 records after scanning ${rows.length} rows. Increase --max-rows; no data was written.`);
  }
  const splitCounts: Record<SourceSplit, number> = { train: 0, validation: 0, test: 0 };
  for (const item of reviewed.accepted) splitCounts[item.split] += 1;
  if (Object.values(splitCounts).some((count) => count === 0)) {
    throw new Error("Bounded scan did not populate all splits. Retry with a larger --max-rows or --documents value; no data was written.");
  }

  const bySplit: Record<SourceSplit, string[]> = { train: [], validation: [], test: [] };
  for (const item of reviewed.accepted) bySplit[item.split].push(item.text);
  const records = reviewed.accepted.map((item) => JSON.stringify(item)).join("\n") + "\n";
  const manifest = {
    format: "nexus-reviewed-corpus",
    formatVersion: 1,
    dataset: COMMON_PILE_ARXIV_ABSTRACTS,
    retrieval: {
      method: "Hugging Face datasets-server rows API",
      rowsScanned: rows.length,
      maxRows,
      targetDocuments,
      retrievedAt: new Date().toISOString(),
      rejectedLicenseOrMetadata: reviewed.rejectedLicenseOrMetadata,
      duplicateDocuments: reviewed.duplicateDocuments,
    },
    splitPolicy: "stable source-record ID hash; all abstract text remains in its source record split",
    splitCounts,
    acceptedRecords: reviewed.accepted.length,
    sourceRecordsSha256: sha256(records),
    preparedSplitsSha256: {
      train: sha256(bySplit.train.join("\n") + "\n"),
      validation: sha256(bySplit.validation.join("\n") + "\n"),
      test: sha256(bySplit.test.join("\n") + "\n"),
    },
    records: reviewed.accepted.map(({ id, textSha256, split, sourceUrl, authors, license, fullTextLicense }) => ({
      id, textSha256, split, sourceUrl, authors, license, fullTextLicense,
    })),
    warning: "Research-only acquisition. Each selected record declares CC0 in source metadata; review upstream provenance and legal suitability before redistribution or production use. This subset contains abstracts, not full papers.",
  };

  await mkdir(outputDirectory, { recursive: true });
  for (const split of ["train", "validation", "test"] as const) {
    const path = join(outputDirectory, `${split}.txt`);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, bySplit[split].join("\n") + "\n", { encoding: "utf8", flag: "wx" });
  }
  await writeFile(join(outputDirectory, "records.jsonl"), records, { encoding: "utf8", flag: "wx" });
  await writeFile(join(outputDirectory, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify({
    outputDirectory,
    dataset: COMMON_PILE_ARXIV_ABSTRACTS.dataset,
    revision: COMMON_PILE_ARXIV_ABSTRACTS.revision,
    rowsScanned: rows.length,
    acceptedRecords: reviewed.accepted.length,
    rejectedLicenseOrMetadata: reviewed.rejectedLicenseOrMetadata,
    duplicateDocuments: reviewed.duplicateDocuments,
    splitCounts,
    manifestSha256: sha256(JSON.stringify(manifest)),
  }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
