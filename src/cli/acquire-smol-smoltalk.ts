import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import {
  prepareSmolRow,
  sha256,
  SMOL_MAGPIE_SOURCE,
  SMOL_SMOLTALK_DATASET,
  SMOL_SMOLTALK_LICENSE,
  SMOL_SMOLTALK_REVISION,
  type SmolPreparedExample,
  type SmolPreparedSplit,
  type SmolSourceSplit,
} from "../data/smol-smoltalk.js";

const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const USER_AGENT = "Nexus-Licensed-SFT-Acquisition/1.0";

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    const value = args[i + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) throw new Error(`Invalid option near ${key}`);
    const name = key.slice(2);
    if (!["output", "max-train-rows", "max-test-rows"].includes(name)) throw new Error(`Unknown option ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option ${key}`);
    options.set(name, value);
    i += 1;
  }
  return options;
}

function boundedInt(options: Map<string, string>, key: string, fallback: number, min: number, max: number): number {
  const raw = options.get(key);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`--${key} must be an integer in [${min}, ${max}].`);
  }
  return value;
}

async function fetchBytes(url: URL): Promise<Uint8Array> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": USER_AGENT },
        signal: AbortSignal.timeout(120_000),
        cache: "no-store",
      });
      if (response.ok) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.byteLength === 0 || bytes.byteLength > MAX_RESPONSE_BYTES) {
          throw new Error(`Unexpected response size from ${url.hostname}: ${bytes.byteLength} bytes.`);
        }
        return bytes;
      }
      const detail = (await response.text()).slice(0, 400).replace(/\s+/gu, " ").trim();
      lastError = new Error(`HTTP ${response.status} from ${url.hostname}${url.pathname}${detail ? `: ${detail}` : ""}`);
      if (response.status < 500 && response.status !== 429) break;
    } catch (error) {
      lastError = error;
    }
    if (attempt < 3) await delay(1000 * 2 ** attempt);
  }
  throw new Error(`Request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

async function verifyRevision(): Promise<void> {
  const url = new URL(`https://huggingface.co/api/datasets/${SMOL_SMOLTALK_DATASET}`);
  const response = await fetchBytes(url);
  const metadata = JSON.parse(new TextDecoder().decode(response)) as { sha?: unknown; cardData?: { license?: unknown } };
  if (metadata.sha !== SMOL_SMOLTALK_REVISION) {
    throw new Error(`Smol-SmolTalk revision changed: expected ${SMOL_SMOLTALK_REVISION}, received ${String(metadata.sha)}. Review the new revision before updating the pin.`);
  }
  if (metadata.cardData?.license !== SMOL_SMOLTALK_LICENSE) {
    throw new Error(`Smol-SmolTalk license mismatch: expected ${SMOL_SMOLTALK_LICENSE}, received ${String(metadata.cardData?.license)}.`);
  }
}

interface ViewerResponse {
  readonly rows?: readonly { readonly row_idx: number; readonly row: unknown }[];
  readonly num_rows_total?: number;
  readonly partial?: boolean;
  readonly features?: readonly { readonly name: string }[];
}

async function fetchPage(split: SmolSourceSplit, offset: number, length: number): Promise<ViewerResponse> {
  // The Filter API currently returns HTTP 500 for this dataset. Read only
  // bounded slices and apply the exact-source allowlist locally rather than
  // downloading Parquet files or the full mixed-source corpus.
  const url = new URL("https://datasets-server.huggingface.co/rows");
  url.searchParams.set("dataset", SMOL_SMOLTALK_DATASET);
  url.searchParams.set("config", "default");
  url.searchParams.set("split", split);
  url.searchParams.set("offset", String(offset));
  url.searchParams.set("length", String(length));
  const response = await fetchBytes(url);
  const parsed = JSON.parse(new TextDecoder().decode(response)) as ViewerResponse;
  if (parsed.partial === true) throw new Error("Dataset Viewer returned a partial row index; refusing an incomplete corpus.");
  if (!Array.isArray(parsed.rows) || !Array.isArray(parsed.features) ||
      !parsed.features.some((feature) => feature.name === "source") ||
      !parsed.features.some((feature) => feature.name === "messages") ||
      !Number.isSafeInteger(parsed.num_rows_total)) {
    throw new Error("Unexpected Smol-SmolTalk Dataset Viewer response schema.");
  }
  return parsed;
}

async function collectRows(split: SmolSourceSplit, maxRows: number): Promise<{ examples: SmolPreparedExample[]; scannedRows: number; datasetRowsTotal: number; sampleOffsets: number[] }> {
  const examples: SmolPreparedExample[] = [];
  let scannedRows = 0;
  const seenContent = new Map<string, SmolPreparedSplit>();
  const metadata = await fetchPage(split, 0, 1);
  const datasetRowsTotal = metadata.num_rows_total!;
  // Distribute a strict row budget across the entire official split instead of
  // assuming that the approved source occupies row zero or the first N rows.
  const sampleOffsets = sampledPageOffsets(datasetRowsTotal, maxRows, 100);
  const seenRowIds = new Set<number>();
  for (const offset of sampleOffsets) {
    const length = Math.min(100, maxRows - scannedRows, datasetRowsTotal - offset);
    if (length <= 0) break;
    const page = await fetchPage(split, offset, length);
    if (page.num_rows_total !== datasetRowsTotal) throw new Error("Smol-SmolTalk dataset row count changed while sampling; refusing a mixed snapshot.");
    for (const item of page.rows!) {
      if (!Number.isSafeInteger(item.row_idx) || item.row_idx < 0) throw new Error("Invalid source row index.");
      // If small splits cause nearby pages to overlap, ignore repeated source row IDs.
      if (seenRowIds.has(item.row_idx)) continue;
      seenRowIds.add(item.row_idx);
      scannedRows += 1;
      const prepared = prepareSmolRow({ row_idx: item.row_idx, row: item.row }, split);
      if (!prepared) continue;
      const previousSplit = seenContent.get(prepared.contentSha256);
      if (previousSplit) {
        if (previousSplit !== prepared.split) throw new Error(`Duplicate example crosses ${previousSplit}/${prepared.split} boundary (sha256 ${prepared.contentSha256}).`);
        continue;
      }
      seenContent.set(prepared.contentSha256, prepared.split);
      examples.push(prepared);
    }
  }
  return { examples, scannedRows, datasetRowsTotal, sampleOffsets };
}

function textBySplit(examples: readonly SmolPreparedExample[]): Record<SmolPreparedSplit, string[]> {
  const result: Record<SmolPreparedSplit, string[]> = { train: [], validation: [], test: [] };
  for (const item of examples) result[item.split].push(item.text);
  return result;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const maxTrainRows = boundedInt(options, "max-train-rows", 1200, 100, 20_000);
  const maxTestRows = boundedInt(options, "max-test-rows", 200, 25, 2_000);
  const output = options.get("output") ?? "artifacts/smol-smoltalk";
  await verifyRevision();

  const [trainPage, testPage] = await Promise.all([
    collectRows("train", maxTrainRows),
    collectRows("test", maxTestRows),
  ]);
  const allExamples = [...trainPage.examples, ...testPage.examples];
  const digestToSplit = new Map<string, SmolPreparedSplit>();
  for (const item of allExamples) {
    const existing = digestToSplit.get(item.contentSha256);
    if (existing && existing !== item.split) {
      throw new Error(`Exact duplicate crosses ${existing}/${item.split} split boundary (sha256 ${item.contentSha256}).`);
    }
    digestToSplit.set(item.contentSha256, item.split);
  }

  const splitTexts = textBySplit(allExamples);
  if (splitTexts.train.length < 50 || splitTexts.validation.length < 10 || splitTexts.test.length < 10) {
    throw new Error(`Insufficient usable source examples after filtering: train=${splitTexts.train.length}, validation=${splitTexts.validation.length}, test=${splitTexts.test.length}.`);
  }

  await verifyRevision();
  // This pinned dataset revision has README.md and data files but no standalone
  // LICENSE file. Preserve the verified card license and canonical license URL
  // instead of pretending an upstream LICENSE file was retrieved.
  const licenseNotice = [
    "Dataset: " + SMOL_SMOLTALK_DATASET,
    "Pinned revision: " + SMOL_SMOLTALK_REVISION,
    "Dataset card license field (verified via Hugging Face Hub API): " + SMOL_SMOLTALK_LICENSE,
    "Selected source label: " + SMOL_MAGPIE_SOURCE,
    "License evidence: https://huggingface.co/datasets/HuggingFaceTB/smoltalk (the upstream card documents Apache-2.0 for the newly created Smol-Magpie-Ultra subset).",
    "Canonical Apache-2.0 license text: https://www.apache.org/licenses/LICENSE-2.0.txt",
    "This source revision has no standalone LICENSE file; this is a provenance notice, not a copy of the license text.",
  ].join("\n") + "\n";
  const manifest = {
    format: "nexus-reviewed-conversation-corpus",
    formatVersion: 1,
    dataset: SMOL_SMOLTALK_DATASET,
    revision: SMOL_SMOLTALK_REVISION,
    sourceUrl: `https://huggingface.co/datasets/${SMOL_SMOLTALK_DATASET}/tree/${SMOL_SMOLTALK_REVISION}`,
    license: SMOL_SMOLTALK_LICENSE,
    attribution: "Hugging Face Smol Models Research contributors, Smol-SmolTalk; subset source smol-magpie-ultra-short; Apache-2.0 as declared in the pinned dataset card. See the upstream SmolTalk card and SmolLM2 citation (arXiv:2502.02737).",
    licenseEvidence: { datasetCardUrl: `https://huggingface.co/datasets/${SMOL_SMOLTALK_DATASET}/tree/${SMOL_SMOLTALK_REVISION}`, hubCardLicense: SMOL_SMOLTALK_LICENSE, subsetLicenseSource: "https://huggingface.co/datasets/HuggingFaceTB/smoltalk", canonicalLicenseText: "https://www.apache.org/licenses/LICENSE-2.0.txt", standaloneLicenseFilePresent: false },
    retrievalMethod: "Hugging Face Dataset Viewer /rows API at default config with a bounded scan and exact source-label allowlist applied locally",
    filter: { source: SMOL_MAGPIE_SOURCE, licenseScope: "only the source label for the Apache-2.0 Smol-Magpie-Ultra-derived subset; no mixed-source rows accepted" },
    caps: { maxTrainRowsScanned: maxTrainRows, maxTestRowsScanned: maxTestRows },
    rows: { trainDatasetRowsTotal: trainPage.datasetRowsTotal, testDatasetRowsTotal: testPage.datasetRowsTotal, trainScanned: trainPage.scannedRows, testScanned: testPage.scannedRows, trainSampleOffsets: trainPage.sampleOffsets, testSampleOffsets: testPage.sampleOffsets },
    acceptedExamples: allExamples.length,
    splits: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, examples.length])),
    sourceFileSha256: { train: sha256(trainPage.examples.map((item) => `${item.rowIndex}:${item.contentSha256}`).join("\n")), test: sha256(testPage.examples.map((item) => `${item.rowIndex}:${item.contentSha256}`).join("\n")) },
    splitSha256: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, sha256(examples.join("\n") + "\n")])),
    rowIds: Object.fromEntries(Object.entries(textBySplit(allExamples)).map(([split, examples]) => [split, allExamples.filter((item) => item.split === split).map((item) => ({ id: item.id, rowIndex: item.rowIndex, sourceSplit: item.sourceSplit, contentSha256: item.contentSha256 }))])),
    filtering: [
      "exact source label smol-magpie-ultra-short only",
      "single user/assistant turn, optional leading system message",
      "unsupported/multi-turn role layouts excluded",
      "obvious email addresses and phone-like strings excluded",
      "short, oversized, or invalid message content excluded",
      "exact duplicates removed within split; cross-split duplicates fail closed",
      "train-source rows deterministically divided into train/validation; official source test rows remain test",
    ],
    warning: "Research data only. This adapter intentionally excludes mixed-source records. Re-verify source-level provenance and evaluation quality before production use.",
  };

  await mkdir(dirname(output), { recursive: true });
  const staging = `${output}.staging-${process.pid}`;
  await rm(staging, { recursive: true, force: true });
  await mkdir(staging, { recursive: true });
  try {
    for (const [split, examples] of Object.entries(splitTexts)) {
      await writeFile(`${staging}/${split}.txt`, examples.join("\n") + "\n", "utf8");
    }
    await writeFile(`${staging}/manifest.json`, JSON.stringify(manifest, null, 2) + "\n", "utf8");
    await writeFile(`${staging}/SMOL-SMOLTALK-LICENSE-NOTICE.txt`, licenseNotice, "utf8");
    await writeFile(`${staging}/SMOL-SMOLTALK-CITATION.txt`, "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. arXiv:2502.02737. https://arxiv.org/abs/2502.02737\n");
    await rm(output, { recursive: true, force: true });
    await rename(staging, output);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({
    output, dataset: SMOL_SMOLTALK_DATASET, revision: SMOL_SMOLTALK_REVISION,
    source: SMOL_MAGPIE_SOURCE, license: SMOL_SMOLTALK_LICENSE,
    datasetRowsTotal: { train: trainPage.datasetRowsTotal, test: testPage.datasetRowsTotal },
    sampledOffsets: { train: trainPage.sampleOffsets, test: testPage.sampleOffsets },
    scannedRows: { train: trainPage.scannedRows, test: testPage.scannedRows },
    acceptedExamples: allExamples.length, splits: manifest.splits,
  }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
