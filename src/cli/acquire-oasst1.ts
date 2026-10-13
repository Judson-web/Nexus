import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const DATASET = "OpenAssistant/oasst1";
const REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b";
const LICENSE = "Apache-2.0";
const SOURCE_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1";
const PAGE_SIZE = 100;

interface Message {
  readonly message_id: string;
  readonly parent_id: string | null;
  readonly text: string;
  readonly role: string;
  readonly lang: string;
  readonly deleted?: boolean;
  readonly synthetic?: boolean;
  readonly message_tree_id: string;
  readonly detoxify?: Record<string, number | null>;
}
interface DatasetPage {
  readonly rows?: readonly { readonly row?: Message }[];
  readonly num_rows_total?: number;
  readonly error?: string;
}
interface Pair {
  readonly tree: string;
  readonly text: string;
  readonly messageId: string;
  readonly parentId: string;
  readonly split: "train" | "validation" | "test";
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
function bucket(value: string): number {
  return Number.parseInt(sha256(value).slice(0, 8), 16) / 0x1_0000_0000;
}
function safeText(value: string): boolean {
  const text = value.trim();
  if (text.length < 12 || text.length > 3000) return false;
  if (/(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/iu.test(text)) return false;
  if (/(?:\+?\d[\d ().-]{8,}\d)/u.test(text)) return false;
  return true;
}
function isUsable(message: Message): boolean {
  if (message.deleted || message.lang !== "en" || message.synthetic === true || !safeText(message.text)) return false;
  const toxicity = message.detoxify;
  if (toxicity && ["severe_toxicity", "sexual_explicit", "threat"].some((key) => {
    const score = toxicity[key];
    return typeof score === "number" && score >= 0.5;
  })) return false;
  return true;
}
async function fetchWithRetry(url: URL): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { accept: "application/json", "user-agent": "Nexus-Open-Dataset-Acquisition/1.0" },
        signal: AbortSignal.timeout(60_000),
        cache: "no-store",
      });
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status} from ${url.hostname}`);
      if (response.status < 500 && response.status !== 429) break;
    } catch (error) {
      lastError = error;
    }
    if (attempt < 3) await delay(500 * 2 ** attempt);
  }
  throw new Error(`Dataset request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}
async function verifyRevision(): Promise<void> {
  const response = await fetchWithRetry(new URL(`https://huggingface.co/api/datasets/${DATASET}`));
  const metadata = await response.json() as { sha?: unknown; cardData?: { license?: unknown } };
  if (metadata.sha !== REVISION) throw new Error(`OASST1 revision mismatch: expected ${REVISION}, received ${String(metadata.sha)}`);
  if (metadata.cardData?.license !== "apache-2.0") throw new Error(`OASST1 license mismatch: expected apache-2.0, received ${String(metadata.cardData?.license)}`);
}
async function fetchSplit(split: "train" | "validation", maxRows: number): Promise<Message[]> {
  const messages: Message[] = [];
  for (let offset = 0; offset < maxRows; offset += PAGE_SIZE) {
    const url = new URL("https://datasets-server.huggingface.co/rows");
    url.searchParams.set("dataset", DATASET);
    url.searchParams.set("config", "default");
    url.searchParams.set("split", split);
    url.searchParams.set("offset", String(offset));
    url.searchParams.set("length", String(Math.min(PAGE_SIZE, maxRows - offset)));
    const response = await fetchWithRetry(url);
    const page = await response.json() as DatasetPage;
    if (!Array.isArray(page.rows)) throw new Error(`Unexpected ${split} dataset response: ${page.error ?? "rows missing"}`);
    for (const item of page.rows) {
      const row = item.row;
      if (!row || typeof row.message_id !== "string" || typeof row.text !== "string" ||
          typeof row.role !== "string" || typeof row.lang !== "string" ||
          typeof row.message_tree_id !== "string") {
        throw new Error(`Malformed OASST1 ${split} row at offset ${offset}`);
      }
      messages.push(row);
    }
    if (page.rows.length < Math.min(PAGE_SIZE, maxRows - offset)) break;
  }
  return messages;
}
function pairsFrom(messages: readonly Message[], split: "train" | "test"): Pair[] {
  const byId = new Map(messages.map((message) => [message.message_id, message]));
  const pairs: Pair[] = [];
  const seen = new Set<string>();
  for (const answer of messages) {
    if (answer.role !== "assistant" || !answer.parent_id || !isUsable(answer)) continue;
    const prompt = byId.get(answer.parent_id);
    if (!prompt || prompt.role !== "prompter" && prompt.role !== "user" || !isUsable(prompt)) continue;
    const identity = `${prompt.message_id}:${answer.message_id}`;
    if (seen.has(identity)) continue;
    seen.add(identity);
    const text = `User: ${prompt.text.trim().replace(/\s+/gu, " ")} Assistant: ${answer.text.trim().replace(/\s+/gu, " ")}`;
    if (Buffer.byteLength(text, "utf8") > 5000) continue;
    const pairSplit = split === "test" ? "test" : bucket(answer.message_tree_id) < 0.9 ? "train" : "validation";
    pairs.push({ tree: answer.message_tree_id, text, messageId: answer.message_id, parentId: prompt.message_id, split: pairSplit });
  }
  return pairs;
}
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--") || !args[i + 1] || args[i + 1]!.startsWith("--")) throw new Error(`Invalid option near ${key}`);
    if (options.has(key.slice(2))) throw new Error(`Duplicate option ${key}`);
    options.set(key.slice(2), args[i + 1]!);
    i += 1;
  }
  const maxTrainRows = Number(options.get("train-rows") ?? "12000");
  const maxValidationRows = Number(options.get("validation-rows") ?? "4401");
  const output = options.get("output") ?? "artifacts/oasst1";
  for (const [name, value, max] of [["train-rows", maxTrainRows, 20000], ["validation-rows", maxValidationRows, 4401]] as const) {
    if (!Number.isSafeInteger(value) || value < 100 || value > max) throw new RangeError(`--${name} must be an integer between 100 and ${max}`);
  }
  await verifyRevision();
  const [trainMessages, validationMessages] = await Promise.all([
    fetchSplit("train", maxTrainRows),
    fetchSplit("validation", maxValidationRows),
  ]);
  await verifyRevision();

  const trainPairs = pairsFrom(trainMessages, "train");
  const testPairs = pairsFrom(validationMessages, "test");
  const trainTreeIds = new Set(trainMessages.map((message) => message.message_tree_id));
  if (testPairs.some((pair) => trainTreeIds.has(pair.tree))) {
    throw new Error("OASST1 train/test tree overlap detected; refusing dataset creation.");
  }
  const train = trainPairs.filter((pair) => pair.split === "train");
  const validation = trainPairs.filter((pair) => pair.split === "validation");
  const test = testPairs;
  if (train.length < 50 || validation.length < 5 || test.length < 5) {
    throw new Error(`Too few usable English pairs: train=${train.length}, validation=${validation.length}, test=${test.length}. Increase --train-rows or inspect the upstream schema.`);
  }
  const splitTexts = {
    train: train.map((pair) => pair.text),
    validation: validation.map((pair) => pair.text),
    test: test.map((pair) => pair.text),
  };
  const crossSplit = new Set<string>();
  for (const [split, examples] of Object.entries(splitTexts)) {
    for (const example of examples) {
      const digest = sha256(example.normalize("NFC").replace(/\s+/gu, " ").trim());
      if (crossSplit.has(digest)) throw new Error(`Duplicate normalized conversation across splits near ${split}; refusing leakage.`);
      crossSplit.add(digest);
    }
  }
  const manifest = {
    format: "nexus-reviewed-conversation-corpus",
    formatVersion: 1,
    dataset: DATASET,
    revision: REVISION,
    sourceUrl: SOURCE_URL,
    license: LICENSE,
    attribution: "OpenAssistant contributors, OpenAssistant Conversations Dataset (OASST1), Apache-2.0.",
    retrievedAt: new Date().toISOString(),
    selection: {
      trainRowsScanned: trainMessages.length,
      validationRowsScanned: validationMessages.length,
      englishPairs: trainPairs.length + testPairs.length,
      trainPairs: train.length,
      validationPairs: validation.length,
      testPairs: test.length,
      filters: ["English only", "deleted messages excluded", "synthetic messages excluded", "emails and phone-like strings excluded", "high toxicity scores excluded where available", "parent-user/assistant-child pairs only"],
      splitPolicy: "train trees are hash-split into train/validation; official OASST1 validation trees are held out as test",
    },
    sha256: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, sha256(examples.join("\n") + "\n")])),
    examples: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, examples.length])),
    warning: "Research dataset only. Preserve Apache-2.0 license and attribution; review the dataset card and contributor terms before redistribution or production use.",
  };
  await mkdir(dirname(output), { recursive: true });
  const staged = `${output}.staging-${process.pid}`;
  await mkdir(staged, { recursive: true });
  try {
    for (const [split, examples] of Object.entries(splitTexts)) {
      await writeFile(`${staged}/${split}.txt`, examples.join("\n") + "\n", "utf8");
    }
    await writeFile(`${staged}/manifest.json`, JSON.stringify(manifest, null, 2) + "\n", "utf8");
    // Replace only after all split files and provenance have been written.
    const { rename, rm } = await import("node:fs/promises");
    await rm(output, { recursive: true, force: true });
    await rename(staged, output);
  } catch (error) {
    const { rm } = await import("node:fs/promises");
    await rm(staged, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({ output, dataset: DATASET, revision: REVISION, license: LICENSE, splits: manifest.examples, manifestSha256: sha256(JSON.stringify(manifest)) }, null, 2));
}
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
