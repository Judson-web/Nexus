import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { gunzipSync } from "node:zlib";

const DATASET = "OpenAssistant/oasst1";
const REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b";
const LICENSE = "Apache-2.0";
const SOURCE_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1";
const DATA_FILE = "2023-04-12_oasst_ready.messages.jsonl.gz";
const MAX_COMPRESSED_BYTES = 100 * 1024 * 1024;

interface Message {
  readonly message_id: string;
  readonly parent_id: string | null;
  readonly text: string;
  readonly role: string;
  readonly lang: string;
  readonly deleted?: boolean;
  readonly review_result?: boolean | null;
  readonly synthetic?: boolean;
  readonly message_tree_id: string;
  readonly detoxify?: Record<string, number | null>;
  readonly labels?: Record<string, { readonly value?: number | null; readonly count?: number }>;
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
  if (message.deleted || message.review_result === false || message.lang !== "en" ||
      message.synthetic === true || !safeText(message.text)) return false;
  const toxicity = message.detoxify;
  if (toxicity && ["severe_toxicity", "sexual_explicit", "threat"].some((key) => {
    const score = toxicity[key];
    return typeof score === "number" && score >= 0.5;
  })) return false;
  const labels = message.labels;
  if (labels && ["pii", "not_appropriate", "hate_speech", "sexual_content"].some((key) => {
    const label = labels[key];
    return label && typeof label.value === "number" && label.value >= 0.5;
  })) return false;
  return true;
}
async function fetchWithRetry(url: URL): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": "Nexus-Open-Dataset-Acquisition/1.0" },
        signal: AbortSignal.timeout(120_000),
        cache: "no-store",
      });
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status} from ${url.hostname}`);
      if (response.status < 500 && response.status !== 429) break;
    } catch (error) {
      lastError = error;
    }
    if (attempt < 3) await delay(1000 * 2 ** attempt);
  }
  throw new Error(`Dataset request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}
async function verifyRevision(): Promise<void> {
  const response = await fetchWithRetry(new URL(`https://huggingface.co/api/datasets/${DATASET}`));
  const metadata = await response.json() as { sha?: unknown; cardData?: { license?: unknown } };
  if (metadata.sha !== REVISION) throw new Error(`OASST1 revision mismatch: expected ${REVISION}, received ${String(metadata.sha)}`);
  if (metadata.cardData?.license !== "apache-2.0") {
    throw new Error(`OASST1 license mismatch: expected apache-2.0, received ${String(metadata.cardData?.license)}`);
  }
}
async function downloadPinnedFile(filename: string): Promise<Uint8Array> {
  const url = new URL(`https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/${filename}?download=true`);
  const response = await fetchWithRetry(url);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_COMPRESSED_BYTES) {
    throw new Error(`Pinned upstream file ${filename} has an invalid size: ${bytes.byteLength} bytes.`);
  }
  return bytes;
}
function parseMessages(text: string): Message[] {
  const messages: Message[] = [];
  for (const [index, line] of text.split(/\r?\n/u).entries()) {
    if (!line.trim()) continue;
    let row: unknown;
    try { row = JSON.parse(line) as unknown; } catch { throw new Error(`Invalid JSONL record at line ${index + 1}.`); }
    if (!row || typeof row !== "object") throw new Error(`Malformed message at line ${index + 1}.`);
    const message = row as Partial<Message>;
    if (typeof message.message_id !== "string" || typeof message.text !== "string" ||
        typeof message.role !== "string" || typeof message.lang !== "string" ||
        typeof message.message_tree_id !== "string" ||
        !(message.parent_id === undefined || message.parent_id === null || typeof message.parent_id === "string")) {
      throw new Error(`Unexpected OASST1 message schema at line ${index + 1}; fields=${Object.keys(message).join(",")}; parent_id_type=${typeof message.parent_id}; tree_id_type=${typeof message.message_tree_id}.`);
    }
    messages.push(message as Message);
  }
  if (messages.length < 1000) throw new Error(`Pinned OASST1 file returned only ${messages.length} messages; refusing incomplete data.`);
  return messages;
}
function makePairs(messages: readonly Message[]): Pair[] {
  const byId = new Map(messages.map((message) => [message.message_id, message]));
  const pairs: Pair[] = [];
  for (const answer of messages) {
    if (answer.role !== "assistant" || !answer.parent_id || !isUsable(answer)) continue;
    const prompt = byId.get(answer.parent_id);
    if (!prompt || (prompt.role !== "prompter" && prompt.role !== "user") ||
        prompt.message_tree_id !== answer.message_tree_id || !isUsable(prompt)) continue;
    // Escape the exact pair delimiter inside user content so the trainer can
    // distinguish prompt context from assistant-only loss targets.
    const promptText = prompt.text.trim().replace(/\s+/gu, " ").replace(/ Assistant: /gu, " Assistant : ");
    const answerText = answer.text.trim().replace(/\s+/gu, " ");
    const text = `User: ${promptText} Assistant: ${answerText}`;
    if (Buffer.byteLength(text, "utf8") > 5000) continue;
    const bucketValue = bucket(answer.message_tree_id);
    const split = bucketValue < 0.9 ? "train" : bucketValue < 0.95 ? "validation" : "test";
    pairs.push({ tree: answer.message_tree_id, text, messageId: answer.message_id, parentId: prompt.message_id, split });
  }
  // Deduplicate normalized examples globally before finalizing split files.
  // This prevents identical conversation pairs from crossing split boundaries.
  const seen = new Set<string>();
  return pairs.filter((pair) => {
    const digest = sha256(pair.text.normalize("NFC").replace(/\s+/gu, " ").trim());
    if (seen.has(digest)) return false;
    seen.add(digest);
    return true;
  });
}
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--") || !args[i + 1] || args[i + 1]!.startsWith("--")) {
      throw new Error(`Invalid option near ${key}`);
    }
    if (options.has(key.slice(2))) throw new Error(`Duplicate option ${key}`);
    options.set(key.slice(2), args[i + 1]!);
    i += 1;
  }
  for (const key of options.keys()) if (key !== "output" && key !== "max-trees") throw new Error(`Unknown option --${key}`);
  const maxTrees = Number(options.get("max-trees") ?? "500");
  if (!Number.isSafeInteger(maxTrees) || maxTrees < 100 || maxTrees > 10000) {
    throw new RangeError("--max-trees must be an integer between 100 and 10000");
  }
  const output = options.get("output") ?? "artifacts/oasst1";
  await verifyRevision();
  const [compressed, licenseBytes] = await Promise.all([
    downloadPinnedFile(DATA_FILE),
    downloadPinnedFile("LICENSE"),
  ]);
  const uncompressed = gunzipSync(compressed);
  const allMessages = parseMessages(uncompressed.toString("utf8"));
  await verifyRevision();
  const treeIds = [...new Set(allMessages.map((message) => message.message_tree_id))]
    .sort((left, right) => sha256("sample:" + left).localeCompare(sha256("sample:" + right)))
    .slice(0, maxTrees);
  const selectedTreeIds = new Set(treeIds);
  const messages = allMessages.filter((message) => selectedTreeIds.has(message.message_tree_id));
  const pairs = makePairs(messages);
  const splitTexts = {
    train: pairs.filter((pair) => pair.split === "train").map((pair) => pair.text),
    validation: pairs.filter((pair) => pair.split === "validation").map((pair) => pair.text),
    test: pairs.filter((pair) => pair.split === "test").map((pair) => pair.text),
  };
  if (splitTexts.train.length < 100 || splitTexts.validation.length < 10 || splitTexts.test.length < 10) {
    throw new Error(`Insufficient usable pairs: train=${splitTexts.train.length}, validation=${splitTexts.validation.length}, test=${splitTexts.test.length}.`);
  }
  const manifest = {
    format: "nexus-reviewed-conversation-corpus",
    formatVersion: 1,
    dataset: DATASET,
    revision: REVISION,
    sourceUrl: SOURCE_URL,
    sourceFile: DATA_FILE,
    license: LICENSE,
    attribution: "OpenAssistant contributors, OpenAssistant Conversations Dataset (OASST1), Apache-2.0.",
    retrievedAt: new Date().toISOString(),
    sourceFileSha256: sha256(compressed),
    uncompressedFileSha256: sha256(uncompressed),
    sourceMessages: allMessages.length,
    selectedTrees: treeIds.length,
    selectedMessages: messages.length,
    maxTrees,
    acceptedEnglishPairs: pairs.length,
    examples: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, examples.length])),
    splitPolicy: "Deterministic SHA-256 ranked subset of maxTrees, then tree-id hash split (90% train, 5% validation, 5% test); complete conversation trees stay within one split.",
    filters: [
      "English user/assistant parent-child pairs only",
      "deleted messages and negative review results excluded",
      "synthetic messages excluded",
      "obvious email addresses and phone-like strings excluded",
      "high toxicity and personal-information label scores excluded where available",
      "exact normalized conversation duplicates removed globally before split output",
    ],
    splitSha256: Object.fromEntries(Object.entries(splitTexts).map(([split, examples]) => [split, sha256(examples.join("\n") + "\n")])),
    warning: "Research dataset only. Retain the included OASST1-LICENSE and attribution; review upstream dataset terms before redistribution or production use.",
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
    await writeFile(`${staging}/OASST1-LICENSE`, licenseBytes);
    await rm(output, { recursive: true, force: true });
    await rename(staging, output);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({
    output,
    dataset: DATASET,
    revision: REVISION,
    license: LICENSE,
    sourceMessages: allMessages.length,
    selectedTrees: treeIds.length,
    selectedMessages: messages.length,
    acceptedEnglishPairs: pairs.length,
    splits: manifest.examples,
    sourceFileSha256: manifest.sourceFileSha256,
  }, null, 2));
}
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
