import { createHash } from "node:crypto";

export const SMOL_SMOLTALK_DATASET = "HuggingFaceTB/smol-smoltalk";
export const SMOL_SMOLTALK_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc";
export const SMOL_SMOLTALK_LICENSE = "apache-2.0";
export const SMOL_MAGPIE_SOURCE = "smol-magpie-ultra-short";

export type SmolSourceSplit = "train" | "test";
export type SmolPreparedSplit = "train" | "validation" | "test";

export interface SmolPreparedExample {
  readonly id: string;
  readonly contentSha256: string;
  readonly rowIndex: number;
  readonly sourceSplit: SmolSourceSplit;
  readonly split: SmolPreparedSplit;
  readonly source: typeof SMOL_MAGPIE_SOURCE;
  readonly text: string;
}

export interface DatasetViewerRow {
  readonly row_idx: number;
  readonly row: {
    readonly source: string;
    readonly messages: readonly { readonly role: string; readonly content: string }[];
  };
}

export function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableBucket(value: string): number {
  return Number.parseInt(sha256(value).slice(0, 8), 16) / 0x1_0000_0000;
}

function normalizeContent(value: string): string {
  return value.normalize("NFC").replace(/\s+/gu, " ").trim();
}

function safeText(value: string, maxCharacters: number): boolean {
  const text = normalizeContent(value);
  if (text.length < 12 || text.length > maxCharacters) return false;
  if (/(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/iu.test(text)) return false;
  if (/(?:\+?\d[\d ().-]{8,}\d)/u.test(text)) return false;
  return true;
}

function parseMessages(value: unknown): readonly { role: string; content: string }[] | null {
  if (!Array.isArray(value) || (value.length !== 2 && value.length !== 3)) return null;
  const messages: { role: string; content: string }[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") return null;
    const record = item as { role?: unknown; content?: unknown };
    if (typeof record.role !== "string" || typeof record.content !== "string") return null;
    messages.push({ role: record.role.toLowerCase(), content: record.content });
  }
  const roles = messages.map((message) => message.role);
  const valid = roles.join(",") === "user,assistant" || roles.join(",") === "system,user,assistant";
  return valid ? messages : null;
}

/**
 * Converts only the explicitly Apache-2.0 Smol-Magpie-Ultra short subset into
 * the current Nexus single-turn SFT format. Mixed-source rows are rejected.
 */
export function prepareSmolRow(
  input: unknown,
  sourceSplit: SmolSourceSplit,
): SmolPreparedExample | null {
  if (!input || typeof input !== "object") return null;
  const rowRecord = input as Partial<DatasetViewerRow>;
  if (!Number.isSafeInteger(rowRecord.row_idx) || rowRecord.row_idx! < 0 ||
      !rowRecord.row || typeof rowRecord.row !== "object") return null;
  const row = rowRecord.row as { source?: unknown; messages?: unknown };
  if (row.source !== SMOL_MAGPIE_SOURCE) return null;

  const messages = parseMessages(row.messages);
  if (!messages) return null;
  const system = messages.length === 3 ? normalizeContent(messages[0]!.content) : "";
  const user = normalizeContent(messages[messages.length - 2]!.content);
  const answer = normalizeContent(messages[messages.length - 1]!.content);
  if (!safeText(user, 2500) || !safeText(answer, 3500) || (system && !safeText(system, 1200))) return null;

  // Embed the optional system instruction into the prompt context. Escape the
  // delimiter so the assistant-only trainer can find one unambiguous boundary.
  const prompt = [system ? "System instruction: " + system : "", system ? "User asks: " + user : user]
    .filter(Boolean).join(" ").replace(/ Assistant: /gu, " Assistant : ");
  const text = `User: ${prompt} Assistant: ${answer}`;
  if (Buffer.byteLength(text, "utf8") > 6000) return null;

  const contentSha256 = sha256(text);
  const rowIndex = rowRecord.row_idx!;
  const id = sha256(`${SMOL_SMOLTALK_REVISION}:${sourceSplit}:${rowIndex}:${contentSha256}`);
  const split: SmolPreparedSplit = sourceSplit === "test"
    ? "test"
    : stableBucket(`split:${id}`) < 0.95 ? "train" : "validation";

  return { id, contentSha256, rowIndex, sourceSplit, split, source: SMOL_MAGPIE_SOURCE, text };
}
