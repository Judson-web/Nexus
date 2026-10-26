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

function textIssue(value: string, maxCharacters: number): string | null {
  const text = normalizeContent(value);
  if (text.length < 12) return "text-too-short";
  if (text.length > maxCharacters) return "text-too-long";
  if (/(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,})/iu.test(text)) return "email-like-content";
  if (/(?:\\+?\\d[\\d ().-]{8,}\\d)/u.test(text)) return "phone-like-content";
  return null;
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
export interface SmolRowPreparation {
  readonly example: SmolPreparedExample | null;
  readonly rejectionReason: string | null;
}

/** Prepare an example while exposing a non-content rejection code for audits. */
export function prepareSmolRowWithReason(
  input: unknown,
  sourceSplit: SmolSourceSplit,
): SmolRowPreparation {
  const reject = (rejectionReason: string): SmolRowPreparation => ({ example: null, rejectionReason });
  if (!input || typeof input !== "object") return reject("malformed-record");
  const rowRecord = input as Partial<DatasetViewerRow>;
  if (!Number.isSafeInteger(rowRecord.row_idx) || rowRecord.row_idx! < 0) return reject("invalid-row-index");
  if (!rowRecord.row || typeof rowRecord.row !== "object") return reject("missing-row-object");
  const row = rowRecord.row as { source?: unknown; messages?: unknown };
  if (row.source !== SMOL_MAGPIE_SOURCE) return reject("source-mismatch");
  if (!Array.isArray(row.messages)) return reject("messages-not-array");
  if (row.messages.length !== 2 && row.messages.length !== 3) return reject("unsupported-message-count");

  const messages: { role: string; content: string }[] = [];
  for (const item of row.messages) {
    if (!item || typeof item !== "object") return reject("malformed-message");
    const record = item as { role?: unknown; content?: unknown };
    if (typeof record.role !== "string" || typeof record.content !== "string") return reject("message-fields-missing");
    messages.push({ role: record.role.toLowerCase(), content: record.content });
  }
  const roles = messages.map((message) => message.role);
  const validRoles = roles.join(",") === "user,assistant" || roles.join(",") === "system,user,assistant";
  if (!validRoles) return reject("unsupported-role-sequence:" + roles.map((role) => role.slice(0, 24)).join(","));

  const system = messages.length === 3 ? normalizeContent(messages[0]!.content) : "";
  const user = normalizeContent(messages[messages.length - 2]!.content);
  const answer = normalizeContent(messages[messages.length - 1]!.content);
  const userIssue = textIssue(user, 2500);
  if (userIssue) return reject("user-" + userIssue);
  const answerIssue = textIssue(answer, 3500);
  if (answerIssue) return reject("assistant-" + answerIssue);
  if (system) {
    const systemIssue = textIssue(system, 1200);
    if (systemIssue) return reject("system-" + systemIssue);
  }

  // Escape the prompt delimiter so the assistant-only trainer sees one boundary.
  const prompt = [system ? "System instruction: " + system : "", system ? "User asks: " + user : user]
    .filter(Boolean).join(" ").replace(/ Assistant: /gu, " Assistant : ");
  const text = `User: ${prompt} Assistant: ${answer}`;
  if (Buffer.byteLength(text, "utf8") > 6000) return reject("combined-too-large");

  const contentSha256 = sha256(text);
  const rowIndex = rowRecord.row_idx!;
  const id = sha256(`${SMOL_SMOLTALK_REVISION}:${sourceSplit}:${rowIndex}:${contentSha256}`);
  const split: SmolPreparedSplit = sourceSplit === "test"
    ? "test"
    : stableBucket(`split:${id}`) < 0.95 ? "train" : "validation";
  return {
    example: { id, contentSha256, rowIndex, sourceSplit, split, source: SMOL_MAGPIE_SOURCE, text },
    rejectionReason: null,
  };
}

/** Converts only the approved Apache-2.0 subset into Nexus's single-turn SFT format. */
export function prepareSmolRow(input: unknown, sourceSplit: SmolSourceSplit): SmolPreparedExample | null {
  return prepareSmolRowWithReason(input, sourceSplit).example;
}

/**
 * Select deterministic page starts across an official split while inspecting
 * no more than maxRows rows. Spreading sample windows avoids bias toward rows
 * at the beginning of a source-concatenated dataset.
 */
export function sampledPageOffsets(totalRows: number, maxRows: number, pageSize = 100): number[] {
  if (!Number.isSafeInteger(totalRows) || totalRows < 0) throw new RangeError("totalRows must be a non-negative integer.");
  if (!Number.isSafeInteger(maxRows) || maxRows < 1) throw new RangeError("maxRows must be a positive integer.");
  if (!Number.isSafeInteger(pageSize) || pageSize < 1 || pageSize > 100) throw new RangeError("pageSize must be an integer between 1 and 100.");
  if (totalRows === 0) return [];
  const pageCount = Math.min(Math.ceil(maxRows / pageSize), Math.ceil(totalRows / pageSize));
  const lastOffset = Math.max(0, totalRows - Math.min(pageSize, maxRows));
  if (pageCount <= 1) return [0];
  return [...new Set(Array.from({ length: pageCount }, (_, index) =>
    Math.min(lastOffset, Math.floor(index * lastOffset / (pageCount - 1))),
  ))].sort((a, b) => a - b);
}
