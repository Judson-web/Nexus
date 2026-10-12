import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { TransformerModel, type TransformerCheckpoint } from "./transformer-model.js";

const MAX_CHECKPOINT_BYTES = 50 * 1024 * 1024;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;

export interface VerifiedTransformerArtifact {
  readonly model: TransformerModel;
  readonly sha256: string;
  readonly bytes: number;
}

/**
 * Load a pinned research checkpoint only when its companion checksum matches.
 * The checksum sidecar is emitted by train:transformer:manifest.
 */
export async function loadVerifiedTransformerArtifact(
  checkpointPath: string,
): Promise<VerifiedTransformerArtifact> {
  if (!checkpointPath.trim()) throw new Error("Checkpoint path must not be empty.");
  const metadata = await stat(checkpointPath);
  if (!metadata.isFile()) throw new Error("Transformer checkpoint path must point to a file.");
  if (metadata.size <= 0 || metadata.size > MAX_CHECKPOINT_BYTES) {
    throw new Error("Transformer checkpoint must be between 1 byte and 50 MiB.");
  }

  const [bytes, sidecar] = await Promise.all([
    readFile(checkpointPath),
    readFile(`${checkpointPath}.sha256`, "utf8"),
  ]);
  const match = /^([a-f0-9]{64})\\s+(\\d+)\\s*$/.exec(sidecar);
  if (!match || !SHA256_PATTERN.test(match[1]!)) {
    throw new Error("Transformer checksum sidecar is malformed.");
  }
  const expectedBytes = Number(match[2]);
  if (!Number.isSafeInteger(expectedBytes) || expectedBytes !== bytes.byteLength) {
    throw new Error("Transformer checkpoint size does not match its checksum sidecar.");
  }
  const actualSha256 = createHash("sha256").update(bytes).digest("hex");
  if (actualSha256 !== match[1]) throw new Error("Transformer checkpoint checksum verification failed.");

  let parsed: unknown;
  try {
    parsed = JSON.parse(bytes.toString("utf8")) as unknown;
  } catch {
    throw new Error("Transformer checkpoint is not valid JSON.");
  }
  if (!parsed || typeof parsed !== "object" || !("format" in parsed) ||
      parsed.format !== "nexus-transformer" || !("version" in parsed) ||
      parsed.version !== 1 || !("config" in parsed) || !("weights" in parsed)) {
    throw new Error("Transformer checkpoint has an unsupported format.");
  }
  const model = TransformerModel.fromCheckpoint(parsed as TransformerCheckpoint);
  return { model, sha256: actualSha256, bytes: bytes.byteLength };
}
