import { createHash } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { mkdir } from "node:fs/promises";
import { BigramModel } from "../model/bigram-model.js";

export interface CheckpointManifest {
  readonly sha256: string;
  readonly bytes: number;
}

export async function saveCheckpoint(path: string, model: BigramModel): Promise<CheckpointManifest> {
  const content = JSON.stringify(model.toCheckpoint());
  const bytes = Buffer.from(content, "utf8");
  const manifest = { sha256: createHash("sha256").update(bytes).digest("hex"), bytes: bytes.length };
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp`;
  await writeFile(temporaryPath, bytes, { flag: "w" });
  await rename(temporaryPath, path);
  await writeFile(`${path}.sha256`, `${manifest.sha256}  ${bytes.length}\n`, { flag: "w" });
  return manifest;
}

export async function loadCheckpoint(path: string): Promise<BigramModel> {
  const bytes = await readFile(path);
  const expected = (await readFile(`${path}.sha256`, "utf8")).trim().split(/\s+/)[0];
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (!expected || expected !== actual) throw new Error("Checkpoint checksum mismatch; refusing to load.");
  return BigramModel.fromCheckpoint(JSON.parse(bytes.toString("utf8")) as unknown);
}
