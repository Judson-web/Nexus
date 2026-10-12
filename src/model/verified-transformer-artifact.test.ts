import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";
import { loadVerifiedTransformerArtifact } from "./verified-transformer-artifact.js";

test("loads a Transformer checkpoint only when its checksum and byte count match", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-artifact-"));
  const path = join(directory, "model.json");
  try {
    const checkpoint = new TransformerModel({
      ...DEFAULT_TRANSFORMER_CONFIG,
      contextSize: 2,
      embeddingSize: 4,
      headCount: 2,
      feedForwardSize: 4,
      seed: 17,
    }).toCheckpoint();
    const bytes = Buffer.from(JSON.stringify(checkpoint) + "\n", "utf8");
    const digest = createHash("sha256").update(bytes).digest("hex");
    await writeFile(path, bytes);
    await writeFile(path + ".sha256", `${digest}  ${bytes.byteLength}\n`);

    const loaded = await loadVerifiedTransformerArtifact(path);
    assert.equal(loaded.sha256, digest);
    assert.equal(loaded.bytes, bytes.byteLength);
    assert.deepEqual(loaded.model.probabilities([78, 101]), TransformerModel.fromCheckpoint(checkpoint).probabilities([78, 101]));

    await writeFile(path, Buffer.concat([bytes, Buffer.from("tampered")]));
    await assert.rejects(loadVerifiedTransformerArtifact(path), /size does not match|checksum verification failed/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("rejects malformed checksum sidecars", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-artifact-"));
  const path = join(directory, "model.json");
  try {
    const checkpoint = new TransformerModel({
      ...DEFAULT_TRANSFORMER_CONFIG,
      contextSize: 2,
      embeddingSize: 4,
      headCount: 2,
      feedForwardSize: 4,
      seed: 17,
    }).toCheckpoint();
    await writeFile(path, JSON.stringify(checkpoint));
    await writeFile(path + ".sha256", "not-a-checksum");
    await assert.rejects(loadVerifiedTransformerArtifact(path), /sidecar is malformed/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
