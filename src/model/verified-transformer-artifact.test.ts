import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "./transformer-model.js";
import { loadVerifiedTransformerArtifact } from "./verified-transformer-artifact.js";

async function writeArtifact(path: string, checkpoint: unknown): Promise<string> {
  const bytes = Buffer.from(JSON.stringify(checkpoint) + "\n", "utf8");
  const digest = createHash("sha256").update(bytes).digest("hex");
  await writeFile(path, bytes);
  await writeFile(path + ".sha256", `${digest}  ${bytes.byteLength}\n`);
  return digest;
}

function tinyModel(vocabSize = 257): TransformerModel {
  return new TransformerModel({
    ...DEFAULT_TRANSFORMER_CONFIG,
    vocabSize,
    contextSize: 2,
    embeddingSize: 4,
    headCount: 2,
    feedForwardSize: 4,
    seed: 17,
  });
}

test("loads a Transformer checkpoint only when its checksum and byte count match", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-artifact-"));
  const path = join(directory, "model.json");
  try {
    const checkpoint = tinyModel().toCheckpoint();
    const digest = await writeArtifact(path, checkpoint);
    const loaded = await loadVerifiedTransformerArtifact(path);
    assert.equal(loaded.sha256, digest);
    assert.equal(loaded.bytes, Buffer.byteLength(JSON.stringify(checkpoint) + "\n", "utf8"));
    assert.deepEqual(loaded.model.probabilities([78, 101]), tinyModelFrom(checkpoint).probabilities([78, 101]));
    assert.equal(loaded.tokenizer.constructor.name, "ByteTokenizer");

    await writeFile(path, Buffer.concat([Buffer.from(JSON.stringify(checkpoint) + "\n"), Buffer.from("tampered")]));
    await assert.rejects(loadVerifiedTransformerArtifact(path), /size does not match|checksum verification failed/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

function tinyModelFrom(checkpoint: ReturnType<TransformerModel["toCheckpoint"]>): TransformerModel {
  return TransformerModel.fromCheckpoint(checkpoint);
}

test("loads the embedded BPE tokenizer and rejects a vocabulary mismatch", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-bpe-artifact-"));
  const path = join(directory, "model.json");
  try {
    const tokenizer = BytePairTokenizer.train(["Sanskrit संस्कृतम् English", "Sanskrit English"], 12);
    const checkpoint = { ...tinyModel(tokenizer.vocabSize).toCheckpoint(), tokenizer: tokenizer.toJSON() };
    const digest = await writeArtifact(path, checkpoint);
    const loaded = await loadVerifiedTransformerArtifact(path);
    assert.equal(loaded.sha256, digest);
    assert.equal(loaded.tokenizer.constructor.name, "BytePairTokenizer");
    assert.deepEqual(loaded.tokenizer.encode("Sanskrit संस्कृतम्"), tokenizer.encode("Sanskrit संस्कृतम्"));
    assert.equal(loaded.tokenizer.decode(loaded.tokenizer.encode("Sanskrit संस्कृतम्")), "Sanskrit संस्कृतम्");

    const mismatched = { ...tinyModel(257).toCheckpoint(), tokenizer: tokenizer.toJSON() };
    await writeArtifact(path, mismatched);
    await assert.rejects(loadVerifiedTransformerArtifact(path), /vocabulary does not match/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("rejects malformed checksum sidecars", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-artifact-"));
  const path = join(directory, "model.json");
  try {
    await writeFile(path, JSON.stringify(tinyModel().toCheckpoint()));
    await writeFile(path + ".sha256", "not-a-checksum");
    await assert.rejects(loadVerifiedTransformerArtifact(path), /sidecar is malformed/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
