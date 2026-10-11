import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BigramModel } from "../model/bigram-model.js";
import { loadCheckpoint, saveCheckpoint } from "./checkpoint.js";

test("checkpoint saves, verifies, and restores model weights", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-checkpoint-"));
  const path = join(directory, "model.json");
  try {
    const model = new BigramModel({ vocabSize: 257, seed: 12, initializationScale: 0.01 });
    const manifest = await saveCheckpoint(path, model);
    assert.equal(manifest.bytes, (await readFile(path)).byteLength);
    const restored = await loadCheckpoint(path);
    assert.deepEqual(Array.from(restored.probabilities(88)), Array.from(model.probabilities(88)));
    await writeFile(path, "{broken");
    await assert.rejects(loadCheckpoint(path), /checksum mismatch/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
