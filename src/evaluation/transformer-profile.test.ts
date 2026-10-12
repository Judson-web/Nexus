import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { profileTransformerTraining } from "./transformer-profile.js";

test("Transformer profiler returns per-epoch loss and throughput with reproducibility metadata", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-profile-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, ("Nexus learns byte patterns. ").repeat(3), "utf8");
    const report = await profileTransformerTraining(path, {
      epochs: 2,
      repeats: 2,
      config: { contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 7 },
    });
    assert.equal(report.model.repeats, 2);
    assert.equal(report.model.optimizer, "sgd");
    assert.equal(report.runtime.nodeVersion, process.version);
    assert.ok(report.runtime.platform.length > 0);
    assert.ok(report.runtime.architecture.length > 0);
    assert.ok(report.runtime.logicalCpuCount > 0);
    assert.ok(report.runtime.cpuModel.length > 0);
    assert.equal(report.profile.runs.length, 2);
    assert.equal(report.profile.runs[0]!.epochs.length, 2);
    assert.ok(Number.isFinite(report.profile.medianFinalLoss));
    assert.ok(report.model.parameterCount > 0);
    assert.match(report.dataset.sourceSha256, /^[a-f0-9]{64}$/);
    for (const epoch of report.profile.runs[0]!.epochs) {
      assert.ok(Number.isFinite(epoch.loss));
      assert.ok(epoch.durationMs >= 0);
      assert.ok(Number.isFinite(epoch.examplesPerSecond));
      assert.ok(epoch.examplesPerSecond > 0);
    }
    assert.ok(report.profile.totalDurationMs >= 0);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Transformer profiler rejects invalid settings and empty input", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-profile-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, "valid input", "utf8");
    await assert.rejects(profileTransformerTraining(path, { epochs: 0 }), /epochs must be/i);
    await assert.rejects(profileTransformerTraining(path, { learningRate: 0 }), /learningRate must be/i);
    await assert.rejects(profileTransformerTraining(path, { repeats: 0 }), /repeats must be/i);
    await writeFile(path, "  ", "utf8");
    await assert.rejects(profileTransformerTraining(path), /at least two encoded tokens/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test("Transformer profiler reports target stride and sampled target throughput", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-profile-stride-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, "Nexus learns deterministic byte patterns. ".repeat(3), "utf8");
    const baseline = await profileTransformerTraining(path, {
      epochs: 1,
      repeats: 1,
      targetStride: 1,
      config: { contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 11 },
    });
    const strided = await profileTransformerTraining(path, {
      epochs: 1,
      repeats: 1,
      targetStride: 2,
      config: { contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 11 },
    });
    assert.equal(baseline.model.targetStride, 1);
    assert.equal(strided.model.targetStride, 2);
    assert.equal(baseline.profile.runs[0]!.epochs[0]!.targetsProcessed, baseline.dataset.tokenCount - 1);
    assert.ok(strided.profile.runs[0]!.epochs[0]!.targetsProcessed < baseline.profile.runs[0]!.epochs[0]!.targetsProcessed);
    assert.ok(Number.isFinite(strided.profile.meanTargetsPerSecond));
    assert.ok(strided.profile.meanTargetsPerSecond > 0);
    await assert.rejects(profileTransformerTraining(path, { targetStride: 0 }), /targetStride/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});


test("Transformer profiler supports controlled Adam comparisons", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nexus-profile-adam-"));
  const path = join(directory, "corpus.txt");
  try {
    await writeFile(path, "Nexus learns deterministic byte patterns. ".repeat(3), "utf8");
    const report = await profileTransformerTraining(path, {
      epochs: 1,
      repeats: 2,
      optimizer: "adam",
      learningRate: 0.0001,
      config: { contextSize: 2, embeddingSize: 4, headCount: 2, feedForwardSize: 4, seed: 13 },
    });
    assert.equal(report.model.optimizer, "adam");
    assert.ok(Number.isFinite(report.profile.medianFinalLoss));
    await assert.rejects(profileTransformerTraining(path, { optimizer: "muon" as "sgd" }), /optimizer must be sgd or adam/i);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
