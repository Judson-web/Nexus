import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { validateDatasetManifest } from "../evaluation/manifest.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel, type LearningRateSchedule, type TransformerOptimizer, type TransformerCheckpoint } from "../model/transformer-model.js";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";
import { encodeTrainingExample } from "../training/supervised-example.js";

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const name = key.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    if (!["manifest", "output", "resume-from", "epochs", "seed", "learning-rate", "context-size", "embedding-size", "heads", "feed-forward-size", "layer-count", "optimizer", "schedule", "weight-decay", "max-minutes", "max-tokens-per-example", "target-stride"].includes(name)) {
      throw new Error(`Unknown option: ${key}`);
    }
    options.set(name, value);
    i += 1;
  }
  return options;
}

function numeric(options: Map<string, string>, key: string, fallback: number): number {
  const raw = options.get(key);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`--${key} must be numeric.`);
  return value;
}

async function writeAtomic(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp`;
  await writeFile(temporaryPath, content, { encoding: "utf8", flag: "w" });
  await rename(temporaryPath, path);
}

function scoreSplit(
  examples: readonly { readonly text: string }[],
  model: TransformerModel,
  tokenizer: BytePairTokenizer,
  contextSize: number,
) {
  let totalNegativeLogLikelihood = 0;
  let tokens = 0;
  for (const example of examples) {
    // Instruction prompts are context, not targets the assistant should reproduce.
    const encoded = encodeTrainingExample(example.text, tokenizer, contextSize);
    const sequence = encoded.tokens;
    for (let i = Math.max(1, encoded.lossStartIndex); i < sequence.length; i += 1) {
      const context = sequence.slice(Math.max(0, i - contextSize), i);
      const probabilities = model.probabilities(context);
      const probability = probabilities[sequence[i]!]!;
      if (!Number.isFinite(probability) || probability <= 0) {
        throw new Error("Model returned an invalid probability while scoring a held-out split.");
      }
      totalNegativeLogLikelihood -= Math.log(probability);
      tokens += 1;
    }
  }
  if (tokens === 0) throw new Error("Cannot score an empty dataset split.");
  const negativeLogLikelihood = totalNegativeLogLikelihood / tokens;
  return {
    examples: examples.length,
    tokens,
    negativeLogLikelihood,
    perplexity: Math.exp(Math.min(negativeLogLikelihood, 700)),
  };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const manifestPath = options.get("manifest") ?? "artifacts/dataset-manifest.json";
  const outputPath = options.get("output") ?? "artifacts/transformer-manifest-v1.json";
  const epochs = numeric(options, "epochs", 1);
  const seed = numeric(options, "seed", 17);
  const learningRate = numeric(options, "learning-rate", 0.0001);
  const maxMinutes = numeric(options, "max-minutes", 30);
  const maxTokensPerExample = numeric(options, "max-tokens-per-example", 256);
  const targetStride = numeric(options, "target-stride", 1);
  if (!Number.isSafeInteger(targetStride) || targetStride < 1 || targetStride > 32) throw new RangeError("target-stride must be an integer between 1 and 32.");
  if (!Number.isSafeInteger(maxTokensPerExample) || maxTokensPerExample < 16 || maxTokensPerExample > 4096) throw new RangeError("max-tokens-per-example must be an integer between 16 and 4096.");
  if (maxMinutes <= 0 || maxMinutes > 300) throw new RangeError("max-minutes must be in (0, 300].");
  const deadline = Date.now() + maxMinutes * 60_000;
  const optimizer = (options.get("optimizer") ?? "adamw") as TransformerOptimizer;
  if (optimizer !== "adamw" && optimizer !== "adam" && optimizer !== "sgd") throw new RangeError("optimizer must be adamw, adam, or sgd.");
  const schedule = (options.get("schedule") ?? "cosine") as LearningRateSchedule;
  if (schedule !== "constant" && schedule !== "cosine") throw new RangeError("schedule must be constant or cosine.");
  const weightDecay = numeric(options, "weight-decay", optimizer === "adamw" ? 0.01 : 0);
  if (!Number.isFinite(weightDecay) || weightDecay < 0 || weightDecay > 0.1) throw new RangeError("weight-decay must be in [0, 0.1].");
  if (optimizer !== "adamw" && weightDecay !== 0) throw new RangeError("weight-decay is supported only with adamw.");
  if (!Number.isSafeInteger(epochs) || epochs < 1 || epochs > 1000) throw new RangeError("epochs must be an integer between 1 and 1000.");
  if (!Number.isSafeInteger(seed)) throw new RangeError("seed must be a safe integer.");
  if (learningRate <= 0 || learningRate > 0.02) throw new RangeError("learning-rate must be in (0, 0.02].");

  const rawManifest = await readFile(manifestPath, "utf8");
  const manifestSha256 = createHash("sha256").update(rawManifest, "utf8").digest("hex");
  const manifest = validateDatasetManifest(JSON.parse(rawManifest) as unknown);
  const trainingExamples = manifest.examples.filter((example) => example.split === "train");
  if (trainingExamples.length === 0) throw new Error("Dataset manifest has no training examples.");

  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    contextSize: numeric(options, "context-size", 32),
    embeddingSize: numeric(options, "embedding-size", 4),
    headCount: numeric(options, "heads", 2),
    feedForwardSize: numeric(options, "feed-forward-size", 8),
    layerCount: numeric(options, "layer-count", 2),
    seed,
  };
  // A resume checkpoint pins both the tokenizer and exact dataset/training configuration.
  // Fresh runs learn merges from training text only; held-out text never influences token IDs.
  const resumeFrom = options.get("resume-from");
  let tokenizer: BytePairTokenizer;
  let modelConfig: typeof config & { vocabSize: number };
  let model: TransformerModel;
  const completedExampleIds = new Set<string>();
  let resumedTruncatedExamples = 0;
  const resumePath = resumeFrom ? `${resumeFrom}.resume.json` : `${outputPath}.resume.json`;
  if (resumeFrom) {
    const rawCheckpoint = JSON.parse(await readFile(resumeFrom, "utf8")) as TransformerCheckpoint & { tokenizer?: import("../tokenizer/byte-pair-tokenizer.js").BytePairTokenizerData };
    if (!rawCheckpoint.tokenizer) throw new Error("Resume checkpoint has no tokenizer data.");
    tokenizer = new BytePairTokenizer(rawCheckpoint.tokenizer);
    modelConfig = { ...config, vocabSize: tokenizer.vocabSize };
    if (JSON.stringify(rawCheckpoint.config) !== JSON.stringify(modelConfig)) throw new Error("Resume checkpoint model configuration does not match the requested configuration.");
    const resumeState = JSON.parse(await readFile(resumePath, "utf8")) as {
      format: string; manifestSha256: string; completedExampleIds: string[]; modelConfig: typeof modelConfig;
      epochs: number; learningRate: number; optimizer: TransformerOptimizer; schedule: LearningRateSchedule; weightDecay: number; targetStride: number; maxTokensPerExample: number; truncatedExamples?: number; lossPolicy?: string;
    };
    if (resumeState.format !== "nexus-training-resume" || resumeState.manifestSha256 !== manifestSha256) throw new Error("Resume checkpoint manifest hash mismatch.");
    if (resumeState.lossPolicy !== "assistant-only-for-user-assistant-pairs-v1") throw new Error("Resume checkpoint uses a different or legacy loss policy. Start a fresh run rather than mixing objectives.");
    if (JSON.stringify(resumeState.modelConfig) !== JSON.stringify(modelConfig) ||
        resumeState.epochs !== epochs || resumeState.learningRate !== learningRate ||
        resumeState.optimizer !== optimizer || resumeState.schedule !== schedule ||
        resumeState.weightDecay !== weightDecay || resumeState.targetStride !== targetStride ||
        resumeState.maxTokensPerExample !== maxTokensPerExample) {
      throw new Error("Resume settings differ from the saved run. Use the same model and training options.");
    }
    if (!Array.isArray(resumeState.completedExampleIds) || resumeState.completedExampleIds.some((id) => typeof id !== "string")) throw new Error("Resume progress metadata is malformed.");
    for (const id of resumeState.completedExampleIds) completedExampleIds.add(id);
    resumedTruncatedExamples = resumeState.truncatedExamples ?? 0;
    model = TransformerModel.fromCheckpoint(rawCheckpoint);
    if (completedExampleIds.size > trainingExamples.length || [...completedExampleIds].some((id) => !trainingExamples.some((example) => example.id === id))) {
      throw new Error("Resume progress contains unknown training example IDs.");
    }
    console.error(`[resume] restored ${completedExampleIds.size}/${trainingExamples.length} completed examples and optimizer state`);
  } else {
    tokenizer = BytePairTokenizer.train(trainingExamples.map((example) => example.text), 767);
    modelConfig = { ...config, vocabSize: tokenizer.vocabSize };
    model = new TransformerModel(modelConfig);
  }
  let weightedLoss = 0;
  let trainedTokens = 0;
  let truncatedExamples = resumedTruncatedExamples;
  let lastGradientNorm = 0;
  let lastGradientScale = 1;

  // Each manifest line is a separate sequence. Only the train split is ever
  // passed to train(); validation and test text are provenance-only here.
  let exampleIndex = 0;
  const startedAt = Date.now();
  for (const example of trainingExamples) {
    exampleIndex += 1;
    if (completedExampleIds.has(example.id)) continue;
    if (Date.now() >= deadline) throw new Error(`Training exceeded the ${maxMinutes}-minute runtime limit before example ${exampleIndex}/${trainingExamples.length}.`);
    const encoded = encodeTrainingExample(example.text, tokenizer, config.contextSize, maxTokensPerExample);
    const { tokens, lossStartIndex } = encoded;
    if (encoded.truncated) truncatedExamples += 1;
    if (tokens.length < 2 || lossStartIndex >= tokens.length) continue;
    const scheduleProgress = (exampleIndex - 1) / Math.max(1, trainingExamples.length - 1);
    const scheduledLearningRate = schedule === "cosine"
      ? learningRate * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * scheduleProgress)))
      : learningRate;
    console.error(`[train] example ${exampleIndex}/${trainingExamples.length}; response_targets=${tokens.length - lossStartIndex}; lr=${scheduledLearningRate.toExponential(3)}; elapsed=${Math.round((Date.now() - startedAt) / 1000)}s`);
    const metrics = model.train(tokens, scheduledLearningRate, epochs, 1, optimizer, (progress) => {
      if (Date.now() >= deadline) throw new Error(`Training exceeded the ${maxMinutes}-minute runtime limit during example ${exampleIndex}/${trainingExamples.length}.`);
      console.error(`[train] example ${exampleIndex}/${trainingExamples.length} epoch ${progress.epoch}/${progress.epochs}: ${progress.targetsProcessed}/${progress.totalTargets} targets; elapsed=${Math.round((Date.now() - startedAt) / 1000)}s`);
    }, targetStride, weightDecay, "constant", lossStartIndex);
    weightedLoss += metrics.loss * metrics.examples;
    trainedTokens += metrics.examples;
    lastGradientNorm = metrics.gradientNorm;
    lastGradientScale = metrics.gradientScale;
    completedExampleIds.add(example.id);
    const resumableCheckpoint = JSON.stringify({ ...model.toCheckpoint(), tokenizer: tokenizer.toJSON() }, null, 2) + "\n";
    const resumableBytes = Buffer.from(resumableCheckpoint, "utf8");
    const resumableSha256 = createHash("sha256").update(resumableBytes).digest("hex");
    await writeAtomic(outputPath, resumableCheckpoint);
    await writeAtomic(`${outputPath}.sha256`, `${resumableSha256}  ${resumableBytes.length}\n`);
    await writeAtomic(`${outputPath}.resume.json`, JSON.stringify({
      format: "nexus-training-resume", manifestSha256, completedExampleIds: [...completedExampleIds],
      modelConfig, epochs, learningRate, optimizer, schedule, weightDecay, targetStride, maxTokensPerExample,
      lossPolicy: "assistant-only-for-user-assistant-pairs-v1",
    }, null, 2) + "\n");
  }
  if (trainedTokens === 0 && completedExampleIds.size === 0) throw new Error("Training examples contain no usable next-token targets.");

  const finalTrainingEvaluation = scoreSplit(trainingExamples, model, tokenizer, config.contextSize);
  const validationExamples = manifest.examples.filter((example) => example.split === "validation");
  const testExamples = manifest.examples.filter((example) => example.split === "test");
  const heldOutEvaluation = {
    validation: scoreSplit(validationExamples, model, tokenizer, config.contextSize),
    test: scoreSplit(testExamples, model, tokenizer, config.contextSize),
    testUsedForModelSelection: false,
  };

  const checkpointContent = JSON.stringify({ ...model.toCheckpoint(), tokenizer: tokenizer.toJSON() }, null, 2) + "\n";
  const checkpointBytes = Buffer.from(checkpointContent, "utf8");
  const checkpointSha256 = createHash("sha256").update(checkpointBytes).digest("hex");
  const provenance = {
    format: "nexus-transformer-provenance",
    version: 1,
    checkpoint: { path: outputPath, sha256: checkpointSha256, bytes: checkpointBytes.length },
    dataset: {
      manifestPath,
      manifestSha256,
      sourceSha256: manifest.sourceSha256,
      preparedSha256: manifest.preparedSha256,
      splitSeed: manifest.splitSeed,
      splitCounts: manifest.splits,
      trainingExampleIds: trainingExamples.map((example) => example.id),
      validationAndTestUsedForTraining: false,
    },
    training: {
      seed, epochsPerExample: epochs, learningRate, optimizer, schedule, weightDecay,
      lossPolicy: "assistant-only-for-user-assistant-pairs-v1",
      tokenizer: { type: "train-only byte-pair", merges: tokenizer.toJSON().merges.length, vocabSize: tokenizer.vocabSize },
      trainedExamples: trainingExamples.length,
      maxTokensPerExample,
      targetStride,
      truncatedExamples,
      trainedNextTokenTargets: trainedTokens,
      meanFinalTrainingLoss: weightedLoss / trainedTokens,
      lastGradientNorm, lastGradientScale,
    },
    model: modelConfig,
    heldOutEvaluation,
    warning: "Research checkpoint only. Held-out metrics are diagnostics, not proof of general language ability. Do not tune hyperparameters against test results.",
  };
  await writeAtomic(outputPath, checkpointContent);
  await writeAtomic(`${outputPath}.sha256`, `${checkpointSha256}  ${checkpointBytes.length}\n`);
  await writeAtomic(`${outputPath}.provenance.json`, JSON.stringify(provenance, null, 2) + "\n");
  await writeAtomic(`${outputPath}.resume.json`, JSON.stringify({ format: "nexus-training-resume", manifestSha256, completedExampleIds: [...completedExampleIds], modelConfig, epochs, learningRate, optimizer, schedule, weightDecay, targetStride, maxTokensPerExample, truncatedExamples, lossPolicy: "assistant-only-for-user-assistant-pairs-v1" }, null, 2) + "\n");
  console.log(JSON.stringify({ outputPath, checkpointSha256, provenancePath: `${outputPath}.provenance.json`, ...provenance.training, heldOutEvaluation }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
