import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { validateDatasetManifest } from "../evaluation/manifest.js";
import { DEFAULT_TRANSFORMER_CONFIG, TransformerModel } from "../model/transformer-model.js";
import { EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";
import { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const name = key.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    if (!["manifest", "output", "epochs", "seed", "learning-rate", "context-size", "embedding-size", "heads", "feed-forward-size", "optimizer", "max-minutes"].includes(name)) {
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
    const sequence = [...tokenizer.encode(example.text, false), EOS_TOKEN_ID];
    for (let i = 0; i < sequence.length; i += 1) {
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
  if (maxMinutes <= 0 || maxMinutes > 300) throw new RangeError("max-minutes must be in (0, 300].");
  const deadline = Date.now() + maxMinutes * 60_000;
  const optimizer = options.get("optimizer") ?? "adam";
  if (optimizer !== "adam" && optimizer !== "sgd") throw new RangeError("optimizer must be adam or sgd.");
  if (!Number.isSafeInteger(epochs) || epochs < 1 || epochs > 1000) throw new RangeError("epochs must be an integer between 1 and 1000.");
  if (!Number.isSafeInteger(seed)) throw new RangeError("seed must be a safe integer.");
  if (learningRate <= 0 || learningRate > 0.02) throw new RangeError("learning-rate must be in (0, 0.02].");

  const rawManifest = await readFile(manifestPath, "utf8");
  const manifest = validateDatasetManifest(JSON.parse(rawManifest) as unknown);
  const trainingExamples = manifest.examples.filter((example) => example.split === "train");
  if (trainingExamples.length === 0) throw new Error("Dataset manifest has no training examples.");

  const config = {
    ...DEFAULT_TRANSFORMER_CONFIG,
    contextSize: numeric(options, "context-size", 32),
    embeddingSize: numeric(options, "embedding-size", 4),
    headCount: numeric(options, "heads", 2),
    feedForwardSize: numeric(options, "feed-forward-size", 8),
    seed,
  };
  // Learn merges from training text only; held-out text never influences token IDs.
  const tokenizer = BytePairTokenizer.train(trainingExamples.map((example) => example.text), 255);
  const modelConfig = { ...config, vocabSize: tokenizer.vocabSize };
  const model = new TransformerModel(modelConfig);
  let weightedLoss = 0;
  let trainedTokens = 0;
  let lastGradientNorm = 0;
  let lastGradientScale = 1;

  // Each manifest line is a separate sequence. Only the train split is ever
  // passed to train(); validation and test text are provenance-only here.
  let exampleIndex = 0;
  const startedAt = Date.now();
  for (const example of trainingExamples) {
    exampleIndex += 1;
    if (Date.now() >= deadline) throw new Error(`Training exceeded the ${maxMinutes}-minute runtime limit before example ${exampleIndex}/${trainingExamples.length}.`);
    const tokens = [...tokenizer.encode(example.text, false), EOS_TOKEN_ID];
    if (tokens.length < 2) continue;
    console.error(`[train] example ${exampleIndex}/${trainingExamples.length}; targets=${tokens.length - 1}; elapsed=${Math.round((Date.now() - startedAt) / 1000)}s`);
    const metrics = model.train(tokens, learningRate, epochs, 1, optimizer, (progress) => {
      if (Date.now() >= deadline) throw new Error(`Training exceeded the ${maxMinutes}-minute runtime limit during example ${exampleIndex}/${trainingExamples.length}.`);
      console.error(`[train] example ${exampleIndex}/${trainingExamples.length} epoch ${progress.epoch}/${progress.epochs}: ${progress.targetsProcessed}/${progress.totalTargets} targets; elapsed=${Math.round((Date.now() - startedAt) / 1000)}s`);
    });
    weightedLoss += metrics.loss * metrics.examples;
    trainedTokens += metrics.examples;
    lastGradientNorm = metrics.gradientNorm;
    lastGradientScale = metrics.gradientScale;
  }
  if (trainedTokens === 0) throw new Error("Training examples contain no usable next-token targets.");

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
  const manifestSha256 = createHash("sha256").update(rawManifest, "utf8").digest("hex");
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
      seed, epochsPerExample: epochs, learningRate, optimizer,
      tokenizer: { type: "train-only byte-pair", merges: tokenizer.toJSON().merges.length, vocabSize: tokenizer.vocabSize },
      trainedExamples: trainingExamples.length,
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
  console.log(JSON.stringify({ outputPath, checkpointSha256, provenancePath: `${outputPath}.provenance.json`, ...provenance.training, heldOutEvaluation }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
