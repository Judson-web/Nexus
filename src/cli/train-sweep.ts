import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

interface Candidate {
  readonly name: string;
  readonly seed: number;
  readonly contextSize: number;
  readonly embeddingSize: number;
  readonly heads: number;
  readonly layers: number;
  readonly feedForwardSize: number;
  readonly learningRate: number;
  readonly optimizer: "adamw" | "adam" | "sgd";
}
interface RunResult {
  readonly candidate: Candidate;
  readonly output: string;
  readonly exitCode: number;
  readonly durationMs: number;
  readonly validationNll?: number;
  readonly validationPerplexity?: number;
  readonly testNll?: number;
  readonly error?: string;
}

function parseArgs(args: string[]): Map<string, string> {
  const options = new Map<string, string>();
  const allowed = new Set(["manifest", "output-dir", "summary", "epochs", "workers", "seed", "max-minutes", "max-tokens-per-example", "target-stride", "weight-decay", "schedule"]);
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
    const name = key.slice(2);
    const value = args[i + 1];
    if (!allowed.has(name)) throw new Error(`Unknown option: ${key}`);
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
    if (options.has(name)) throw new Error(`Duplicate option: ${key}`);
    options.set(name, value);
    i += 1;
  }
  return options;
}
function intOption(options: Map<string, string>, name: string, fallback: number, min: number, max: number): number {
  const raw = options.get(name);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`--${name} must be an integer in [${min}, ${max}].`);
  return value;
}
function runCandidate(candidate: Candidate, options: Map<string, string>, manifest: string, output: string): Promise<RunResult> {
  return new Promise((resolveResult) => {
    const started = Date.now();
    const args = [
      "run", "train:transformer:manifest", "--",
      "--manifest", manifest, "--output", output,
      "--epochs", options.get("epochs") ?? "1",
      "--seed", String(candidate.seed),
      "--context-size", String(candidate.contextSize),
      "--embedding-size", String(candidate.embeddingSize),
      "--heads", String(candidate.heads),
      "--layer-count", String(candidate.layers),
      "--feed-forward-size", String(candidate.feedForwardSize),
      "--learning-rate", String(candidate.learningRate),
      "--optimizer", candidate.optimizer,
      "--schedule", options.get("schedule") ?? "cosine",
      "--weight-decay", options.get("weight-decay") ?? (candidate.optimizer === "adamw" ? "0.01" : "0"),
      "--max-minutes", options.get("max-minutes") ?? "30",
      "--max-tokens-per-example", options.get("max-tokens-per-example") ?? "192",
      "--target-stride", options.get("target-stride") ?? "2",
    ];
    const child = spawn("npm", args, { stdio: ["ignore", "pipe", "pipe"], env: process.env });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk: string) => { stdout = (stdout + chunk).slice(-200_000); });
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => { stderr = (stderr + chunk).slice(-30_000); });
    child.on("error", (error) => resolveResult({
      candidate, output, exitCode: 127, durationMs: Date.now() - started, error: error.message,
    }));
    child.on("close", async (code) => {
      const base: RunResult = {
        candidate, output, exitCode: code ?? 1, durationMs: Date.now() - started,
        ...(code === 0 ? {} : { error: stderr.slice(-6000) || stdout.slice(-6000) || `Child process exited with code ${code}.` }),
      };
      if (code !== 0) return resolveResult(base);
      try {
        const provenance = JSON.parse(await readFile(`${output}.provenance.json`, "utf8")) as {
          heldOutEvaluation?: { validation?: { negativeLogLikelihood?: number; perplexity?: number }; test?: { negativeLogLikelihood?: number } };
        };
        const validation = provenance.heldOutEvaluation?.validation;
        const test = provenance.heldOutEvaluation?.test;
        if (!validation || typeof validation.negativeLogLikelihood !== "number" || !Number.isFinite(validation.negativeLogLikelihood) || typeof validation.perplexity !== "number" || !Number.isFinite(validation.perplexity)) {
          return resolveResult({ ...base, exitCode: 2, error: "Run completed but validation metrics were missing or invalid." });
        }
        resolveResult({
          ...base,
          validationNll: validation.negativeLogLikelihood,
          validationPerplexity: validation.perplexity,
          ...(typeof test?.negativeLogLikelihood === "number" && Number.isFinite(test.negativeLogLikelihood) ? { testNll: test.negativeLogLikelihood } : {}),
        });
      } catch (error) {
        resolveResult({ ...base, exitCode: 2, error: `Could not read provenance: ${error instanceof Error ? error.message : String(error)}` });
      }
    });
  });
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const manifest = resolve(options.get("manifest") ?? "artifacts/dataset-manifest.json");
  const outputDir = resolve(options.get("output-dir") ?? "artifacts/sweep");
  const summaryPath = resolve(options.get("summary") ?? `${outputDir}/sweep-summary.json`);
  const workers = intOption(options, "workers", 2, 1, 8);
  const seed = intOption(options, "seed", 17, -2_147_483_648, 2_147_483_647);
  intOption(options, "epochs", 1, 1, 1000);
  await readFile(manifest, "utf8");
  await mkdir(outputDir, { recursive: true });

  // Small, deliberately diverse candidates. Increase model sizes only after the
  // local CPU profiler confirms that the available memory/compute can sustain it.
  const candidates: Candidate[] = [
    { name: "adam-context8", seed, contextSize: 8, embeddingSize: 8, heads: 2, layers: 2, feedForwardSize: 16, learningRate: 0.0003, optimizer: "adamw" },
    { name: "adam-context32", seed: seed + 1, contextSize: 32, embeddingSize: 8, heads: 2, layers: 2, feedForwardSize: 16, learningRate: 0.0003, optimizer: "adamw" },
    { name: "adam-wider", seed: seed + 2, contextSize: 12, embeddingSize: 12, heads: 3, layers: 2, feedForwardSize: 24, learningRate: 0.0002, optimizer: "adam" },
    { name: "sgd-control", seed: seed + 3, contextSize: 8, embeddingSize: 8, heads: 2, layers: 1, feedForwardSize: 16, learningRate: 0.0001, optimizer: "sgd" },
  ];
  const results: RunResult[] = new Array(candidates.length);
  let nextIndex = 0;
  async function worker(): Promise<void> {
    while (true) {
      const index = nextIndex++;
      if (index >= candidates.length) return;
      const candidate = candidates[index]!;
      const output = resolve(outputDir, `${candidate.name}.json`);
      console.error(`[sweep] starting ${candidate.name} (${index + 1}/${candidates.length}); concurrency=${workers}`);
      results[index] = await runCandidate(candidate, options, manifest, output);
      const result = results[index]!;
      console.error(`[sweep] finished ${candidate.name}; exit=${result.exitCode}; validation_nll=${result.validationNll ?? "unavailable"}; duration_ms=${result.durationMs}`);
    }
  }
  await Promise.all(Array.from({ length: Math.min(workers, candidates.length) }, () => worker()));
  const ranked = [...results].filter((result) => result.exitCode === 0 && Number.isFinite(result.validationNll))
    .sort((a, b) => a.validationNll! - b.validationNll!);
  const summary = {
    format: "nexus-training-sweep",
    version: 1,
    createdAt: new Date().toISOString(),
    manifest,
    selectionMetric: "validation negative log-likelihood (lower is better)",
    testUsedForSelection: false,
    workers,
    epochs: options.get("epochs") ?? "1",
    winner: ranked[0] ? { name: ranked[0].candidate.name, output: ranked[0].output, validationNll: ranked[0].validationNll } : null,
    runs: results,
    warning: "A sweep ranks only these experimental candidates. It does not promote a checkpoint or establish general language ability. Test metrics are reported only for diagnosis and never used for ranking.",
  };
  await mkdir(dirname(summaryPath), { recursive: true });
  await writeFile(summaryPath, JSON.stringify(summary, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({ summaryPath, successfulRuns: ranked.length, failedRuns: results.length - ranked.length, winner: summary.winner }, null, 2));
  if (ranked.length === 0) process.exitCode = 1;
}
main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
