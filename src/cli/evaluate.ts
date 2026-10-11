import { evaluateHeldOutFile } from "../evaluation/held-out.js";
import { recordEvaluation } from "../training/experiment-store.js";

const args = process.argv.slice(2);
const options = new Map<string, string>();
for (let i = 0; i < args.length; i += 1) {
  const key = args[i]!;
  if (!key.startsWith("--")) throw new Error(`Unexpected argument: ${key}`);
  const value = args[i + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
  options.set(key.slice(2), value);
  i += 1;
}

try {
  const report = await evaluateHeldOutFile(options.get("input") ?? "data/corpus.txt", Number(options.get("epochs") ?? "20"));
  const contextMetrics = report.evaluations.find((item) => item.modelFamily === "nexus-context-mlp");
  if (contextMetrics) {
    await recordEvaluation([{
      modelFamily: "nexus-context-mlp",
      metrics: { ...contextMetrics, split: report.split, benchmark: "ordered-80-10-10-byte-split" },
      notes: report.warning,
    }]);
  }
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
