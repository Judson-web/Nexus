import { readFile } from "node:fs/promises";
import { compareTransformerProfileReports } from "../evaluation/compare-transformer-profiles.js";

const args = process.argv.slice(2);
const options = new Map<string, string>();
for (let index = 0; index < args.length; index += 1) {
  const key = args[index]!;
  if (key !== "--baseline" && key !== "--current") throw new Error(`Unknown option: ${key}`);
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${key}`);
  if (options.has(key)) throw new Error(`Duplicate option: ${key}`);
  options.set(key, value);
  index += 1;
}

try {
  const baselinePath = options.get("--baseline");
  const currentPath = options.get("--current");
  if (!baselinePath || !currentPath) {
    throw new Error("Usage: npm run compare:transformer -- --baseline baseline.json --current current.json");
  }
  const [baselineText, currentText] = await Promise.all([
    readFile(baselinePath, "utf8"),
    readFile(currentPath, "utf8"),
  ]);
  const result = compareTransformerProfileReports(JSON.parse(baselineText) as unknown, JSON.parse(currentText) as unknown);
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
