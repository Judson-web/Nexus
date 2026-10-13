import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

type Split = "train" | "validation" | "test";

interface Work {
  readonly id: number;
  readonly title: string;
  readonly author: string;
  readonly year: number;
}

const WORKS: readonly Work[] = [
  { id: 11, title: "Alice's Adventures in Wonderland", author: "Lewis Carroll", year: 1865 },
  { id: 16, title: "Peter Pan", author: "J. M. Barrie", year: 1911 },
  { id: 35, title: "The Time Machine", author: "H. G. Wells", year: 1895 },
  { id: 36, title: "The War of the Worlds", author: "H. G. Wells", year: 1898 },
  { id: 46, title: "A Christmas Carol", author: "Charles Dickens", year: 1843 },
  { id: 55, title: "The Wonderful Wizard of Oz", author: "L. Frank Baum", year: 1900 },
  { id: 74, title: "The Adventures of Tom Sawyer", author: "Mark Twain", year: 1876 },
  { id: 76, title: "Adventures of Huckleberry Finn", author: "Mark Twain", year: 1884 },
  { id: 84, title: "Frankenstein", author: "Mary Shelley", year: 1818 },
  { id: 98, title: "A Tale of Two Cities", author: "Charles Dickens", year: 1859 },
  { id: 100, title: "The Complete Works of William Shakespeare", author: "William Shakespeare", year: 1623 },
  { id: 120, title: "Robinson Crusoe", author: "Daniel Defoe", year: 1719 },
  { id: 174, title: "Little Women", author: "Louisa May Alcott", year: 1868 },
  { id: 215, title: "The Call of the Wild", author: "Jack London", year: 1903 },
  { id: 236, title: "The Jungle Book", author: "Rudyard Kipling", year: 1894 },
  { id: 345, title: "Dracula", author: "Bram Stoker", year: 1897 },
  { id: 408, title: "Treasure Island", author: "Robert Louis Stevenson", year: 1883 },
  { id: 768, title: "Wuthering Heights", author: "Emily Brontë", year: 1847 },
  { id: 1184, title: "The Count of Monte Cristo", author: "Alexandre Dumas", year: 1844 },
  { id: 1260, title: "Jane Eyre", author: "Charlotte Brontë", year: 1847 },
  { id: 1342, title: "Pride and Prejudice", author: "Jane Austen", year: 1813 },
  { id: 1400, title: "Great Expectations", author: "Charles Dickens", year: 1861 },
  { id: 1513, title: "Romeo and Juliet", author: "William Shakespeare", year: 1597 },
  { id: 1661, title: "The Adventures of Sherlock Holmes", author: "Arthur Conan Doyle", year: 1892 },
  { id: 2591, title: "Grimms' Fairy Tales", author: "Jacob and Wilhelm Grimm", year: 1812 },
  { id: 2701, title: "Moby Dick", author: "Herman Melville", year: 1851 },
  { id: 5740, title: "The Secret Garden", author: "Frances Hodgson Burnett", year: 1911 },
] as const;

function parseArgs(args: string[]): Map<string, string> {
  const result = new Map<string, string>();
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i]!;
    const value = args[i + 1];
    if (!key.startsWith("--") || !value || value.startsWith("--")) throw new Error(`Invalid argument near ${key}`);
    if (result.has(key.slice(2))) throw new Error(`Duplicate option ${key}`);
    if (!["output", "paragraphs-per-book", "timeout-ms"].includes(key.slice(2))) throw new Error(`Unknown option ${key}`);
    result.set(key.slice(2), value);
    i += 1;
  }
  return result;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function download(url: string, timeoutMs: number): Promise<string> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": "Nexus-Licensed-Corpus-Acquisition/1.0 (research; public-domain works)", accept: "text/plain" },
        signal: AbortSignal.timeout(timeoutMs),
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
      const text = await response.text();
      if (text.length < 20_000) throw new Error(`Unexpectedly short Gutenberg response for ${url} (${text.length} chars)`);
      return text.replace(/^\uFEFF/, "").replaceAll("\r\n", "\n");
    } catch (error) {
      lastError = error;
      if (attempt < 4) await delay(500 * 2 ** (attempt - 1));
    }
  }
  throw new Error(`Download failed after retries: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

function extractBody(raw: string, id: number): string {
  const start = raw.search(/\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK/i);
  const end = raw.search(/\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK/i);
  if (start < 0 || end <= start) throw new Error(`Gutenberg start/end markers missing for ebook ${id}; refusing to train on unverified boilerplate.`);
  const startLine = raw.indexOf("\n", start);
  const body = raw.slice(startLine < 0 ? start : startLine + 1, end).trim();
  if (body.length < 10_000) throw new Error(`Extracted body for ebook ${id} is unexpectedly small.`);
  return body;
}

function splitForWork(id: number): Split {
  // Deterministic whole-work splits prevent passages from the same book leaking across evaluation sets.
  const bucket = Number.parseInt(sha256(`nexus-pd-work-v1:${id}`).slice(0, 8), 16) % 100;
  return bucket < 80 ? "train" : bucket < 90 ? "validation" : "test";
}

function paragraphs(body: string, limit: number): string[] {
  const candidates = body.split(/\n\s*\n+/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter((p) => p.length >= 100 && p.length <= 5000)
    .filter((p) => !/^chapter\s+[\divxlc]+\b/i.test(p) && !/^illustration:/i.test(p));
  // Deterministically sample across the entire work, rather than taking only opening chapters.
  if (candidates.length <= limit) return candidates;
  const chosen: string[] = [];
  const used = new Set<number>();
  for (let i = 0; i < limit; i += 1) {
    const index = Math.floor(i * (candidates.length - 1) / (limit - 1));
    if (!used.has(index)) {
      used.add(index);
      chosen.push(candidates[index]!);
    }
  }
  return chosen;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const output = options.get("output") ?? "artifacts/public-domain-books";
  const paragraphsPerBook = Number(options.get("paragraphs-per-book") ?? "120");
  const timeoutMs = Number(options.get("timeout-ms") ?? "90000");
  if (!Number.isSafeInteger(paragraphsPerBook) || paragraphsPerBook < 20 || paragraphsPerBook > 1000) {
    throw new RangeError("--paragraphs-per-book must be an integer from 20 to 1000.");
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 120000) throw new RangeError("--timeout-ms must be 1000..120000.");

  const records: { id: string; text: string; split: Split; workId: number; textSha256: string }[] = [];
  const works: { id: number; title: string; author: string; year: number; url: string; split: Split; rawSha256: string; bodySha256: string; acceptedParagraphs: number; bodyCharacters: number }[] = [];
  for (const work of WORKS) {
    const url = `https://www.gutenberg.org/cache/epub/${work.id}/pg${work.id}.txt.utf8`;
    console.log(`[corpus] downloading ${work.id}: ${work.title}`);
    const raw = await download(url, timeoutMs);
    const body = extractBody(raw, work.id);
    const split = splitForWork(work.id);
    const selected = paragraphs(body, paragraphsPerBook);
    if (selected.length < 20) throw new Error(`Only ${selected.length} eligible paragraphs found in ${work.title}; refusing a thin corpus.`);
    works.push({ ...work, url, split, rawSha256: sha256(raw), bodySha256: sha256(body), acceptedParagraphs: selected.length, bodyCharacters: body.length });
    for (let i = 0; i < selected.length; i += 1) {
      const text = selected[i]!;
      records.push({ id: `gutenberg-${work.id}-p${String(i + 1).padStart(4, "0")}`, text, split, workId: work.id, textSha256: sha256(text) });
    }
  }
  const splits: Record<Split, string[]> = { train: [], validation: [], test: [] };
  for (const record of records) splits[record.split].push(record.text);
  for (const split of ["train", "validation", "test"] as const) {
    if (splits[split].length < 100) throw new Error(`Only ${splits[split].length} paragraphs in ${split}; check whole-work split balance before training.`);
  }
  const jsonl = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  const manifest = {
    format: "nexus-public-domain-book-corpus",
    formatVersion: 1,
    source: "Project Gutenberg plain-text ebooks",
    sourceUrl: "https://www.gutenberg.org/",
    licensePolicy: "Curated works published before 1931; intended as US public-domain texts as of 2026. Verify local legal status before redistribution outside the United States. The downloaded Gutenberg wrapper and trademark are not included in model training.",
    splitPolicy: "Stable hash of whole work ID; all paragraphs from each work remain in exactly one split. No work-level split leakage.",
    retrieval: { retrievedAt: new Date().toISOString(), workCount: works.length, paragraphsPerWorkLimit: paragraphsPerBook, acceptedParagraphs: records.length },
    splits: Object.fromEntries(Object.entries(splits).map(([key, values]) => [key, values.length])),
    totalCharacters: records.reduce((sum, record) => sum + record.text.length, 0),
    works,
    records: records.map(({ id, textSha256, split, workId }) => ({ id, textSha256, split, workId })),
    recordsSha256: sha256(jsonl),
    warning: "Research corpus assembled from public-domain-in-the-US literary works; some books contain archaic language and do not provide modern technical or conversational coverage. No text from validation/test books is used for gradient updates.",
  };

  await mkdir(dirname(output), { recursive: true });
  const staging = await mkdtemp(`${output}.staging-`);
  try {
    for (const split of ["train", "validation", "test"] as const) {
      await writeFile(join(staging, `${split}.txt`), splits[split].join("\n") + "\n", "utf8");
    }
    await writeFile(join(staging, "records.jsonl"), jsonl, "utf8");
    await writeFile(join(staging, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
    await rename(staging, output);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({ output, workCount: works.length, splits: manifest.splits, totalCharacters: manifest.totalCharacters, recordsSha256: manifest.recordsSha256 }, null, 2));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
