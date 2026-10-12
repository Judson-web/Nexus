import { createHash } from "node:crypto";

export const COMMON_PILE_ARXIV_ABSTRACTS = {
  dataset: "common-pile/arxiv_abstracts",
  revision: "828e35d1000f94579da8850f5f640c138279bdb5",
  config: "default",
  split: "train",
  license: "Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/",
  licenseUrl: "https://creativecommons.org/publicdomain/zero/1.0/",
  datasetUrl: "https://huggingface.co/datasets/common-pile/arxiv_abstracts",
} as const;

export interface CommonPileRow {
  readonly id: string;
  readonly text: string;
  readonly source: string;
  readonly created?: string;
  readonly added?: string;
  readonly metadata: {
    readonly license?: string;
    readonly full_text_license?: string | null;
    readonly authors?: string;
    readonly submitter?: string;
    readonly url?: string;
  };
}

export type SourceSplit = "train" | "validation" | "test";

export interface ReviewedAbstract {
  readonly id: string;
  readonly text: string;
  readonly textSha256: string;
  readonly split: SourceSplit;
  readonly sourceUrl: string;
  readonly authors: string;
  readonly license: string;
  readonly fullTextLicense: string | null;
}

export function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function normalizeAbstract(value: string): string {
  return value.normalize("NFC").replace(/\\s+/gu, " ").trim();
}

export function isExplicitCc0Row(row: CommonPileRow): boolean {
  return row.metadata.license?.trim() === COMMON_PILE_ARXIV_ABSTRACTS.license
    && /^https?:\\/\\/(?:www\\.)?arxiv\\.org\\/abs\\/[a-zA-Z0-9.\\-/]+$/.test(row.metadata.url ?? "")
    && row.id.trim().length > 0
    && row.source === "arxiv-abstracts"
    && normalizeAbstract(row.text).length >= 80;
}

/** Assigns an entire source record to exactly one stable split using its immutable source ID. */
export function splitForSourceId(id: string): SourceSplit {
  if (!id.trim()) throw new Error("Source record ID must not be empty.");
  const bucket = Number.parseInt(sha256("nexus-common-pile-v1:" + id).slice(0, 8), 16) % 100;
  return bucket < 90 ? "train" : bucket < 95 ? "validation" : "test";
}

export function reviewRows(rows: readonly CommonPileRow[], targetDocuments: number): {
  readonly accepted: readonly ReviewedAbstract[];
  readonly rejectedLicenseOrMetadata: number;
  readonly duplicateDocuments: number;
} {
  if (!Number.isSafeInteger(targetDocuments) || targetDocuments < 3 || targetDocuments > 5000) {
    throw new RangeError("targetDocuments must be an integer between 3 and 5000.");
  }
  const seenIds = new Set<string>();
  const seenText = new Set<string>();
  const accepted: ReviewedAbstract[] = [];
  let rejectedLicenseOrMetadata = 0;
  let duplicateDocuments = 0;

  for (const row of rows) {
    if (!isExplicitCc0Row(row)) {
      rejectedLicenseOrMetadata += 1;
      continue;
    }
    const text = normalizeAbstract(row.text);
    const digest = sha256(text);
    if (seenIds.has(row.id) || seenText.has(digest)) {
      duplicateDocuments += 1;
      continue;
    }
    seenIds.add(row.id);
    seenText.add(digest);
    accepted.push({
      id: row.id,
      text,
      textSha256: digest,
      split: splitForSourceId(row.id),
      sourceUrl: row.metadata.url!,
      authors: row.metadata.authors ?? "",
      license: row.metadata.license!,
      fullTextLicense: row.metadata.full_text_license ?? null,
    });
    if (accepted.length >= targetDocuments) break;
  }

  const splitCounts = { train: 0, validation: 0, test: 0 };
  for (const item of accepted) splitCounts[item.split] += 1;
  if (Object.values(splitCounts).some((count) => count === 0)) {
    throw new Error("Selected sample does not populate all three document-level splits; increase --documents or scan more rows.");
  }
  return { accepted, rejectedLicenseOrMetadata, duplicateDocuments };
}
