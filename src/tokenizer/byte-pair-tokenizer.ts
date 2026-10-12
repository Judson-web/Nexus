import { EOS_TOKEN_ID } from "./byte-tokenizer.js";

export interface BytePairMerge {
  readonly left: number;
  readonly right: number;
}

export interface BytePairTokenizerData {
  readonly format: "nexus-byte-pair-tokenizer";
  readonly version: 1;
  readonly merges: readonly BytePairMerge[];
}

const BYTE_COUNT = 256;
const BASE_VOCAB_SIZE = 257;

function pairKey(left: number, right: number): string {
  return `${left},${right}`;
}

/**
 * Small deterministic byte-pair tokenizer. It starts with lossless UTF-8 bytes
 * and learns frequent adjacent pairs from training text only. Merge IDs are
 * stable and self-contained in the checkpoint; validation/test data never
 * influences the vocabulary.
 */
export class BytePairTokenizer {
  private readonly merges: BytePairMerge[];
  private readonly mergeRanks: Map<string, number>;
  private readonly expandedBytes = new Map<number, number[]>();

  public constructor(data?: BytePairTokenizerData) {
    if (data && (data.format !== "nexus-byte-pair-tokenizer" || data.version !== 1 || !Array.isArray(data.merges))) {
      throw new Error("Unsupported byte-pair tokenizer data.");
    }
    this.merges = (data?.merges ?? []).map(({ left, right }, index) => {
      const tokenId = BASE_VOCAB_SIZE + index;
      if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right) || left < 0 || right < 0 || left >= tokenId || right >= tokenId || left === EOS_TOKEN_ID || right === EOS_TOKEN_ID) {
        throw new Error("Invalid byte-pair merge table.");
      }
      return { left, right };
    });
    this.mergeRanks = new Map(this.merges.map((merge, index) => [pairKey(merge.left, merge.right), index]));
  }

  public static train(texts: readonly string[], maxMerges = 256, minFrequency = 2): BytePairTokenizer {
    if (!Number.isSafeInteger(maxMerges) || maxMerges < 0 || maxMerges > 4096) throw new RangeError("maxMerges must be between 0 and 4096.");
    if (!Number.isSafeInteger(minFrequency) || minFrequency < 2) throw new RangeError("minFrequency must be an integer of at least 2.");
    let sequences = texts.map((text) => Array.from(new TextEncoder().encode(text)));
    const merges: BytePairMerge[] = [];
    for (let step = 0; step < maxMerges; step += 1) {
      const counts = new Map<string, { left: number; right: number; count: number }>();
      for (const sequence of sequences) {
        for (let i = 0; i + 1 < sequence.length; i += 1) {
          const left = sequence[i]!;
          const right = sequence[i + 1]!;
          const key = pairKey(left, right);
          const entry = counts.get(key);
          if (entry) entry.count += 1;
          else counts.set(key, { left, right, count: 1 });
        }
      }
      let best: { left: number; right: number; count: number } | undefined;
      for (const entry of counts.values()) {
        if (!best || entry.count > best.count || (entry.count === best.count && (entry.left < best.left || (entry.left === best.left && entry.right < best.right)))) best = entry;
      }
      if (!best || best.count < minFrequency) break;
      const tokenId = BASE_VOCAB_SIZE + merges.length;
      merges.push({ left: best.left, right: best.right });
      sequences = sequences.map((sequence) => {
        const result: number[] = [];
        for (let i = 0; i < sequence.length; i += 1) {
          if (i + 1 < sequence.length && sequence[i] === best!.left && sequence[i + 1] === best!.right) {
            result.push(tokenId);
            i += 1;
          } else result.push(sequence[i]!);
        }
        return result;
      });
    }
    return new BytePairTokenizer({ format: "nexus-byte-pair-tokenizer", version: 1, merges });
  }

  public get vocabSize(): number {
    return BASE_VOCAB_SIZE + this.merges.length;
  }

  public encode(text: string, appendEos = true): number[] {
    let sequence = Array.from(new TextEncoder().encode(text));
    for (let rank = 0; rank < this.merges.length; rank += 1) {
      const merge = this.merges[rank]!;
      const tokenId = BASE_VOCAB_SIZE + rank;
      const next: number[] = [];
      for (let i = 0; i < sequence.length; i += 1) {
        if (i + 1 < sequence.length && sequence[i] === merge.left && sequence[i + 1] === merge.right) {
          next.push(tokenId);
          i += 1;
        } else next.push(sequence[i]!);
      }
      sequence = next;
    }
    if (appendEos) sequence.push(EOS_TOKEN_ID);
    return sequence;
  }

  public decode(tokens: readonly number[]): string {
    const bytes: number[] = [];
    for (const token of tokens) {
      if (!Number.isSafeInteger(token) || token < 0 || token >= this.vocabSize) throw new RangeError(`Invalid token ID: ${token}`);
      if (token === EOS_TOKEN_ID) break;
      bytes.push(...this.bytesFor(token));
    }
    const decoder = new TextDecoder("utf-8", { fatal: true });
    for (let length = bytes.length; length >= 0; length -= 1) {
      try { return decoder.decode(new Uint8Array(bytes.slice(0, length))); } catch { /* trim invalid UTF-8 suffix */ }
    }
    return "";
  }

  public toJSON(): BytePairTokenizerData {
    return { format: "nexus-byte-pair-tokenizer", version: 1, merges: this.merges.map((merge) => ({ ...merge })) };
  }

  private bytesFor(token: number): number[] {
    if (token < BYTE_COUNT) return [token];
    const cached = this.expandedBytes.get(token);
    if (cached) return cached;
    const merge = this.merges[token - BASE_VOCAB_SIZE];
    if (!merge) throw new RangeError(`Invalid merged token ID: ${token}`);
    const bytes = [...this.bytesFor(merge.left), ...this.bytesFor(merge.right)];
    this.expandedBytes.set(token, bytes);
    return bytes;
  }
}
