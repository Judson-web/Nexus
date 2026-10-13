export type TrainingDataOrder = "manifest" | "length-cyclic";

export interface OrderableTrainingExample {
  readonly id: string;
  readonly text: string;
}

/**
 * Deterministic length-cyclic ordering: sort by tokenized length, divide into
 * four similarly sized complexity bands, then interleave the bands. This keeps
 * a range of sentence lengths in each local training segment instead of
 * presenting all short examples before all long examples. It is a Nexus
 * heuristic inspired by research on curriculum continuity/local diversity,
 * not a reproduction of STR or SAW.
 */
export function orderTrainingExamples<T extends OrderableTrainingExample>(
  examples: readonly T[],
  tokenLength: (text: string) => number,
  strategy: TrainingDataOrder,
): T[] {
  if (strategy !== "manifest" && strategy !== "length-cyclic") {
    throw new RangeError("data order must be manifest or length-cyclic.");
  }
  if (strategy === "manifest" || examples.length < 2) return [...examples];

  const sorted = examples.map((example, index) => ({
    example,
    index,
    length: tokenLength(example.text),
  }));
  if (sorted.some((entry) => !Number.isSafeInteger(entry.length) || entry.length < 0)) {
    throw new RangeError("tokenLength must return non-negative safe integers.");
  }
  sorted.sort((a, b) => a.length - b.length || a.example.id.localeCompare(b.example.id) || a.index - b.index);

  const bandCount = Math.min(4, sorted.length);
  const bands = Array.from({ length: bandCount }, () => [] as typeof sorted);
  sorted.forEach((entry, index) => {
    const band = Math.min(bandCount - 1, Math.floor(index * bandCount / sorted.length));
    bands[band]!.push(entry);
  });

  const ordered: T[] = [];
  let remaining = sorted.length;
  let cursor = 0;
  while (remaining > 0) {
    const bandIndex = cursor % bandCount;
    const entry = bands[bandIndex]!.shift();
    if (entry) {
      ordered.push(entry.example);
      remaining -= 1;
    }
    cursor += 1;
  }
  return ordered;
}
