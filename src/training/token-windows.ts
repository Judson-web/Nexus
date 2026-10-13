export interface TokenWindow {
  readonly tokens: number[];
  /** First token index eligible as a next-token target; preceding tokens are context only. */
  readonly targetStartIndex: number;
  readonly sourceStart: number;
}

/**
 * Split a tokenized document into windows with overlapping context while
 * assigning each token after the first to exactly one window's loss targets.
 * Overlap tokens are retained as context but are not trained twice.
 */
export function makeOverlappingTokenWindows(
  tokens: readonly number[],
  maxTokensPerWindow: number,
  contextOverlap: number,
): TokenWindow[] {
  if (!Number.isSafeInteger(maxTokensPerWindow) || maxTokensPerWindow < 2) {
    throw new RangeError("maxTokensPerWindow must be an integer >= 2.");
  }
  if (!Number.isSafeInteger(contextOverlap) || contextOverlap < 1 || contextOverlap >= maxTokensPerWindow) {
    throw new RangeError("contextOverlap must be an integer in [1, maxTokensPerWindow).");
  }
  if (tokens.some((token) => !Number.isSafeInteger(token) || token < 0)) {
    throw new RangeError("tokens must contain non-negative integer token IDs.");
  }
  if (tokens.length < 2) return [];
  const windows: TokenWindow[] = [];
  let sourceStart = 0;
  while (sourceStart < tokens.length - 1) {
    const sourceEnd = Math.min(tokens.length, sourceStart + maxTokensPerWindow);
    const windowTokens = tokens.slice(sourceStart, sourceEnd);
    const targetStartIndex = sourceStart === 0 ? 1 : contextOverlap + 1;
    if (targetStartIndex < windowTokens.length) {
      windows.push({ tokens: windowTokens, targetStartIndex, sourceStart });
    }
    if (sourceEnd === tokens.length) break;
    sourceStart = sourceEnd - contextOverlap;
  }
  return windows;
}
