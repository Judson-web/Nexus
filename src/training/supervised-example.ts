import { EOS_TOKEN_ID } from "../tokenizer/byte-tokenizer.js";
import type { BytePairTokenizer } from "../tokenizer/byte-pair-tokenizer.js";

export interface EncodedTrainingExample {
  readonly tokens: number[];
  /** First target token. For user/assistant pairs this is the first assistant token. */
  readonly lossStartIndex: number;
  readonly truncated: boolean;
  readonly lossPolicy: "assistant-only-for-user-assistant-pairs-v1" | "full-sequence-next-token-v1";
}

const PAIR_PREFIX = "User: ";
const ASSISTANT_SEPARATOR = " Assistant: ";

/**
 * Encode a conversation instruction and response at a distinct token boundary.
 * Prompt tokens are context only; assistant tokens and EOS contribute to loss.
 * Plain-text rows keep their full-sequence next-token objective.
 */
export function encodeTrainingExample(
  text: string,
  tokenizer: BytePairTokenizer,
  contextSize: number,
  maxTokens?: number,
): EncodedTrainingExample {
  if (!Number.isSafeInteger(contextSize) || contextSize < 1) {
    throw new RangeError("contextSize must be a positive integer.");
  }
  if (maxTokens !== undefined && (!Number.isSafeInteger(maxTokens) || maxTokens < 2)) {
    throw new RangeError("maxTokens must be an integer of at least 2 so the sample can contain a context and a target.");
  }

  const delimiterIndex = text.startsWith(PAIR_PREFIX) ? text.lastIndexOf(ASSISTANT_SEPARATOR) : -1;
  const responseStart = delimiterIndex < 0 ? -1 : delimiterIndex + ASSISTANT_SEPARATOR.length;
  const response = responseStart < 0 ? "" : text.slice(responseStart).trim();
  if (delimiterIndex >= 0 && response.length > 0) {
    const prefixText = text.slice(0, responseStart);
    const promptLimit = maxTokens === undefined ? contextSize : Math.min(contextSize, maxTokens - 1);
    const promptTokens = tokenizer.encode(prefixText, false).slice(-promptLimit);
    const responseTokens = tokenizer.encode(response, false);
    const tailBudget = maxTokens === undefined ? Number.POSITIVE_INFINITY : maxTokens - promptTokens.length;
    if (tailBudget < 1) throw new Error("maxTokens leaves no room for an assistant response.");
    const truncated = responseTokens.length + 1 > tailBudget;
    const retainedResponse = truncated ? responseTokens.slice(0, tailBudget) : responseTokens;
    const tokens = [...promptTokens, ...retainedResponse];
    if (!truncated) tokens.push(EOS_TOKEN_ID);
    if (tokens.length < 2 || promptTokens.length >= tokens.length) {
      throw new Error("Conversation example contains no assistant-response targets after encoding.");
    }
    return { tokens, lossStartIndex: promptTokens.length, truncated, lossPolicy: "assistant-only-for-user-assistant-pairs-v1" };
  }

  const rawTokens = tokenizer.encode(text, false);
  const truncated = maxTokens !== undefined && rawTokens.length + 1 > maxTokens;
  const tokens = truncated
    ? [...rawTokens.slice(0, maxTokens - 1), EOS_TOKEN_ID]
    : [...rawTokens, EOS_TOKEN_ID];
  if (tokens.length < 2) throw new Error("Training example contains fewer than two tokens.");
  return { tokens, lossStartIndex: 1, truncated, lossPolicy: "full-sequence-next-token-v1" };
}
