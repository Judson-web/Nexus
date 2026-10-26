import { NextResponse } from "next/server";
import { loadVerifiedTransformerArtifact, type TransformerTokenizer } from "../../../src/model/verified-transformer-artifact.js";
import type { TransformerModel, TransformerTrainingFormat } from "../../../src/model/transformer-model.js";
import { formatChatSftPrompt, NEXUS_CHAT_SFT_FORMAT } from "../../../src/chat/prompt-format.js";
import { readChatCompletion, resolveChatProvider, type ChatMessage, type ChatProviderConfig } from "../../../src/inference/openai-compatible.js";

export const runtime = "nodejs";
export const maxDuration = 30;

const MAX_BODY_BYTES = 20 * 1024;
const MAX_PROMPT_CHARS = 4_000;
const MAX_MESSAGES = 12;
const MAX_TOTAL_MESSAGE_CHARS = 16_000;

type LocalTransformer = {
  model: TransformerModel;
  tokenizer: TransformerTokenizer;
  sha256: string;
  trainingFormat: TransformerTrainingFormat;
};
let cachedTransformer: Promise<LocalTransformer> | undefined;

class RequestValidationError extends Error {}

function errorResponse(error: string, code: string, status: number) {
  return NextResponse.json({ error, code }, { status, headers: { "Cache-Control": "no-store" } });
}

async function readBoundedJson(request: Request): Promise<unknown> {
  const contentLength = request.headers.get("content-length");
  if (contentLength !== null) {
    const declaredLength = Number(contentLength);
    if (!Number.isSafeInteger(declaredLength) || declaredLength < 0 || declaredLength > MAX_BODY_BYTES) {
      throw new RangeError("Request body must be 20 KB or smaller.");
    }
  }
  if (!request.body) throw new SyntaxError("Request body is required.");
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > MAX_BODY_BYTES) {
        await reader.cancel("Request body too large.");
        throw new RangeError("Request body must be 20 KB or smaller.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as unknown;
}

function parseMessages(body: Record<string, unknown>): ChatMessage[] {
  if ("messages" in body) {
    if (!Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > MAX_MESSAGES) {
      throw new RequestValidationError("messages must contain between 1 and " + MAX_MESSAGES + " user/assistant messages.");
    }
    let totalChars = 0;
    const messages: ChatMessage[] = [];
    for (const item of body.messages) {
      if (!item || typeof item !== "object" || !("role" in item) || !("content" in item) ||
          (item.role !== "user" && item.role !== "assistant") || typeof item.content !== "string") {
        throw new RequestValidationError("Each message must have a user or assistant role and text content.");
      }
      const content = item.content.trim();
      if (!content || content.length > MAX_PROMPT_CHARS) {
        throw new RequestValidationError("Each message must contain text and be " + MAX_PROMPT_CHARS + " characters or fewer.");
      }
      totalChars += content.length;
      if (totalChars > MAX_TOTAL_MESSAGE_CHARS) {
        throw new RequestValidationError("Conversation text must be " + MAX_TOTAL_MESSAGE_CHARS + " characters or fewer.");
      }
      messages.push({ role: item.role, content });
    }
    if (messages.at(-1)?.role !== "user") {
      throw new RequestValidationError("The last conversation message must be from the user.");
    }
    return messages;
  }

  if (typeof body.prompt !== "string") {
    throw new RequestValidationError("A text prompt or messages array is required.");
  }
  const prompt = body.prompt.trim();
  if (!prompt) throw new RequestValidationError("Enter a prompt first.");
  if (prompt.length > MAX_PROMPT_CHARS) {
    throw new RequestValidationError("Prompts must be " + MAX_PROMPT_CHARS + " characters or fewer.");
  }
  return [{ role: "user", content: prompt }];
}

function parseMaxTokens(value: unknown): number {
  const maxTokens = typeof value === "number" ? value : 256;
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) {
    throw new RequestValidationError("maxTokens must be an integer between 1 and 512.");
  }
  return maxTokens;
}

async function getConfiguredTransformer(checkpointPath: string): Promise<LocalTransformer> {
  if (!cachedTransformer) {
    cachedTransformer = loadVerifiedTransformerArtifact(checkpointPath).then(
      ({ model, tokenizer, sha256, trainingFormat }) => ({ model, tokenizer, sha256, trainingFormat }),
    ).catch((error: unknown) => {
      cachedTransformer = undefined;
      throw error;
    });
  }
  return cachedTransformer;
}

async function generateWithProvider(
  provider: ChatProviderConfig,
  messages: readonly ChatMessage[],
  maxTokens: number,
): Promise<NextResponse> {
  const systemMessage: ChatMessage = {
    role: "system",
    content: "You are Nexus, a helpful and clear AI assistant. Answer the user's actual question directly. Explain important reasoning briefly, use readable structure when useful, and never invent facts or claim to have performed actions you did not perform. If uncertain, say so.",
  };
  let response: Response;
  try {
    response = await fetch(provider.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(provider.apiKey ? { Authorization: "Bearer " + provider.apiKey } : {}),
      },
      body: JSON.stringify({
        model: provider.model,
        messages: [systemMessage, ...messages],
        max_tokens: maxTokens,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(26_000),
    });
  } catch (error) {
    if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) {
      return errorResponse("The model took too long to respond. Retry, or use a faster/smaller model provider.", "LLM_PROVIDER_TIMEOUT", 504);
    }
    return errorResponse("Nexus could not reach the configured model provider. Check its URL and service status.", "LLM_PROVIDER_UNREACHABLE", 502);
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      return errorResponse("The model provider rejected its credentials. Check HF_TOKEN permissions or the configured API key.", "LLM_PROVIDER_AUTH_FAILED", 502);
    }
    if (response.status === 402) {
      return errorResponse("The model provider has no available inference credits or billing budget.", "LLM_PROVIDER_BUDGET_EXHAUSTED", 402);
    }
    if (response.status === 429) {
      return errorResponse("The model provider is rate-limiting requests. Wait briefly or configure a provider with available capacity.", "LLM_PROVIDER_RATE_LIMITED", 429);
    }
    if (response.status === 404 || response.status === 400) {
      return errorResponse("The configured model or provider route was not found. Verify NEXUS_LLM_MODEL and provider availability.", "LLM_MODEL_UNAVAILABLE", 502);
    }
    return errorResponse("The configured model provider is temporarily unavailable.", "LLM_PROVIDER_FAILED", 503);
  }

  let payload: unknown;
  try {
    payload = await response.json() as unknown;
  } catch {
    return errorResponse("The model provider returned malformed JSON.", "LLM_PROVIDER_INVALID_RESPONSE", 502);
  }
  let text: string;
  try {
    text = readChatCompletion(payload);
  } catch {
    return errorResponse("The model provider did not return a usable assistant response.", "LLM_PROVIDER_EMPTY_RESPONSE", 502);
  }

  return NextResponse.json({
    text,
    model: provider.model,
    provider: provider.providerName,
    note: "Response generated by a pretrained open-weight model. Nexus's own from-scratch training checkpoint remains a separate research experiment.",
  }, { headers: { "Cache-Control": "no-store" } });
}

async function generateWithExperimentalTransformer(prompt: string, maxTokens: number): Promise<NextResponse> {
  const checkpointPath = process.env.NEXUS_TRANSFORMER_CHECKPOINT?.trim();
  if (process.env.NEXUS_ENABLE_EXPERIMENTAL_TRANSFORMER !== "true" || !checkpointPath) {
    return errorResponse(
      "Pretrained inference is not configured. Set HF_TOKEN for Hugging Face Inference Providers or NEXUS_LLM_BASE_URL for a self-hosted endpoint. The tiny from-scratch checkpoint is disabled in the public API unless explicitly enabled for research.",
      "LLM_PROVIDER_NOT_CONFIGURED",
      503,
    );
  }
  try {
    const transformer = await getConfiguredTransformer(checkpointPath);
    const isChatSft = transformer.trainingFormat === NEXUS_CHAT_SFT_FORMAT;
    const modelPrompt = isChatSft ? formatChatSftPrompt(prompt) : prompt;
    const promptTokens = transformer.tokenizer.encode(modelPrompt, false);
    const generated = transformer.model.generate(promptTokens, maxTokens);
    const outputTokens = isChatSft ? generated.slice(promptTokens.length) : generated;
    const text = transformer.tokenizer.decode(outputTokens).trim().slice(0, 12_000);
    if (!text) return errorResponse("The experimental checkpoint generated an empty continuation. It is not suitable for public inference.", "EXPERIMENTAL_MODEL_EMPTY_RESPONSE", 502);
    return NextResponse.json({
      text,
      model: "Nexus Transformer (experimental research checkpoint)",
      provider: "Nexus local experimental runtime",
      trainingFormat: transformer.trainingFormat,
      modelSha256: transformer.sha256,
      note: "Experimental from-scratch model. Output may be incoherent; this is not the pretrained assistant path.",
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return errorResponse("The experimental checkpoint could not be loaded or generate a response. Verify the file and SHA-256 sidecar.", "EXPERIMENTAL_MODEL_FAILED", 500);
  }
}

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await readBoundedJson(request);
  } catch (error) {
    if (error instanceof RangeError) return errorResponse(error.message, "REQUEST_TOO_LARGE", 413);
    return errorResponse("Send a valid JSON request.", "INVALID_JSON", 400);
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return errorResponse("The request must be a JSON object.", "INVALID_REQUEST", 400);
  }

  let messages: ChatMessage[];
  let maxTokens: number;
  try {
    messages = parseMessages(body as Record<string, unknown>);
    maxTokens = parseMaxTokens("maxTokens" in body ? body.maxTokens : undefined);
  } catch (error) {
    if (error instanceof RequestValidationError) return errorResponse(error.message, "INVALID_REQUEST", 400);
    return errorResponse("The request is invalid.", "INVALID_REQUEST", 400);
  }

  // A configured pretrained model is always preferred. Never silently serve
  // the tiny from-scratch checkpoint as though it were a capable chatbot.
  let provider: ChatProviderConfig | undefined;
  try {
    provider = resolveChatProvider();
  } catch (error) {
    return errorResponse(error instanceof Error ? error.message : "LLM provider configuration is invalid.", "LLM_PROVIDER_CONFIGURATION_INVALID", 500);
  }
  if (provider) return generateWithProvider(provider, messages, maxTokens);

  const lastUserMessage = [...messages].reverse().find((message) => message.role === "user")?.content ?? "";
  return generateWithExperimentalTransformer(lastUserMessage, maxTokens);
}
