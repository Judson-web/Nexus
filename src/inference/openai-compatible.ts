export type ChatRole = "system" | "user" | "assistant";

export interface ChatMessage {
  readonly role: ChatRole;
  readonly content: string;
}

export interface ChatProviderConfig {
  readonly kind: "huggingface" | "openai-compatible";
  readonly endpoint: string;
  readonly model: string;
  readonly apiKey?: string;
  readonly providerName: string;
}

const HUGGING_FACE_ROUTER = "https://router.huggingface.co/v1";
const DEFAULT_HUGGING_FACE_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M:fastest";

/**
 * Resolve provider configuration exclusively from server-side environment
 * variables. The request body can never choose a URL or supply credentials.
 * A custom OpenAI-compatible endpoint supports llama.cpp, vLLM, SGLang,
 * Ollama-compatible gateways, and other OpenAI-compatible servers.
 */
export function resolveChatProvider(
  env: Record<string, string | undefined> = process.env,
): ChatProviderConfig | undefined {
  const customBase = env.NEXUS_LLM_BASE_URL?.trim();
  const hfToken = env.HF_TOKEN?.trim();
  const rawBase = customBase || (hfToken ? HUGGING_FACE_ROUTER : "");
  if (!rawBase) return undefined;

  let parsed: URL;
  try {
    parsed = new URL(rawBase);
  } catch {
    throw new Error("NEXUS_LLM_BASE_URL must be an absolute HTTP(S) URL.");
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error("NEXUS_LLM_BASE_URL must use HTTP or HTTPS.");
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error("NEXUS_LLM_BASE_URL must not contain credentials, query parameters, or a fragment.");
  }

  parsed.pathname = parsed.pathname.replace(/\/+$/u, "");
  const isHuggingFace = parsed.hostname === "router.huggingface.co";
  const key = customBase
    ? (env.NEXUS_LLM_API_KEY?.trim() || (isHuggingFace ? hfToken : undefined))
    : hfToken;
  if (isHuggingFace && !key) return undefined;

  const basePath = parsed.pathname || "";
  const endpoint = `${parsed.origin}${basePath}/chat/completions`;
  const configuredModel = env.NEXUS_LLM_MODEL?.trim();
  const model = configuredModel || (isHuggingFace ? DEFAULT_HUGGING_FACE_MODEL : "local-model");
  if (model.length > 200 || /[\r\n]/u.test(model)) {
    throw new Error("NEXUS_LLM_MODEL must be a model identifier no longer than 200 characters.");
  }

  return {
    kind: isHuggingFace ? "huggingface" : "openai-compatible",
    endpoint,
    model,
    ...(key ? { apiKey: key } : {}),
    providerName: isHuggingFace ? "Hugging Face Inference Providers" : "OpenAI-compatible endpoint",
  };
}

/** Extract a non-empty assistant message from an OpenAI-compatible response. */
export function readChatCompletion(payload: unknown): string {
  if (!payload || typeof payload !== "object" || !("choices" in payload) || !Array.isArray(payload.choices)) {
    throw new Error("The inference provider returned an invalid chat-completion payload.");
  }
  const first = payload.choices[0];
  if (!first || typeof first !== "object" || !("message" in first) ||
      !first.message || typeof first.message !== "object" || !("content" in first.message)) {
    throw new Error("The inference provider did not return an assistant message.");
  }
  const content = first.message.content;
  const text = typeof content === "string"
    ? content
    : Array.isArray(content)
      ? content.flatMap((part) => {
          if (part && typeof part === "object" && "text" in part && typeof part.text === "string") return [part.text];
          return [];
        }).join("")
      : "";
  const normalized = text.trim();
  if (!normalized) throw new Error("The inference provider returned an empty assistant message.");
  return normalized.slice(0, 12_000);
}
