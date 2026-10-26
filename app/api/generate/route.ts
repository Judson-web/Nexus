import { NextResponse } from "next/server";
import { loadVerifiedTransformerArtifact, type TransformerTokenizer } from "../../../src/model/verified-transformer-artifact.js";
import type { TransformerModel, TransformerTrainingFormat } from "../../../src/model/transformer-model.js";
import { formatChatSftPrompt, NEXUS_CHAT_SFT_FORMAT } from "../../../src/chat/prompt-format.js";

export const runtime = "nodejs";
export const maxDuration = 10;

const MAX_BODY_BYTES = 4096;
let cachedTransformer: Promise<{ model: TransformerModel; tokenizer: TransformerTokenizer; sha256: string; trainingFormat: TransformerTrainingFormat }> | undefined;

async function getConfiguredTransformer(): Promise<{ model: TransformerModel; tokenizer: TransformerTokenizer; sha256: string; trainingFormat: TransformerTrainingFormat } | undefined> {
  const checkpointPath = process.env.NEXUS_TRANSFORMER_CHECKPOINT?.trim();
  if (!checkpointPath) return undefined;
  if (!cachedTransformer) {
    cachedTransformer = loadVerifiedTransformerArtifact(checkpointPath).then(({ model, tokenizer, sha256, trainingFormat }) => ({ model, tokenizer, sha256, trainingFormat }));
  }
  return cachedTransformer;
}

async function readBoundedJson(request: Request): Promise<unknown> {
  const contentLength = request.headers.get("content-length");
  if (contentLength !== null) {
    const declaredLength = Number(contentLength);
    if (!Number.isSafeInteger(declaredLength) || declaredLength < 0 || declaredLength > MAX_BODY_BYTES) {
      throw new RangeError("Request body must be 4 KB or smaller.");
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
        throw new RangeError("Request body must be 4 KB or smaller.");
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

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await readBoundedJson(request);
  } catch (error) {
    if (error instanceof RangeError) {
      return NextResponse.json({ error: error.message }, { status: 413 });
    }
    return NextResponse.json({ error: "Send a valid JSON request." }, { status: 400 });
  }

  if (!body || typeof body !== "object" || !("prompt" in body) || typeof body.prompt !== "string") {
    return NextResponse.json({ error: "A text prompt is required." }, { status: 400 });
  }
  const prompt = body.prompt.trim();
  if (!prompt) return NextResponse.json({ error: "Enter a prompt first." }, { status: 400 });
  if (prompt.length > 160) return NextResponse.json({ error: "Prompts must be 160 characters or fewer." }, { status: 400 });
  const maxTokens = "maxTokens" in body && typeof body.maxTokens === "number" ? body.maxTokens : 100;
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 160) {
    return NextResponse.json({ error: "maxTokens must be an integer between 1 and 160." }, { status: 400 });
  }

  try {
    const transformer = await getConfiguredTransformer();
    if (transformer) {
      const isChatSft = transformer.trainingFormat === NEXUS_CHAT_SFT_FORMAT;
      const modelPrompt = isChatSft ? formatChatSftPrompt(prompt) : prompt;
      const promptTokens = transformer.tokenizer.encode(modelPrompt, false);
      const generated = transformer.model.generate(promptTokens, maxTokens);
      const outputTokens = isChatSft ? generated.slice(promptTokens.length) : generated;
      return NextResponse.json({
        text: transformer.tokenizer.decode(outputTokens).trim().slice(0, 1200),
        model: "Nexus Transformer (research checkpoint)",
        trainingFormat: transformer.trainingFormat,
        modelSha256: transformer.sha256,
        note: isChatSft
          ? "Experimental self-trained Transformer. Assistant prompt format inferred from checksum-verified checkpoint metadata; output quality remains experimental."
          : "Experimental self-trained Transformer. Checkpoint checksum verified at load; output quality depends on training data and evaluation.",
      }, { headers: { "Cache-Control": "no-store" } });
    }
    return NextResponse.json({
      error: "No verified Transformer checkpoint is configured. Inference is disabled until a trained checkpoint and matching SHA-256 sidecar are provisioned.",
      code: "MODEL_NOT_CONFIGURED",
    }, { status: 503, headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "The model could not generate a response." }, { status: 500 });
  }
}
