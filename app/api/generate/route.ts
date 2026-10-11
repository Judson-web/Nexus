import { NextResponse } from "next/server";
import { BigramModel } from "../../../src/model/bigram-model";
import { ByteTokenizer } from "../../../src/tokenizer/byte-tokenizer";

export const runtime = "nodejs";
export const maxDuration = 10;

const MAX_BODY_BYTES = 4096;
const CORPUS = [
  "Nexus is an independent research project building a language model from the ground up.",
  "A language model learns patterns from examples and predicts the next token.",
  "Reliable experiments use curated data, explicit configurations, and repeatable evaluation.",
  "Good engineering favors small verified steps over unsupported claims.",
  "TypeScript helps keep training code explicit, testable, and maintainable.",
  "Independent AI research requires careful data provenance, measurement, and iteration.",
].join("\n");

let cachedModel: BigramModel | undefined;
const tokenizer = new ByteTokenizer();

function getModel(): BigramModel {
  if (cachedModel) return cachedModel;
  const model = new BigramModel({ vocabSize: 257, seed: 17, initializationScale: 0.01 });
  model.train(tokenizer.encode(CORPUS), 0.08, 8);
  cachedModel = model;
  return model;
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
    const generated = getModel().generate(tokenizer.encode(prompt, false), maxTokens);
    return NextResponse.json({
      text: tokenizer.decode(generated).slice(0, 1200),
      model: "Nexus Bigram v0.1",
      note: "Experimental baseline trained in-memory on a tiny seed corpus. Outputs are expected to be incoherent; this is a runtime smoke test, not a capable conversational model.",
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "The model could not generate a response." }, { status: 500 });
  }
}
