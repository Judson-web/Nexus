import assert from "node:assert/strict";
import test from "node:test";
import { POST } from "./route.js";

function jsonRequest(value: unknown): Request {
  return new Request("http://localhost/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(value),
  });
}

test("rejects malformed JSON with a 400 response", async () => {
  const response = await POST(new Request("http://localhost/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{invalid",
  }));
  assert.equal(response.status, 400);
});

test("rejects a body over the declared size limit before parsing", async () => {
  const response = await POST(new Request("http://localhost/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json", "content-length": "4097" },
    body: JSON.stringify({ prompt: "hello" }),
  }));
  assert.equal(response.status, 413);
});

test("rejects oversized streamed payloads without Content-Length", async () => {
  const response = await POST(new Request("http://localhost/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: "x".repeat(5000) }),
  }));
  assert.equal(response.status, 413);
});

test("rejects blank prompts and invalid token limits", async () => {
  const blank = await POST(jsonRequest({ prompt: "   " }));
  assert.equal(blank.status, 400);
  const tokens = await POST(jsonRequest({ prompt: "hello", maxTokens: 0 }));
  assert.equal(tokens.status, 400);
});

test("generates a bounded response from the in-repository model", async () => {
  const response = await POST(jsonRequest({ prompt: "Nexus", maxTokens: 8 }));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store");
  const body = await response.json() as { text: string; model: string; note: string };
  assert.equal(typeof body.text, "string");
  assert.equal(body.model, "Nexus Bigram v0.1");
  assert.match(body.note, /tiny seed corpus/i);
});
