import assert from "node:assert/strict";
import test from "node:test";
import { readChatCompletion, resolveChatProvider } from "./openai-compatible.js";

test("does not silently configure a provider without credentials or an endpoint", () => {
  assert.equal(resolveChatProvider({}), undefined);
  assert.equal(resolveChatProvider({ HF_TOKEN: "  " }), undefined);
});

test("uses the Hugging Face OpenAI-compatible router and an open-weight instruct default", () => {
  const config = resolveChatProvider({ HF_TOKEN: "secret-token" });
  assert.ok(config);
  assert.equal(config.kind, "huggingface");
  assert.equal(config.endpoint, "https://router.huggingface.co/v1/chat/completions");
  assert.equal(config.model, "Qwen/Qwen2.5-7B-Instruct-1M:fastest");
  assert.equal(config.apiKey, "secret-token");
  assert.equal(config.providerName, "Hugging Face Inference Providers");
});

test("supports self-hosted OpenAI-compatible endpoints and never leaks the HF token to them", () => {
  const config = resolveChatProvider({
    NEXUS_LLM_BASE_URL: "http://127.0.0.1:8080/v1/",
    HF_TOKEN: "must-not-be-forwarded",
  });
  assert.ok(config);
  assert.equal(config.endpoint, "http://127.0.0.1:8080/v1/chat/completions");
  assert.equal(config.kind, "openai-compatible");
  assert.equal(config.model, "local-model");
  assert.equal(config.apiKey, undefined);
});

test("uses the configured key and model for a custom endpoint", () => {
  const config = resolveChatProvider({
    NEXUS_LLM_BASE_URL: "https://llm.example.test/v1",
    NEXUS_LLM_API_KEY: "custom-key",
    NEXUS_LLM_MODEL: "org/model-instruct",
  });
  assert.ok(config);
  assert.equal(config.endpoint, "https://llm.example.test/v1/chat/completions");
  assert.equal(config.apiKey, "custom-key");
  assert.equal(config.model, "org/model-instruct");
});

test("rejects malformed or credential-bearing endpoint URLs", () => {
  assert.throws(() => resolveChatProvider({ NEXUS_LLM_BASE_URL: "not-a-url" }), /absolute HTTP\(S\) URL/i);
  assert.throws(() => resolveChatProvider({ NEXUS_LLM_BASE_URL: "https://user:pass@example.test/v1" }), /must not contain credentials/i);
  assert.throws(() => resolveChatProvider({ NEXUS_LLM_BASE_URL: "ftp://example.test/v1" }), /HTTP or HTTPS/i);
});

test("parses standard and text-part chat completion responses", () => {
  assert.equal(readChatCompletion({ choices: [{ message: { role: "assistant", content: "  Hello from a pretrained model.  " } }] }), "Hello from a pretrained model.");
  assert.equal(readChatCompletion({ choices: [{ message: { content: [{ type: "text", text: "First " }, { type: "text", text: "second." }] } }] }), "First second.");
  assert.throws(() => readChatCompletion({ choices: [] }), /did not return an assistant message/i);
  assert.throws(() => readChatCompletion({ choices: [{ message: { content: "  " } }] }), /empty assistant message/i);
});
