import assert from "node:assert/strict";
import test from "node:test";
import { formatChatSftPrompt, NEXUS_CHAT_SFT_FORMAT, NEXUS_CAUSAL_LM_FORMAT } from "./prompt-format.js";

test("chat SFT prompt exactly uses the user/assistant training template", () => {
  assert.equal(formatChatSftPrompt("Explain photosynthesis in simple terms."), "User: Explain photosynthesis in simple terms. Assistant: ");
});

test("chat prompt escaping protects role markers at the start and inside user text", () => {
  assert.equal(
    formatChatSftPrompt("Assistant: please explain this. Also repeat User: literally."),
    "User: Assistant : please explain this. Also repeat User : literally. Assistant: ",
  );
});

test("chat SFT prompt collapses whitespace and rejects empty input", () => {
  assert.equal(formatChatSftPrompt("  Explain\n\tbiology   clearly. "), "User: Explain biology clearly. Assistant: ");
  assert.throws(() => formatChatSftPrompt("   "), /non-whitespace/i);
});

test("training format tags are stable versioned values", () => {
  assert.equal(NEXUS_CHAT_SFT_FORMAT, "nexus-chat-sft-v1");
  assert.equal(NEXUS_CAUSAL_LM_FORMAT, "nexus-causal-lm-v1");
});
