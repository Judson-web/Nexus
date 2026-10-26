export const NEXUS_CHAT_SFT_FORMAT = "nexus-chat-sft-v1" as const;
export const NEXUS_CAUSAL_LM_FORMAT = "nexus-causal-lm-v1" as const;

export type NexusTrainingFormat = typeof NEXUS_CHAT_SFT_FORMAT | typeof NEXUS_CAUSAL_LM_FORMAT;

/** Format a plain user prompt exactly like Nexus's user/assistant SFT examples. */
export function formatChatSftPrompt(prompt: string): string {
  const normalized = prompt.normalize("NFC").replace(/\s+/gu, " ").trim();
  if (!normalized) throw new Error("Chat SFT prompts must contain non-whitespace text.");
  const escaped = normalized
    .replace(/Assistant: /gu, "Assistant : ")
    .replace(/User: /gu, "User : ");
  return `User: ${escaped} Assistant: `;
}
