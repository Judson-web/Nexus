import { BYTE_VOCAB_SIZE } from "../tokenizer/byte-tokenizer.js";

export interface BigramModelConfig {
  readonly vocabSize: number;
  readonly seed: number;
  readonly initializationScale: number;
}

export const DEFAULT_MODEL_CONFIG: BigramModelConfig = {
  vocabSize: BYTE_VOCAB_SIZE,
  seed: 17,
  initializationScale: 0.01,
};

export function validateConfig(config: BigramModelConfig): void {
  if (!Number.isInteger(config.vocabSize) || config.vocabSize < 2 || config.vocabSize > 4096) {
    throw new RangeError("vocabSize must be an integer between 2 and 4096.");
  }
  if (!Number.isSafeInteger(config.seed)) throw new RangeError("seed must be a safe integer.");
  if (!Number.isFinite(config.initializationScale) || config.initializationScale <= 0) {
    throw new RangeError("initializationScale must be a positive finite number.");
  }
}
