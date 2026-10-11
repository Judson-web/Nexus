import { SeededRandom } from "../core/random.js";
import { DEFAULT_MODEL_CONFIG, validateConfig, type BigramModelConfig } from "./config.js";

export interface TrainingMetrics {
  readonly loss: number;
  readonly examples: number;
}

export interface BigramCheckpoint {
  readonly format: "nexus-bigram";
  readonly version: 1;
  readonly config: BigramModelConfig;
  readonly weights: number[];
}

export class BigramModel {
  public readonly config: BigramModelConfig;
  private readonly weights: Float64Array;

  public constructor(config: BigramModelConfig = DEFAULT_MODEL_CONFIG, weights?: readonly number[]) {
    validateConfig(config);
    this.config = { ...config };
    const size = config.vocabSize * config.vocabSize;
    if (weights && weights.length !== size) throw new Error("Checkpoint weight count does not match config.");
    this.weights = new Float64Array(size);
    if (weights) {
      weights.forEach((value, index) => {
        if (!Number.isFinite(value)) throw new Error("Weights must all be finite numbers.");
        this.weights[index] = value;
      });
    } else {
      const random = new SeededRandom(config.seed);
      for (let i = 0; i < size; i += 1) this.weights[i] = (random.next() * 2 - 1) * config.initializationScale;
    }
  }

  public train(tokens: readonly number[], learningRate = 0.08, epochs = 1): TrainingMetrics {
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      throw new RangeError("learningRate must be in (0, 1].");
    }
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 10000) {
      throw new RangeError("epochs must be an integer between 1 and 10000.");
    }
    this.validateTokens(tokens);
    if (tokens.length < 2) throw new Error("At least two tokens are required for training.");
    let lossSum = 0;
    let examples = 0;
    const vocab = this.config.vocabSize;
    const logits = new Float64Array(vocab);
    const probabilities = new Float64Array(vocab);

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      lossSum = 0;
      examples = 0;
      for (let i = 0; i < tokens.length - 1; i += 1) {
        const context = tokens[i]!;
        const target = tokens[i + 1]!;
        const offset = context * vocab;
        let maxLogit = Number.NEGATIVE_INFINITY;
        for (let token = 0; token < vocab; token += 1) {
          const value = this.weights[offset + token]!;
          logits[token] = value;
          if (value > maxLogit) maxLogit = value;
        }
        let sum = 0;
        for (let token = 0; token < vocab; token += 1) {
          const probability = Math.exp(logits[token]! - maxLogit);
          probabilities[token] = probability;
          sum += probability;
        }
        for (let token = 0; token < vocab; token += 1) probabilities[token] /= sum;
        lossSum -= Math.log(Math.max(probabilities[target]!, Number.MIN_VALUE));
        for (let token = 0; token < vocab; token += 1) {
          const gradient = probabilities[token]! - (token === target ? 1 : 0);
          this.weights[offset + token] -= learningRate * gradient;
        }
        examples += 1;
      }
    }
    return { loss: lossSum / examples, examples };
  }

  public probabilities(contextToken: number): Float64Array {
    this.validateToken(contextToken);
    const vocab = this.config.vocabSize;
    const offset = contextToken * vocab;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < vocab; i += 1) max = Math.max(max, this.weights[offset + i]!);
    const result = new Float64Array(vocab);
    let total = 0;
    for (let i = 0; i < vocab; i += 1) {
      result[i] = Math.exp(this.weights[offset + i]! - max);
      total += result[i]!;
    }
    for (let i = 0; i < vocab; i += 1) result[i] /= total;
    return result;
  }

  public sample(contextToken: number, random: () => number = Math.random): number {
    const probabilities = this.probabilities(contextToken);
    const draw = random();
    if (!(draw >= 0 && draw < 1)) throw new RangeError("Random source must return a value in [0, 1).");
    let cumulative = 0;
    for (let token = 0; token < probabilities.length; token += 1) {
      cumulative += probabilities[token]!;
      if (draw < cumulative) return token;
    }
    return probabilities.length - 1;
  }

  public generate(prompt: readonly number[], maxTokens: number, random: () => number = Math.random): number[] {
    this.validateTokens(prompt);
    if (!Number.isInteger(maxTokens) || maxTokens < 0 || maxTokens > 100000) {
      throw new RangeError("maxTokens must be an integer between 0 and 100000.");
    }
    const result = [...prompt];
    for (let i = 0; i < maxTokens; i += 1) {
      const next = this.sample(result[result.length - 1]!, random);
      result.push(next);
      if (next === this.config.vocabSize - 1) break;
    }
    return result;
  }

  public toCheckpoint(): BigramCheckpoint {
    return { format: "nexus-bigram", version: 1, config: this.config, weights: Array.from(this.weights) };
  }

  public static fromCheckpoint(value: unknown): BigramModel {
    if (!value || typeof value !== "object") throw new Error("Invalid checkpoint: expected an object.");
    const checkpoint = value as Partial<BigramCheckpoint>;
    if (checkpoint.format !== "nexus-bigram" || checkpoint.version !== 1 ||
        !checkpoint.config || !Array.isArray(checkpoint.weights)) {
      throw new Error("Invalid or unsupported Nexus checkpoint.");
    }
    return new BigramModel(checkpoint.config, checkpoint.weights);
  }

  private validateToken(token: number): void {
    if (!Number.isInteger(token) || token < 0 || token >= this.config.vocabSize) {
      throw new RangeError(`Token ID ${token} is outside the model vocabulary.`);
    }
  }

  private validateTokens(tokens: readonly number[]): void {
    for (const token of tokens) this.validateToken(token);
  }
}
