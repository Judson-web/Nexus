import { SeededRandom } from "../core/random.js";

export interface ContextModelConfig {
  readonly vocabSize: number;
  readonly contextSize: number;
  readonly embeddingSize: number;
  readonly hiddenSize: number;
  readonly seed: number;
}

export const DEFAULT_CONTEXT_CONFIG: ContextModelConfig = {
  vocabSize: 257,
  contextSize: 4,
  embeddingSize: 12,
  hiddenSize: 32,
  seed: 41,
};

export interface ContextCheckpoint {
  readonly format: "nexus-context-mlp";
  readonly version: 1;
  readonly config: ContextModelConfig;
  readonly embedding: number[];
  readonly w1: number[];
  readonly b1: number[];
  readonly w2: number[];
  readonly b2: number[];
}

export interface ContextTrainingMetrics {
  readonly loss: number;
  readonly examples: number;
}

function checkedSize(value: number, name: string, min: number, max: number): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer between ${min} and ${max}.`);
  }
}

export class ContextModel {
  public readonly config: ContextModelConfig;
  private readonly embedding: Float64Array;
  private readonly w1: Float64Array;
  private readonly b1: Float64Array;
  private readonly w2: Float64Array;
  private readonly b2: Float64Array;

  public constructor(config: ContextModelConfig = DEFAULT_CONTEXT_CONFIG, checkpoint?: Partial<ContextCheckpoint>) {
    checkedSize(config.vocabSize, "vocabSize", 2, 512);
    checkedSize(config.contextSize, "contextSize", 1, 16);
    checkedSize(config.embeddingSize, "embeddingSize", 2, 128);
    checkedSize(config.hiddenSize, "hiddenSize", 2, 256);
    if (!Number.isSafeInteger(config.seed)) throw new RangeError("seed must be a safe integer.");
    this.config = { ...config };
    const inputSize = config.contextSize * config.embeddingSize;
    this.embedding = this.makeWeights("embedding", config.vocabSize * config.embeddingSize, checkpoint?.embedding, 0.08);
    this.w1 = this.makeWeights("w1", inputSize * config.hiddenSize, checkpoint?.w1, 1 / Math.sqrt(inputSize));
    this.b1 = this.makeWeights("b1", config.hiddenSize, checkpoint?.b1, 0);
    this.w2 = this.makeWeights("w2", config.hiddenSize * config.vocabSize, checkpoint?.w2, 0.08);
    this.b2 = this.makeWeights("b2", config.vocabSize, checkpoint?.b2, 0);
  }

  public train(tokens: readonly number[], learningRate = 0.03, epochs = 1): ContextTrainingMetrics {
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.5) {
      throw new RangeError("learningRate must be in (0, 0.5].");
    }
    checkedSize(epochs, "epochs", 1, 10000);
    this.validateTokens(tokens);
    if (tokens.length <= this.config.contextSize) throw new Error("Training data must contain more tokens than contextSize.");
    const c = this.config;
    const inputSize = c.contextSize * c.embeddingSize;
    const embedded = new Float64Array(inputSize);
    const hidden = new Float64Array(c.hiddenSize);
    const logits = new Float64Array(c.vocabSize);
    const probs = new Float64Array(c.vocabSize);
    const dLogits = new Float64Array(c.vocabSize);
    const dHidden = new Float64Array(c.hiddenSize);
    const dPreHidden = new Float64Array(c.hiddenSize);
    const gradW2 = new Float64Array(this.w2.length);
    const gradW1 = new Float64Array(this.w1.length);
    const gradEmbedding = new Float64Array(this.embedding.length);
    let loss = 0;
    let examples = 0;

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      loss = 0;
      examples = 0;
      for (let position = 0; position < tokens.length; position += 1) {
        const target = tokens[position]!;
        const context = new Array<number>(c.contextSize).fill(0);
        for (let j = 0; j < c.contextSize; j += 1) {
          const tokenIndex = position - c.contextSize + j;
          context[j] = tokenIndex >= 0 ? tokens[tokenIndex]! : 0;
        }
        embedded.fill(0);
        for (let j = 0; j < c.contextSize; j += 1) {
          const tokenOffset = context[j]! * c.embeddingSize;
          const inputOffset = j * c.embeddingSize;
          for (let k = 0; k < c.embeddingSize; k += 1) {
            embedded[inputOffset + k] = this.embedding[tokenOffset + k]!;
          }
        }
        for (let h = 0; h < c.hiddenSize; h += 1) {
          let value = this.b1[h]!;
          for (let i = 0; i < inputSize; i += 1) value += embedded[i]! * this.w1[i * c.hiddenSize + h]!;
          hidden[h] = Math.tanh(value);
        }
        let maxLogit = Number.NEGATIVE_INFINITY;
        for (let v = 0; v < c.vocabSize; v += 1) {
          let value = this.b2[v]!;
          for (let h = 0; h < c.hiddenSize; h += 1) value += hidden[h]! * this.w2[h * c.vocabSize + v]!;
          logits[v] = value;
          maxLogit = Math.max(maxLogit, value);
        }
        let sum = 0;
        for (let v = 0; v < c.vocabSize; v += 1) {
          probs[v] = Math.exp(logits[v]! - maxLogit);
          sum += probs[v]!;
        }
        for (let v = 0; v < c.vocabSize; v += 1) probs[v] = probs[v]! / sum;
        loss -= Math.log(Math.max(probs[target]!, Number.MIN_VALUE));
        for (let v = 0; v < c.vocabSize; v += 1) dLogits[v] = probs[v]! - (v === target ? 1 : 0);
        dHidden.fill(0);
        for (let h = 0; h < c.hiddenSize; h += 1) {
          for (let v = 0; v < c.vocabSize; v += 1) {
            const index = h * c.vocabSize + v;
            gradW2[index] = hidden[h]! * dLogits[v]!;
            dHidden[h] += this.w2[index]! * dLogits[v]!;
          }
        }
        for (let h = 0; h < c.hiddenSize; h += 1) dPreHidden[h] = dHidden[h]! * (1 - hidden[h]! * hidden[h]!);
        gradW1.fill(0);
        for (let i = 0; i < inputSize; i += 1) {
          for (let h = 0; h < c.hiddenSize; h += 1) gradW1[i * c.hiddenSize + h] = embedded[i]! * dPreHidden[h]!;
        }
        gradEmbedding.fill(0);
        for (let j = 0; j < c.contextSize; j += 1) {
          const tokenOffset = context[j]! * c.embeddingSize;
          const inputOffset = j * c.embeddingSize;
          for (let k = 0; k < c.embeddingSize; k += 1) {
            let gradient = 0;
            for (let h = 0; h < c.hiddenSize; h += 1) gradient += this.w1[(inputOffset + k) * c.hiddenSize + h]! * dPreHidden[h]!;
            gradEmbedding[tokenOffset + k] += gradient;
          }
        }
        for (let i = 0; i < this.w2.length; i += 1) this.w2[i] = this.w2[i]! - learningRate * gradW2[i]!;
        for (let i = 0; i < this.w1.length; i += 1) this.w1[i] = this.w1[i]! - learningRate * gradW1[i]!;
        for (let h = 0; h < c.hiddenSize; h += 1) this.b1[h] = this.b1[h]! - learningRate * dPreHidden[h]!;
        for (let v = 0; v < c.vocabSize; v += 1) this.b2[v] = this.b2[v]! - learningRate * dLogits[v]!;
        for (let i = 0; i < this.embedding.length; i += 1) this.embedding[i] = this.embedding[i]! - learningRate * gradEmbedding[i]!;
        examples += 1;
      }
    }
    return { loss: loss / examples, examples };
  }

  public probabilities(context: readonly number[]): Float64Array {
    this.validateTokens(context);
    const c = this.config;
    const padded = new Array<number>(c.contextSize).fill(0);
    const tail = context.slice(-c.contextSize);
    padded.splice(c.contextSize - tail.length, tail.length, ...tail);
    const inputSize = c.contextSize * c.embeddingSize;
    const hidden = new Float64Array(c.hiddenSize);
    for (let j = 0; j < c.contextSize; j += 1) {
      const tokenOffset = padded[j]! * c.embeddingSize;
      for (let k = 0; k < c.embeddingSize; k += 1) {
        const value = this.embedding[tokenOffset + k]!;
        for (let h = 0; h < c.hiddenSize; h += 1) hidden[h] += value * this.w1[(j * c.embeddingSize + k) * c.hiddenSize + h]!;
      }
    }
    for (let h = 0; h < c.hiddenSize; h += 1) hidden[h] = Math.tanh(hidden[h]! + this.b1[h]!);
    const logits = new Float64Array(c.vocabSize);
    let max = Number.NEGATIVE_INFINITY;
    for (let v = 0; v < c.vocabSize; v += 1) {
      let value = this.b2[v]!;
      for (let h = 0; h < c.hiddenSize; h += 1) value += hidden[h]! * this.w2[h * c.vocabSize + v]!;
      logits[v] = value;
      max = Math.max(max, value);
    }
    let total = 0;
    for (let v = 0; v < c.vocabSize; v += 1) { logits[v] = Math.exp(logits[v]! - max); total += logits[v]!; }
    for (let v = 0; v < c.vocabSize; v += 1) logits[v] = logits[v]! / total;
    return logits;
  }

  public generate(prompt: readonly number[], maxTokens: number, random: () => number = Math.random): number[] {
    this.validateTokens(prompt);
    if (!Number.isInteger(maxTokens) || maxTokens < 0 || maxTokens > 100000) throw new RangeError("maxTokens must be between 0 and 100000.");
    const result = [...prompt];
    for (let i = 0; i < maxTokens; i += 1) {
      const probabilities = this.probabilities(result);
      const draw = random();
      if (!(draw >= 0 && draw < 1)) throw new RangeError("Random source must return a value in [0, 1).");
      let cumulative = 0;
      let next = probabilities.length - 1;
      for (let token = 0; token < probabilities.length; token += 1) {
        cumulative += probabilities[token]!;
        if (draw < cumulative) { next = token; break; }
      }
      result.push(next);
      if (next === this.config.vocabSize - 1) break;
    }
    return result;
  }

  public toCheckpoint(): ContextCheckpoint {
    return { format: "nexus-context-mlp", version: 1, config: this.config, embedding: Array.from(this.embedding), w1: Array.from(this.w1), b1: Array.from(this.b1), w2: Array.from(this.w2), b2: Array.from(this.b2) };
  }

  public static fromCheckpoint(value: unknown): ContextModel {
    if (!value || typeof value !== "object") throw new Error("Invalid context model checkpoint.");
    const item = value as Partial<ContextCheckpoint>;
    if (item.format !== "nexus-context-mlp" || item.version !== 1 || !item.config || !Array.isArray(item.embedding) || !Array.isArray(item.w1) || !Array.isArray(item.b1) || !Array.isArray(item.w2) || !Array.isArray(item.b2)) throw new Error("Invalid or unsupported context model checkpoint.");
    return new ContextModel(item.config, item);
  }

  private makeWeights(name: string, size: number, supplied: readonly number[] | undefined, scale: number): Float64Array {
    if (supplied && supplied.length !== size) throw new Error(`Checkpoint ${name} weight count mismatch.`);
    const result = new Float64Array(size);
    if (supplied) {
      supplied.forEach((value, i) => { if (!Number.isFinite(value)) throw new Error("Checkpoint weights must be finite."); result[i] = value; });
    } else {
      const random = new SeededRandom(this.config.seed + name.length * 997 + size);
      for (let i = 0; i < size; i += 1) result[i] = (random.next() * 2 - 1) * scale;
    }
    return result;
  }

  private validateTokens(tokens: readonly number[]): void {
    for (const token of tokens) if (!Number.isInteger(token) || token < 0 || token >= this.config.vocabSize) throw new RangeError(`Token ID ${token} is outside the model vocabulary.`);
  }
}
