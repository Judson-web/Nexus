import { SeededRandom } from "../core/random.js";

export interface CausalAttentionConfig {
  readonly vocabSize: number;
  readonly contextSize: number;
  readonly embeddingSize: number;
  readonly seed: number;
}

export interface CausalAttentionCheckpoint {
  readonly format: "nexus-causal-attention";
  readonly version: 1;
  readonly config: CausalAttentionConfig;
  readonly embedding: number[];
  readonly positionBias: number[];
  readonly output: number[];
  readonly outputBias: number[];
}

export interface CausalAttentionMetrics {
  readonly loss: number;
  readonly examples: number;
}

export const DEFAULT_CAUSAL_ATTENTION_CONFIG: CausalAttentionConfig = {
  vocabSize: 257,
  contextSize: 16,
  embeddingSize: 16,
  seed: 73,
};

function checkInteger(value: number, name: string, min: number, max: number): void {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer between ${min} and ${max}.`);
  }
}

/**
 * A compact causal self-attention language model.
 *
 * Each position attends only to itself and earlier positions. Attention logits
 * are learned relative-position biases; token embeddings and the vocabulary
 * projection are trained with backpropagation. This is an intentionally small
 * research model, not a full multi-head Transformer block.
 */
export class CausalAttentionModel {
  public readonly config: CausalAttentionConfig;
  private readonly embedding: Float64Array;
  private readonly positionBias: Float64Array;
  private readonly output: Float64Array;
  private readonly outputBias: Float64Array;

  public constructor(config: CausalAttentionConfig = DEFAULT_CAUSAL_ATTENTION_CONFIG, checkpoint?: Partial<CausalAttentionCheckpoint>) {
    checkInteger(config.vocabSize, "vocabSize", 2, 512);
    checkInteger(config.contextSize, "contextSize", 1, 64);
    checkInteger(config.embeddingSize, "embeddingSize", 2, 64);
    if (!Number.isSafeInteger(config.seed)) throw new RangeError("seed must be a safe integer.");
    this.config = { ...config };
    const random = new SeededRandom(config.seed);
    const init = (length: number, values: number[] | undefined, scale: number): Float64Array => {
      if (values !== undefined) {
        if (!Array.isArray(values) || values.length !== length || values.some((v) => !Number.isFinite(v))) {
          throw new Error("Checkpoint weight count mismatch or non-finite value.");
        }
        return Float64Array.from(values);
      }
      return Float64Array.from({ length }, () => (random.next() * 2 - 1) * scale);
    };
    this.embedding = init(config.vocabSize * config.embeddingSize, checkpoint?.embedding, 0.08);
    this.positionBias = init(config.contextSize, checkpoint?.positionBias, 0.01);
    this.output = init(config.embeddingSize * config.vocabSize, checkpoint?.output, 0.08);
    this.outputBias = init(config.vocabSize, checkpoint?.outputBias, 0);
  }

  private forward(context: readonly number[]) {
    const { vocabSize, contextSize, embeddingSize } = this.config;
    const ids = Array.from({ length: contextSize }, (_, i) => {
      const offset = context.length - contextSize + i;
      return offset >= 0 ? (context[offset] ?? 0) : 0;
    });
    const vectors: Float64Array[] = ids.map((id) => this.embedding.slice(id * embeddingSize, (id + 1) * embeddingSize));
    const last = contextSize - 1;
    const scores = new Float64Array(contextSize);
    for (let i = 0; i <= last; i++) scores[i] = this.positionBias[last - i]!;
    const max = Math.max(...scores);
    let total = 0;
    for (let i = 0; i <= last; i++) { scores[i] = Math.exp(scores[i]! - max); total += scores[i]!; }
    for (let i = 0; i <= last; i++) scores[i] = scores[i]! / total;

    const pooled = new Float64Array(embeddingSize);
    for (let i = 0; i <= last; i++) {
      for (let d = 0; d < embeddingSize; d++) pooled[d] = pooled[d]! + scores[i]! * vectors[i]![d]!;
    }
    const logits = new Float64Array(vocabSize);
    for (let token = 0; token < vocabSize; token++) {
      let value = this.outputBias[token]!;
      for (let d = 0; d < embeddingSize; d++) value += pooled[d]! * this.output[d * vocabSize + token]!;
      logits[token] = value;
    }
    const logitMax = Math.max(...logits);
    let normalizer = 0;
    for (let i = 0; i < vocabSize; i++) normalizer = normalizer + Math.exp(logits[i]! - logitMax);
    const logZ = logitMax + Math.log(normalizer);
    const probabilities = Float64Array.from(logits, (v) => Math.exp(v - logZ));
    return { ids, vectors, scores, pooled, logits, probabilities, logZ };
  }

  public probabilities(context: readonly number[]): number[] {
    this.validateTokens(context);
    return Array.from(this.forward(context).probabilities);
  }

  public train(tokens: readonly number[], learningRate = 0.03, epochs = 1): CausalAttentionMetrics {
    this.validateTokens(tokens);
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.25) throw new RangeError("learningRate must be in (0, 0.25].");
    checkInteger(epochs, "epochs", 1, 10000);
    if (tokens.length < 2) throw new Error("At least two tokens are required.");
    const { vocabSize, contextSize, embeddingSize } = this.config;
    let loss = 0;
    let examples = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      loss = 0;
      examples = 0;
      for (let targetIndex = 1; targetIndex < tokens.length; targetIndex++) {
        const context = tokens.slice(Math.max(0, targetIndex - contextSize), targetIndex);
        const target = tokens[targetIndex]!;
        const pass = this.forward(context);
        loss += pass.logZ - pass.logits[target]!;
        examples++;

        const gradLogits = Float64Array.from(pass.probabilities);
        gradLogits[target] = gradLogits[target]! - 1;
        const gradPooled = new Float64Array(embeddingSize);
        for (let d = 0; d < embeddingSize; d++) {
          for (let token = 0; token < vocabSize; token++) {
            const index = d * vocabSize + token;
            gradPooled[d] = gradPooled[d]! + this.output[index]! * gradLogits[token]!;
            this.output[index] = this.output[index]! - learningRate * pass.pooled[d]! * gradLogits[token]!;
          }
        }
        for (let token = 0; token < vocabSize; token++) this.outputBias[token] = this.outputBias[token]! - learningRate * gradLogits[token]!;
        // Backpropagate through the weighted sum and the learned attention
        // distribution, including its relative-position logits.
        const gradAttention = new Float64Array(contextSize);
        const gradVectors = Array.from({ length: contextSize }, () => new Float64Array(embeddingSize));
        for (let i = 0; i < contextSize; i++) {
          let dot = 0;
          for (let d = 0; d < embeddingSize; d++) {
            gradVectors[i]![d] = pass.scores[i]! * gradPooled[d]!;
            dot += gradPooled[d]! * pass.vectors[i]![d]!;
          }
          gradAttention[i] = dot;
        }
        let weightedGrad = 0;
        for (let i = 0; i < contextSize; i++) weightedGrad += pass.scores[i]! * gradAttention[i]!;
        for (let i = 0; i < contextSize; i++) {
          const relative = contextSize - 1 - i;
          const gradScore = pass.scores[i]! * (gradAttention[i]! - weightedGrad);
          this.positionBias[relative] = this.positionBias[relative]! - learningRate * gradScore;
          const tokenId = pass.ids[i]!;
          for (let d = 0; d < embeddingSize; d++) {
            const index = tokenId * embeddingSize + d;
            this.embedding[index] = this.embedding[index]! - learningRate * gradVectors[i]![d]!;
          }
        }
      }
    }
    return { loss: loss / Math.max(1, examples), examples };
  }

  public generate(prompt: readonly number[], maxTokens: number, random: () => number = Math.random): number[] {
    this.validateTokens(prompt);
    checkInteger(maxTokens, "maxTokens", 0, 10000);
    const result = [...prompt];
    for (let i = 0; i < maxTokens; i++) {
      const context = result.slice(-this.config.contextSize);
      const probabilities = this.forward(context).probabilities;
      const draw = random();
      if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw new RangeError("Random draw must be in [0, 1).");
      let cumulative = 0;
      let selected = probabilities.length - 1;
      for (let token = 0; token < probabilities.length; token++) {
        cumulative += probabilities[token]!;
        if (draw < cumulative) { selected = token; break; }
      }
      result.push(selected);
    }
    return result;
  }

  public toCheckpoint(): CausalAttentionCheckpoint {
    return {
      format: "nexus-causal-attention",
      version: 1,
      config: { ...this.config },
      embedding: Array.from(this.embedding),
      positionBias: Array.from(this.positionBias),
      output: Array.from(this.output),
      outputBias: Array.from(this.outputBias),
    };
  }

  public static fromCheckpoint(checkpoint: CausalAttentionCheckpoint): CausalAttentionModel {
    if (!checkpoint || checkpoint.format !== "nexus-causal-attention" || checkpoint.version !== 1 || !checkpoint.config) {
      throw new Error("Unsupported causal-attention checkpoint.");
    }
    return new CausalAttentionModel(checkpoint.config, checkpoint);
  }

  private validateTokens(tokens: readonly number[]): void {
    if (!Array.isArray(tokens) || tokens.some((token) => !Number.isInteger(token) || token < 0 || token >= this.config.vocabSize)) {
      throw new RangeError("Input contains an invalid token ID.");
    }
  }
}
