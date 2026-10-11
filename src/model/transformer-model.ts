import { SeededRandom } from "../core/random.js";
import { transformerLossAndGradients } from "./transformer-gradient.js";

export interface TransformerConfig {
  readonly vocabSize: number;
  readonly contextSize: number;
  readonly embeddingSize: number;
  readonly headCount: number;
  readonly feedForwardSize: number;
  readonly seed: number;
}

export interface TransformerCheckpoint {
  readonly format: "nexus-transformer";
  readonly version: 1;
  readonly config: TransformerConfig;
  readonly weights: number[];
}

export interface TransformerMetrics {
  readonly loss: number;
  readonly examples: number;
}

export const DEFAULT_TRANSFORMER_CONFIG: TransformerConfig = {
  vocabSize: 257,
  contextSize: 8,
  embeddingSize: 16,
  headCount: 2,
  feedForwardSize: 32,
  seed: 101,
};

function checkInteger(value: number, name: string, min: number, max: number): void {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer between ${min} and ${max}.`);
  }
}

/**
 * A compact decoder-only Transformer research model: learned token and position
 * embeddings, masked multi-head self-attention (Q/K/V projections), residual
 * connections, layer normalization, a GELU feed-forward block, and a vocabulary
 * projection. This is a single-block CPU reference implementation, not a
 * production-scale model.
 *
 * Training uses simultaneous perturbation stochastic approximation (SPSA) to
 * estimate gradients over the complete parameter vector. It is deliberately
 * simple and derivative-free; it is a research stepping stone, not a substitute
 * for the exact backpropagation implementation needed for larger models.
 */
export class TransformerModel {
  public readonly config: TransformerConfig;
  private readonly weights: Float64Array;
  private readonly ranges: Record<string, { start: number; length: number }>;

  public constructor(config: TransformerConfig = DEFAULT_TRANSFORMER_CONFIG, checkpoint?: Partial<TransformerCheckpoint>) {
    checkInteger(config.vocabSize, "vocabSize", 2, 512);
    checkInteger(config.contextSize, "contextSize", 1, 32);
    checkInteger(config.embeddingSize, "embeddingSize", 4, 32);
    checkInteger(config.headCount, "headCount", 1, 8);
    if (config.embeddingSize % config.headCount !== 0) {
      throw new RangeError("embeddingSize must be divisible by headCount.");
    }
    checkInteger(config.feedForwardSize, "feedForwardSize", 4, 128);
    if (!Number.isSafeInteger(config.seed)) throw new RangeError("seed must be a safe integer.");
    this.config = { ...config };
    const e = config.embeddingSize;
    const v = config.vocabSize;
    const c = config.contextSize;
    const f = config.feedForwardSize;
    const sizes: Record<string, number> = {
      token: v * e, position: c * e,
      q: e * e, k: e * e, value: e * e, attentionOut: e * e,
      attentionNormScale: e, attentionNormBias: e,
      ff1: e * f, ff1Bias: f, ff2: f * e, ff2Bias: e,
      ffNormScale: e, ffNormBias: e,
      lmHead: e * v, lmBias: v,
    };
    this.ranges = {};
    let length = 0;
    for (const [name, size] of Object.entries(sizes)) {
      this.ranges[name] = { start: length, length: size };
      length += size;
    }
    if (checkpoint?.weights !== undefined) {
      if (checkpoint.weights.length !== length || checkpoint.weights.some((x) => !Number.isFinite(x))) {
        throw new Error("Checkpoint weight count mismatch or non-finite value.");
      }
      this.weights = Float64Array.from(checkpoint.weights);
    } else {
      const random = new SeededRandom(config.seed);
      this.weights = new Float64Array(length);
      for (const [name, range] of Object.entries(this.ranges)) {
        const scale = name === "attentionNormScale" || name === "ffNormScale" ? 0 : 1 / Math.sqrt(Math.max(1, e));
        for (let i = 0; i < range.length; i++) {
          this.weights[range.start + i] = name === "attentionNormScale" || name === "ffNormScale"
            ? 1
            : name === "attentionNormBias" || name === "ffNormBias" || name.endsWith("Bias")
              ? 0
              : (random.next() * 2 - 1) * scale;
        }
      }
    }
  }

  private vector(name: string): Float64Array {
    const range = this.ranges[name]!;
    return this.weights.slice(range.start, range.start + range.length);
  }

  private normalize(input: Float64Array, scale: Float64Array, bias: Float64Array): Float64Array {
    const mean = input.reduce((sum, x) => sum + x, 0) / input.length;
    const variance = input.reduce((sum, x) => sum + (x - mean) ** 2, 0) / input.length;
    const inverse = 1 / Math.sqrt(variance + 1e-5);
    return Float64Array.from(input, (x, i) => (x - mean) * inverse * scale[i]! + bias[i]!);
  }

  private matVec(input: Float64Array, name: string, outSize: number): Float64Array {
    const matrix = this.vector(name);
    const out = new Float64Array(outSize);
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < outSize; j++) out[j] = out[j]! + input[i]! * matrix[i * outSize + j]!;
    }
    return out;
  }

  private forward(context: readonly number[]): Float64Array {
    const { vocabSize: v, contextSize: c, embeddingSize: e, headCount } = this.config;
    const padded = new Array<number>(c).fill(0);
    const tail = context.slice(-c);
    padded.splice(c - tail.length, tail.length, ...tail);
    const tokenEmbedding = this.vector("token");
    const positionEmbedding = this.vector("position");
    const qMatrix = this.vector("q");
    const kMatrix = this.vector("k");
    const valueMatrix = this.vector("value");
    const headSize = e / headCount;
    const states: Float64Array[] = [];
    const qs: Float64Array[] = [];
    const ks: Float64Array[] = [];
    const vs: Float64Array[] = [];
    for (let p = 0; p < c; p++) {
      const state = new Float64Array(e);
      for (let d = 0; d < e; d++) state[d] = tokenEmbedding[padded[p]! * e + d]! + positionEmbedding[p * e + d]!;
      states.push(state);
      const project = (matrix: Float64Array) => {
        const out = new Float64Array(e);
        for (let i = 0; i < e; i++) for (let j = 0; j < e; j++) out[j] = out[j]! + state[i]! * matrix[i * e + j]!;
        return out;
      };
      qs.push(project(qMatrix)); ks.push(project(kMatrix)); vs.push(project(valueMatrix));
    }

    const attended = states.map(() => new Float64Array(e));
    for (let p = 0; p < c; p++) {
      for (let h = 0; h < headCount; h++) {
        const start = h * headSize;
        const scores: number[] = [];
        for (let source = 0; source <= p; source++) {
          let dot = 0;
          for (let d = 0; d < headSize; d++) dot += qs[p]![start + d]! * ks[source]![start + d]!;
          scores.push(dot / Math.sqrt(headSize));
        }
        const max = Math.max(...scores);
        const exp = scores.map((score) => Math.exp(score - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        for (let source = 0; source <= p; source++) {
          const probability = exp[source]! / sum;
          for (let d = 0; d < headSize; d++) {
            attended[p]![start + d] = attended[p]![start + d]! + probability * vs[source]![start + d]!;
          }
        }
      }
    }

    const outMatrix = this.vector("attentionOut");
    const attentionScale = this.vector("attentionNormScale");
    const attentionBias = this.vector("attentionNormBias");
    const ff1 = this.vector("ff1");
    const ff1Bias = this.vector("ff1Bias");
    const ff2 = this.vector("ff2");
    const ff2Bias = this.vector("ff2Bias");
    const ffScale = this.vector("ffNormScale");
    const ffBias = this.vector("ffNormBias");
    const last = c - 1;
    let state = states[last]!;
    const projected = new Float64Array(e);
    for (let i = 0; i < e; i++) for (let j = 0; j < e; j++) projected[j] = projected[j]! + attended[last]![i]! * outMatrix[i * e + j]!;
    state = this.normalize(Float64Array.from(state, (x, i) => x + projected[i]!), attentionScale, attentionBias);
    const hidden = new Float64Array(this.config.feedForwardSize);
    for (let i = 0; i < e; i++) for (let j = 0; j < hidden.length; j++) {
      const z = state[i]! * ff1[i * hidden.length + j]!;
      hidden[j] = hidden[j]! + z;
    }
    for (let j = 0; j < hidden.length; j++) {
      const z = hidden[j]! + ff1Bias[j]!;
      hidden[j] = 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));
    }
    const ffOutput = new Float64Array(e);
    for (let i = 0; i < hidden.length; i++) for (let j = 0; j < e; j++) {
      ffOutput[j] = ffOutput[j]! + hidden[i]! * ff2[i * e + j]!;
    }
    for (let j = 0; j < e; j++) ffOutput[j] = ffOutput[j]! + ff2Bias[j]!;
    state = this.normalize(Float64Array.from(state, (x, i) => x + ffOutput[i]!), ffScale, ffBias);
    const head = this.vector("lmHead");
    const bias = this.vector("lmBias");
    const logits = new Float64Array(v);
    for (let i = 0; i < e; i++) for (let j = 0; j < v; j++) logits[j] = logits[j]! + state[i]! * head[i * v + j]!;
    for (let j = 0; j < v; j++) logits[j] = logits[j]! + bias[j]!;
    const max = Math.max(...logits);
    let sum = 0;
    for (let i = 0; i < v; i++) { logits[i] = Math.exp(logits[i]! - max); sum += logits[i]!; }
    for (let i = 0; i < v; i++) logits[i] = logits[i]! / sum;
    return logits;
  }

  public probabilities(context: readonly number[]): number[] {
    this.validateTokens(context);
    return Array.from(this.forward(context));
  }

  /**
   * Trains using reverse-mode analytical gradients through the complete
   * single-block Transformer. One example is accumulated and applied at a time.
   */
  public train(tokens: readonly number[], learningRate = 0.002, epochs = 1): TransformerMetrics {
    this.validateTokens(tokens);
    checkInteger(epochs, "epochs", 1, 1000);
    if (tokens.length < 2) throw new Error("At least two tokens are required.");
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.02) {
      throw new RangeError("learningRate must be in (0, 0.02].");
    }
    const examples = tokens.length - 1;
    let loss = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      loss = 0;
      for (let i = 1; i < tokens.length; i++) {
        const context = tokens.slice(Math.max(0, i - this.config.contextSize), i);
        const result = transformerLossAndGradients(this.config, this.weights, context, tokens[i]!);
        if (!Number.isFinite(result.loss) || result.gradients.some((gradient) => !Number.isFinite(gradient))) {
          throw new Error("Transformer training produced a non-finite loss or gradient.");
        }
        loss += result.loss;
        for (let j = 0; j < this.weights.length; j++) {
          this.weights[j] = this.weights[j]! - learningRate * result.gradients[j]!;
        }
      }
      loss /= examples;
    }
    return { loss, examples };
  }

  public generate(prompt: readonly number[], maxTokens: number, random: () => number = Math.random): number[] {
    this.validateTokens(prompt);
    checkInteger(maxTokens, "maxTokens", 0, 10000);
    const output = [...prompt];
    for (let i = 0; i < maxTokens; i++) {
      const probabilities = this.forward(output.slice(-this.config.contextSize));
      const draw = random();
      if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw new RangeError("Random draw must be in [0, 1).");
      let cumulative = 0;
      let selected = probabilities.length - 1;
      for (let token = 0; token < probabilities.length; token++) {
        cumulative += probabilities[token]!;
        if (draw < cumulative) { selected = token; break; }
      }
      output.push(selected);
    }
    return output;
  }

  public toCheckpoint(): TransformerCheckpoint {
    return { format: "nexus-transformer", version: 1, config: { ...this.config }, weights: Array.from(this.weights) };
  }

  public static fromCheckpoint(checkpoint: TransformerCheckpoint): TransformerModel {
    if (!checkpoint || checkpoint.format !== "nexus-transformer" || checkpoint.version !== 1 || !checkpoint.config) {
      throw new Error("Unsupported Transformer checkpoint.");
    }
    return new TransformerModel(checkpoint.config, checkpoint);
  }

  private validateTokens(tokens: readonly number[]): void {
    if (!Array.isArray(tokens) || tokens.some((token) => !Number.isSafeInteger(token) || token < 0 || token >= this.config.vocabSize)) {
      throw new RangeError("Input contains an invalid token ID.");
    }
  }
}
