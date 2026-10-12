import { Value } from "./autodiff.js";
import { softmaxCrossEntropy } from "./autodiff-loss.js";
import type { TransformerConfig } from "./transformer-model.js";

export interface TransformerGradientResult {
  readonly loss: number;
  readonly gradients: Float64Array;
}

/**
 * Builds the full scalar computation graph for one next-token example and
 * returns exact reverse-mode gradients for every parameter in checkpoint order.
 * This is intentionally a tiny CPU reference path; it favors inspectability
 * over throughput and is not intended for large corpora or production serving.
 */
export function transformerLossAndGradients(
  config: TransformerConfig,
  weights: Float64Array,
  context: readonly number[],
  target: number,
): TransformerGradientResult {
  const { vocabSize: v, contextSize: c, embeddingSize: e, headCount } = config;
  const f = config.feedForwardSize;
  const expectedLength = v * e + c * e + 4 * e * e + 2 * e + e * f + f + f * e + e + e * v + v;
  if (weights.length !== expectedLength) throw new RangeError("Transformer weight vector length mismatch.");
  if (!Number.isSafeInteger(target) || target < 0 || target >= v) throw new RangeError("Target token is out of range.");
  if (context.some((token) => !Number.isSafeInteger(token) || token < 0 || token >= v)) throw new RangeError("Context contains an invalid token.");
  const params = Array.from(weights, (weight) => Value.of(weight));
  let offset = 0;
  const take = (length: number): Value[] => {
    const result = params.slice(offset, offset + length);
    offset += length;
    return result;
  };
  const token = take(v * e);
  const position = take(c * e);
  const q = take(e * e);
  const k = take(e * e);
  const value = take(e * e);
  const attentionOut = take(e * e);
  const attentionScale = take(e);
  const attentionBias = take(e);
  const ff1 = take(e * f);
  const ff1Bias = take(f);
  const ff2 = take(f * e);
  const ff2Bias = take(e);
  const ffScale = take(e);
  const ffBias = take(e);
  const lmHead = take(e * v);
  const lmBias = take(v);
  if (offset !== params.length) throw new Error("Internal Transformer parameter layout mismatch.");

  const zero = (): Value => Value.of(0);
  const add = (a: Value, b: Value): Value => a.add(b);
  const dotProjection = (input: readonly Value[], matrix: readonly Value[], outputSize: number): Value[] => {
    const out = Array.from({ length: outputSize }, zero);
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < outputSize; j++) {
        out[j] = out[j]!.add(input[i]!.mul(matrix[i * outputSize + j]!));
      }
    }
    return out;
  };
  const normalize = (input: readonly Value[], scale: readonly Value[], bias: readonly Value[]): Value[] => {
    let mean = zero();
    for (const x of input) mean = mean.add(x.div(input.length));
    let variance = zero();
    for (const x of input) variance = variance.add(x.sub(mean).pow(2).div(input.length));
    const inverse = variance.add(1e-5).pow(-0.5);
    return input.map((x, i) => x.sub(mean).mul(inverse).mul(scale[i]!).add(bias[i]!));
  };
  const tail = context.slice(-c);
  const padded = [...new Array<number>(c - tail.length).fill(0), ...tail];
  const states: Value[][] = [];
  const qs: Value[][] = [];
  const ks: Value[][] = [];
  const vs: Value[][] = [];
  for (let p = 0; p < c; p++) {
    const state = Array.from({ length: e }, (_, d) => token[padded[p]! * e + d]!.add(position[p * e + d]!));
    states.push(state);
    qs.push(dotProjection(state, q, e));
    ks.push(dotProjection(state, k, e));
    vs.push(dotProjection(state, value, e));
  }

  const headSize = e / headCount;
  const attended: Value[][] = states.map(() => Array.from({ length: e }, zero));
  for (let p = 0; p < c; p++) {
    for (let h = 0; h < headCount; h++) {
      const start = h * headSize;
      const scores: Value[] = [];
      for (let source = 0; source <= p; source++) {
        let score = zero();
        for (let d = 0; d < headSize; d++) {
          score = score.add(qs[p]![start + d]!.mul(ks[source]![start + d]!));
        }
        scores.push(score.div(Math.sqrt(headSize)));
      }
      let max = Number.NEGATIVE_INFINITY;
      for (const score of scores) max = Math.max(max, score.data);
      const exponentials = scores.map((score) => score.sub(max).exp());
      let denominator = zero();
      for (const exponential of exponentials) denominator = denominator.add(exponential);
      for (let source = 0; source <= p; source++) {
        const probability = exponentials[source]!.div(denominator);
        for (let d = 0; d < headSize; d++) {
          attended[p]![start + d] = attended[p]![start + d]!.add(probability.mul(vs[source]![start + d]!));
        }
      }
    }
  }

  const last = c - 1;
  const projected = dotProjection(attended[last]!, attentionOut, e);
  let state = normalize(states[last]!.map((x, i) => x.add(projected[i]!)), attentionScale, attentionBias);
  const hidden = Array.from({ length: f }, (_, j) => {
    let z = ff1Bias[j]!;
    for (let i = 0; i < e; i++) z = z.add(state[i]!.mul(ff1[i * f + j]!));
    const inner = z.add(z.pow(3).mul(0.044715)).mul(Math.sqrt(2 / Math.PI));
    return z.mul(0.5).mul(inner.tanh().add(1));
  });
  const ffOutput = dotProjection(hidden, ff2, e).map((x, i) => x.add(ff2Bias[i]!));
  state = normalize(state.map((x, i) => x.add(ffOutput[i]!)), ffScale, ffBias);
  const logits = dotProjection(state, lmHead, v).map((x, i) => x.add(lmBias[i]!));
  const loss = softmaxCrossEntropy(logits, target);
  loss.backward();
  const gradients = Float64Array.from(params, (parameter) => parameter.grad);
  if (gradients.some((gradient) => !Number.isFinite(gradient))) throw new RangeError("Transformer produced a non-finite gradient.");
  return { loss: loss.data, gradients };
}
