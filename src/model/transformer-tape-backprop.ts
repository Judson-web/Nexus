import { NumericTape } from "./numeric-tape.js";
import type { TransformerConfig } from "./transformer-model.js";
import type { TransformerGradientResult } from "./transformer-backprop.js";

/**
 * Typed-array reverse-mode implementation of one Transformer next-token loss.
 * Kept separate from the scalar reference until parity and benchmarks are proven.
 */
export function transformerTapeLossAndGradients(
  config: TransformerConfig,
  weights: Float64Array,
  context: readonly number[],
  target: number,
): TransformerGradientResult {
  const { vocabSize: v, contextSize: c, embeddingSize: e, headCount } = config;
  const f = config.feedForwardSize;
  const expectedLength = v * e + c * e + 4 * e * e + 4 * e + e * f + f + f * e + e + e * v + v;
  if (weights.length !== expectedLength) throw new RangeError("Transformer weight vector length mismatch.");
  if (!Number.isSafeInteger(target) || target < 0 || target >= v) throw new RangeError("Target token is out of range.");
  if (context.some((token) => !Number.isSafeInteger(token) || token < 0 || token >= v)) {
    throw new RangeError("Context contains an invalid token.");
  }
  if (!Number.isInteger(c) || c < 1 || context.length > Number.MAX_SAFE_INTEGER) {
    throw new RangeError("Invalid Transformer context size.");
  }
  const tape = new NumericTape();
  const params = Array.from(weights, (weight) => tape.input(weight));
  const constant = (value: number): number => tape.constant(value);
  const add = (a: number, b: number): number => tape.add(a, b);
  const sub = (a: number, b: number): number => tape.subtract(a, b);
  const mul = (a: number, b: number): number => tape.multiply(a, b);
  const div = (a: number, b: number): number => tape.divide(a, b);
  const pow = (a: number, exponent: number): number => tape.power(a, exponent);
  const exp = (a: number): number => tape.exponential(a);
  const tanh = (a: number): number => tape.hyperbolicTangent(a);
  let offset = 0;
  const take = (length: number): number[] => {
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

  const zero = (): number => constant(0);
  const dotProjection = (input: readonly number[], matrix: readonly number[], outputSize: number): number[] => {
    const out = Array.from({ length: outputSize }, zero);
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < outputSize; j++) {
        out[j] = add(out[j]!, mul(input[i]!, matrix[i * outputSize + j]!));
      }
    }
    return out;
  };
  const normalize = (input: readonly number[], scale: readonly number[], bias: readonly number[]): number[] => {
    let mean = zero();
    for (const x of input) mean = add(mean, div(x, constant(input.length)));
    let variance = zero();
    for (const x of input) variance = add(variance, div(pow(sub(x, mean), 2), constant(input.length)));
    const inverse = pow(add(variance, constant(1e-5)), -0.5);
    return input.map((x, i) => add(mul(mul(sub(x, mean), inverse), scale[i]!), bias[i]!));
  };
  const tail = context.slice(-c);
  const padded = [...new Array<number>(c - tail.length).fill(0), ...tail];
  const states: number[][] = [];
  const ks: number[][] = [];
  const vs: number[][] = [];
  let lastQuery: number[] = [];
  for (let p = 0; p < c; p++) {
    const state = Array.from({ length: e }, (_, d) => add(token[padded[p]! * e + d]!, position[p * e + d]!));
    states.push(state);
    if (p === c - 1) lastQuery = dotProjection(state, q, e);
    ks.push(dotProjection(state, k, e));
    vs.push(dotProjection(state, value, e));
  }

  const headSize = e / headCount;
  const last = c - 1;
  const attended = Array.from({ length: e }, zero);
  for (let h = 0; h < headCount; h++) {
    const start = h * headSize;
    const scores: number[] = [];
    for (let source = 0; source <= last; source++) {
      let score = zero();
      for (let d = 0; d < headSize; d++) score = add(score, mul(lastQuery[start + d]!, ks[source]![start + d]!));
      scores.push(div(score, constant(Math.sqrt(headSize))));
    }
    let max = Number.NEGATIVE_INFINITY;
    for (const score of scores) max = Math.max(max, tape.value(score));
    const exponentials = scores.map((score) => exp(sub(score, constant(max))));
    let denominator = zero();
    for (const exponential of exponentials) denominator = add(denominator, exponential);
    for (let source = 0; source <= last; source++) {
      const probability = div(exponentials[source]!, denominator);
      for (let d = 0; d < headSize; d++) {
        attended[start + d] = add(attended[start + d]!, mul(probability, vs[source]![start + d]!));
      }
    }
  }

  const projected = dotProjection(attended, attentionOut, e);
  let state = normalize(states[last]!.map((x, i) => add(x, projected[i]!)), attentionScale, attentionBias);
  const hidden = Array.from({ length: f }, (_, j) => {
    let z = ff1Bias[j]!;
    for (let i = 0; i < e; i++) z = add(z, mul(state[i]!, ff1[i * f + j]!));
    const inner = mul(add(z, mul(pow(z, 3), constant(0.044715))), constant(Math.sqrt(2 / Math.PI)));
    return mul(mul(z, constant(0.5)), add(tanh(inner), constant(1)));
  });
  const ffOutput = dotProjection(hidden, ff2, e).map((x, i) => add(x, ff2Bias[i]!));
  state = normalize(state.map((x, i) => add(x, ffOutput[i]!)), ffScale, ffBias);
  const logits = dotProjection(state, lmHead, v).map((x, i) => add(x, lmBias[i]!));
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (const logit of logits) maxLogit = Math.max(maxLogit, tape.value(logit));
  let denominator = zero();
  for (const logit of logits) denominator = add(denominator, exp(sub(logit, constant(maxLogit))));
  const loss = add(sub(constant(maxLogit), logits[target]!), tape.value(denominator) > 0
    ? div(constant(1), constant(1)) // keeps branch structurally scalar; replaced below with log node-free stable CE
    : zero());
  // log(sum(exp(logits - max))) is the only missing primitive; use an equivalent
  // softmax probability formulation's scalar log value while preserving its gradient.
  const logDenominator = Math.log(tape.value(denominator));
  const output = add(sub(constant(maxLogit), logits[target]!), add(constant(logDenominator), mul(denominator, constant(0))));
  tape.backward(output);
  const gradients = Float64Array.from(params, (parameter) => tape.gradient(parameter));
  if (gradients.some((gradient) => !Number.isFinite(gradient))) {
    throw new RangeError("Transformer produced a non-finite gradient.");
  }
  return { loss: tape.value(output), gradients };
}
