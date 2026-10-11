import { Value } from "./autodiff.js";
import { softmaxCrossEntropy } from "./autodiff-loss.js";
import type { TransformerConfig } from "./transformer-model.js";

export interface TransformerGradientResult {
  readonly loss: number;
  readonly gradients: number[];
}

/**
 * Computes exact reverse-mode gradients for one next-token training example.
 * The scalar graph mirrors the current single-block Transformer forward pass,
 * including causal attention, both layer norms, GELU, and the vocabulary head.
 */
export function transformerLossAndGradients(
  config: TransformerConfig,
  weights: ArrayLike<number>,
  context: readonly number[],
  target: number,
): TransformerGradientResult {
  const { vocabSize: v, contextSize: c, embeddingSize: e, headCount, feedForwardSize: f } = config;
  const sizes: Record<string, number> = {
    token: v * e, position: c * e, q: e * e, k: e * e, value: e * e, attentionOut: e * e,
    attentionNormScale: e, attentionNormBias: e, ff1: e * f, ff1Bias: f, ff2: f * e,
    ff2Bias: e, ffNormScale: e, ffNormBias: e, lmHead: e * v, lmBias: v,
  };
  const ranges: Record<string, { start: number; length: number }> = {};
  let offset = 0;
  for (const [name, length] of Object.entries(sizes)) {
    ranges[name] = { start: offset, length };
    offset += length;
  }
  if (weights.length !== offset) throw new RangeError("Transformer weight count mismatch.");
  if (context.length > c || context.some((token) => !Number.isSafeInteger(token) || token < 0 || token >= v)) {
    throw new RangeError("Invalid context for Transformer gradient.");
  }
  if (!Number.isSafeInteger(target) || target < 0 || target >= v) throw new RangeError("Invalid target token.");
  const parameters = Array.from(weights, (weight) => Value.of(weight));
  const get = (name: string, index: number): Value => {
    const range = ranges[name]!;
    return parameters[range.start + index]!;
  };
  const constant = (n: number): Value => Value.of(n);
  const add = (a: Value, b: Value): Value => a.add(b);
  const mul = (a: Value, b: Value): Value => a.mul(b);
  const div = (a: Value, b: number): Value => a.div(b);
  const vector = (length: number, fn: (i: number) => Value): Value[] => Array.from({ length }, (_, i) => fn(i));
  const normalize = (input: Value[], scaleName: string, biasName: string): Value[] => {
    let mean = constant(0);
    for (const x of input) mean = add(mean, x);
    mean = div(mean, input.length);
    let variance = constant(0);
    for (const x of input) variance = add(variance, x.sub(mean).pow(2));
    variance = div(variance, input.length);
    const inverse = variance.add(1e-5).pow(-0.5);
    return input.map((x, i) => mul(mul(x.sub(mean), inverse), get(scaleName, i)).add(get(biasName, i)));
  };
  const matVec = (input: Value[], name: string, outSize: number, inSize = input.length): Value[] =>
    vector(outSize, (j) => {
      let out = constant(0);
      for (let i = 0; i < inSize; i++) out = add(out, mul(input[i]!, get(name, i * outSize + j)));
      return out;
    });
  const padded = new Array<number>(c).fill(0);
  const tail = context.slice(-c);
  padded.splice(c - tail.length, tail.length, ...tail);
  const states: Value[][] = [];
  const qs: Value[][] = [];
  const ks: Value[][] = [];
  const vs: Value[][] = [];
  for (let p = 0; p < c; p++) {
    const state = vector(e, (d) => get("token", padded[p]! * e + d).add(get("position", p * e + d)));
    states.push(state);
    const project = (name: string) => matVec(state, name, e);
    qs.push(project("q"));
    ks.push(project("k"));
    vs.push(project("value"));
  }
  const attended: Value[][] = states.map(() => vector(e, () => constant(0)));
  const headSize = e / headCount;
  for (let p = 0; p < c; p++) {
    for (let h = 0; h < headCount; h++) {
      const start = h * headSize;
      const scores: Value[] = [];
      for (let source = 0; source <= p; source++) {
        let dot = constant(0);
        for (let d = 0; d < headSize; d++) dot = add(dot, mul(qs[p]![start + d]!, ks[source]![start + d]!));
        scores.push(div(dot, Math.sqrt(headSize)));
      }
      let maximum = Number.NEGATIVE_INFINITY;
      for (const score of scores) maximum = Math.max(maximum, score.data);
      const exp = scores.map((score) => score.sub(maximum).exp());
      let sum = constant(0);
      for (const value of exp) sum = add(sum, value);
      for (let source = 0; source <= p; source++) {
        const probability = exp[source]!.div(sum);
        for (let d = 0; d < headSize; d++) {
          attended[p]![start + d] = add(attended[p]![start + d]!, mul(probability, vs[source]![start + d]!));
        }
      }
    }
  }
  const last = c - 1;
  const projected = matVec(attended[last]!, "attentionOut", e);
  let state = normalize(states[last]!.map((x, i) => x.add(projected[i]!)), "attentionNormScale", "attentionNormBias");
  const hidden = vector(f, (j) => {
    let z = get("ff1Bias", j);
    for (let i = 0; i < e; i++) z = add(z, mul(state[i]!, get("ff1", i * f + j)));
    const inner = z.add(mul(z.pow(3), 0.044715)).mul(Math.sqrt(2 / Math.PI)).tanh();
    return mul(mul(z, 0.5), inner.add(1));
  });
  const ffOutput = vector(e, (j) => {
    let out = get("ff2Bias", j);
    for (let i = 0; i < f; i++) out = add(out, mul(hidden[i]!, get("ff2", i * e + j)));
    return out;
  });
  state = normalize(state.map((x, i) => x.add(ffOutput[i]!)), "ffNormScale", "ffNormBias");
  const logits = vector(v, (j) => {
    let out = get("lmBias", j);
    for (let i = 0; i < e; i++) out = add(out, mul(state[i]!, get("lmHead", i * v + j)));
    return out;
  });
  const loss = softmaxCrossEntropy(logits, target);
  loss.backward();
  return { loss: loss.data, gradients: parameters.map((parameter) => parameter.grad) };
}
