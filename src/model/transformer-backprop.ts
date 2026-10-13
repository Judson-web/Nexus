import { Value } from "./autodiff.js";
import { softmaxCrossEntropy } from "./autodiff-loss.js";
import type { TransformerConfig } from "./transformer-model.js";

export interface TransformerGradientResult {
  readonly loss: number;
  readonly gradients: Float64Array;
}

/** Exact reverse-mode gradients for the configured decoder stack. */
export function transformerLossAndGradients(
  config: TransformerConfig,
  weights: Float64Array,
  context: readonly number[],
  target: number,
): TransformerGradientResult {
  const { vocabSize: v, contextSize: c, embeddingSize: e, headCount } = config;
  const f = config.feedForwardSize;
  const layerCount = config.layerCount ?? 1;
  const perLayer = 4 * e * e + 5 * e + 2 * e * f + f;
  const expectedLength = v * e + c * e + layerCount * perLayer + e * v + v;
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
  interface LayerParameters {
    q: Value[]; k: Value[]; value: Value[]; attentionOut: Value[];
    attentionScale: Value[]; attentionBias: Value[];
    ff1: Value[]; ff1Bias: Value[]; ff2: Value[]; ff2Bias: Value[];
    ffScale: Value[]; ffBias: Value[];
  }
  const layers: LayerParameters[] = [];
  for (let layer = 0; layer < layerCount; layer++) {
    layers.push({
      q: take(e * e), k: take(e * e), value: take(e * e), attentionOut: take(e * e),
      attentionScale: take(e), attentionBias: take(e),
      ff1: take(e * f), ff1Bias: take(f), ff2: take(f * e), ff2Bias: take(e),
      ffScale: take(e), ffBias: take(e),
    });
  }
  const lmHead = take(e * v);
  const lmBias = take(v);
  if (offset !== params.length) throw new Error("Internal Transformer parameter layout mismatch.");

  const zero = (): Value => Value.of(0);
  const dotProjection = (input: readonly Value[], matrix: readonly Value[], outputSize: number): Value[] => {
    const out = Array.from({ length: outputSize }, zero);
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < outputSize; j++) out[j] = out[j]!.add(input[i]!.mul(matrix[i * outputSize + j]!));
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
  let states: Value[][] = [];
  for (let p = 0; p < c; p++) {
    states.push(Array.from({ length: e }, (_, d) => token[padded[p]! * e + d]!.add(position[p * e + d]!)));
  }
  const headSize = e / headCount;
  for (const layer of layers) {
    const qs = states.map((state) => dotProjection(state, layer.q, e));
    const ks = states.map((state) => dotProjection(state, layer.k, e));
    const vs = states.map((state) => dotProjection(state, layer.value, e));
    const nextStates: Value[][] = [];
    for (let p = 0; p < c; p++) {
      const attended = Array.from({ length: e }, zero);
      for (let h = 0; h < headCount; h++) {
        const start = h * headSize;
        const scores: Value[] = [];
        for (let source = 0; source <= p; source++) {
          let score = zero();
          for (let d = 0; d < headSize; d++) score = score.add(qs[p]![start + d]!.mul(ks[source]![start + d]!));
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
            attended[start + d] = attended[start + d]!.add(probability.mul(vs[source]![start + d]!));
          }
        }
      }
      const projected = dotProjection(attended, layer.attentionOut, e);
      let state = normalize(states[p]!.map((x, i) => x.add(projected[i]!)), layer.attentionScale, layer.attentionBias);
      const hidden = Array.from({ length: f }, (_, j) => {
        let z = layer.ff1Bias[j]!;
        for (let i = 0; i < e; i++) z = z.add(state[i]!.mul(layer.ff1[i * f + j]!));
        const inner = z.add(z.pow(3).mul(0.044715)).mul(Math.sqrt(2 / Math.PI));
        return z.mul(0.5).mul(inner.tanh().add(1));
      });
      const ffOutput = dotProjection(hidden, layer.ff2, e).map((x, i) => x.add(layer.ff2Bias[i]!));
      state = normalize(state.map((x, i) => x.add(ffOutput[i]!)), layer.ffScale, layer.ffBias);
      nextStates.push(state);
    }
    states = nextStates;
  }
  const logits = dotProjection(states[c - 1]!, lmHead, v).map((x, i) => x.add(lmBias[i]!));
  const loss = softmaxCrossEntropy(logits, target);
  loss.backward();
  const gradients = Float64Array.from(params, (parameter) => parameter.grad);
  if (gradients.some((gradient) => !Number.isFinite(gradient))) throw new RangeError("Transformer produced a non-finite gradient.");
  return { loss: loss.data, gradients };
}
