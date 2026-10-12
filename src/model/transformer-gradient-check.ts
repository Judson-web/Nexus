import { transformerLossAndGradients } from "./transformer-backprop.js";
import type { TransformerConfig } from "./transformer-model.js";

export interface GradientCheckResult {
  readonly checkedParameters: number;
  readonly maxAbsoluteError: number;
  readonly maxRelativeError: number;
  readonly failures: readonly {
    readonly index: number;
    readonly analytical: number;
    readonly numerical: number;
    readonly absoluteError: number;
    readonly relativeError: number;
  }[];
}

/**
 * Compare reverse-mode gradients with central finite differences for a bounded
 * set of parameters. Intended for tiny test configurations only: each checked
 * parameter requires two complete forward-loss evaluations.
 */
export function checkTransformerGradients(
  config: TransformerConfig,
  weights: Float64Array,
  context: readonly number[],
  target: number,
  options: { readonly indices?: readonly number[]; readonly epsilon?: number; readonly absoluteTolerance?: number; readonly relativeTolerance?: number } = {},
): GradientCheckResult {
  const epsilon = options.epsilon ?? 1e-5;
  const absoluteTolerance = options.absoluteTolerance ?? 2e-5;
  const relativeTolerance = options.relativeTolerance ?? 2e-3;
  if (!Number.isFinite(epsilon) || epsilon <= 0 || epsilon > 1e-2) throw new RangeError("epsilon must be in (0, 0.01].");
  if (!Number.isFinite(absoluteTolerance) || absoluteTolerance < 0) throw new RangeError("absoluteTolerance must be non-negative.");
  if (!Number.isFinite(relativeTolerance) || relativeTolerance < 0) throw new RangeError("relativeTolerance must be non-negative.");
  const indices = options.indices ?? Array.from({ length: Math.min(weights.length, 16) }, (_, i) =>
    weights.length <= 16 ? i : Math.floor(i * (weights.length - 1) / 15));
  if (indices.length > 64) throw new RangeError("Gradient checking is limited to 64 parameters per call.");
  if (new Set(indices).size !== indices.length || indices.some((index) => !Number.isSafeInteger(index) || index < 0 || index >= weights.length)) {
    throw new RangeError("Gradient-check indices must be unique valid parameter indices.");
  }
  const analytical = transformerLossAndGradients(config, weights, context, target).gradients;
  const failures: { index: number; analytical: number; numerical: number; absoluteError: number; relativeError: number }[] = [];
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  const working = new Float64Array(weights);
  for (const index of indices) {
    const original = working[index]!;
    working[index] = original + epsilon;
    const plus = transformerLossAndGradients(config, working, context, target).loss;
    working[index] = original - epsilon;
    const minus = transformerLossAndGradients(config, working, context, target).loss;
    working[index] = original;
    const numerical = (plus - minus) / (2 * epsilon);
    const expected = analytical[index]!;
    const absoluteError = Math.abs(expected - numerical);
    const relativeError = absoluteError / Math.max(1e-8, Math.abs(expected), Math.abs(numerical));
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError);
    maxRelativeError = Math.max(maxRelativeError, relativeError);
    if (absoluteError > absoluteTolerance && relativeError > relativeTolerance) {
      failures.push({ index, analytical: expected, numerical, absoluteError, relativeError });
    }
  }
  return { checkedParameters: indices.length, maxAbsoluteError, maxRelativeError, failures };
}
