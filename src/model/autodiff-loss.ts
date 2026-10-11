import { Value } from "./autodiff.js";

/**
 * Numerically stable cross-entropy from logits, built from differentiable
 * scalar operations. The caller supplies raw, finite logits and the target
 * class index. Softmax is implicit in log-sum-exp, avoiding probability
 * underflow before the logarithm.
 */
export function softmaxCrossEntropy(logits: readonly Value[], target: number): Value {
  if (logits.length < 2) throw new RangeError("At least two logits are required.");
  if (!Number.isSafeInteger(target) || target < 0 || target >= logits.length) {
    throw new RangeError("Target index is outside the logits range.");
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (const logit of logits) {
    if (!Number.isFinite(logit.data)) throw new RangeError("Logits must be finite.");
    maximum = Math.max(maximum, logit.data);
  }
  // Subtract a constant maximum for numerical stability. The maximum is treated
  // as a constant shift; log-softmax is invariant to this shift, so its
  // derivative remains exactly the usual softmax-minus-one-hot gradient.
  let exponentialSum = Value.of(0);
  for (const logit of logits) exponentialSum = exponentialSum.add(logit.sub(maximum).exp());
  return exponentialSum.log().sub(logits[target]!.sub(maximum));
}
