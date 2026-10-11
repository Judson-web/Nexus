/**
 * Small reverse-mode automatic differentiation engine for scalar research
 * workloads. Each value records its parents and local derivatives; backward()
 * accumulates gradients in reverse topological order.
 *
 * This engine is intentionally minimal and dependency-free. It is a building
 * block for replacing SPSA with exact gradients in the Transformer, not a claim
 * that the existing Transformer trainer already uses backpropagation.
 */
export class Value {
  public grad = 0;
  private readonly parents: readonly { value: Value; derivative: number }[];

  private constructor(public readonly data: number, parents: readonly { value: Value; derivative: number }[] = []) {
    if (!Number.isFinite(data)) throw new RangeError("Autodiff values must be finite.");
    this.parents = parents;
  }

  public static of(data: number): Value {
    return new Value(data);
  }

  public add(other: Value | number): Value {
    const rhs = asValue(other);
    return new Value(this.data + rhs.data, [
      { value: this, derivative: 1 },
      { value: rhs, derivative: 1 },
    ]);
  }

  public sub(other: Value | number): Value {
    const rhs = asValue(other);
    return new Value(this.data - rhs.data, [
      { value: this, derivative: 1 },
      { value: rhs, derivative: -1 },
    ]);
  }

  public mul(other: Value | number): Value {
    const rhs = asValue(other);
    return new Value(this.data * rhs.data, [
      { value: this, derivative: rhs.data },
      { value: rhs, derivative: this.data },
    ]);
  }

  public div(other: Value | number): Value {
    const rhs = asValue(other);
    if (rhs.data === 0) throw new RangeError("Division by zero in autodiff graph.");
    return new Value(this.data / rhs.data, [
      { value: this, derivative: 1 / rhs.data },
      { value: rhs, derivative: -this.data / (rhs.data * rhs.data) },
    ]);
  }

  public pow(exponent: number): Value {
    if (!Number.isFinite(exponent) || (this.data < 0 && !Number.isInteger(exponent))) {
      throw new RangeError("Invalid exponent for autodiff power.");
    }
    const data = this.data ** exponent;
    const derivative = exponent === 0 ? 0 : exponent * this.data ** (exponent - 1);
    return new Value(data, [{ value: this, derivative }]);
  }

  public exp(): Value {
    const data = Math.exp(this.data);
    if (!Number.isFinite(data)) throw new RangeError("Autodiff exponential overflow.");
    return new Value(data, [{ value: this, derivative: data }]);
  }

  public log(): Value {
    if (this.data <= 0) throw new RangeError("Autodiff logarithm requires a positive value.");
    return new Value(Math.log(this.data), [{ value: this, derivative: 1 / this.data }]);
  }

  public tanh(): Value {
    const data = Math.tanh(this.data);
    return new Value(data, [{ value: this, derivative: 1 - data * data }]);
  }

  public relu(): Value {
    return new Value(Math.max(0, this.data), [{ value: this, derivative: this.data > 0 ? 1 : 0 }]);
  }

  public backward(seed = 1): void {
    if (!Number.isFinite(seed)) throw new RangeError("Gradient seed must be finite.");
    const order: Value[] = [];
    const visited = new Set<Value>();
    const visit = (node: Value): void => {
      if (visited.has(node)) return;
      visited.add(node);
      for (const parent of node.parents) visit(parent.value);
      order.push(node);
    };
    visit(this);
    for (const node of order) node.grad = 0;
    this.grad = seed;
    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i]!;
      for (const parent of node.parents) {
        parent.value.grad += node.grad * parent.derivative;
      }
    }
    if (order.some((node) => !Number.isFinite(node.grad))) {
      throw new RangeError("Autodiff produced a non-finite gradient.");
    }
  }
}

function asValue(value: Value | number): Value {
  if (value instanceof Value) return value;
  if (!Number.isFinite(value)) throw new RangeError("Autodiff operands must be finite.");
  return Value.of(value);
}
