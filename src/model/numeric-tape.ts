/**
 * Append-only reverse-mode numeric tape.
 *
 * Nodes are stored in structure-of-arrays typed buffers rather than one object
 * per scalar operation. Appends are topological, so reverse-mode propagation
 * walks nodes backward without allocating a graph, Set, or traversal stack.
 *
 * This is a primitive foundation; the Transformer still uses scalar autodiff
 * until full loss-and-gradient parity has been demonstrated.
 */
export const enum TapeOpcode {
  Input = 0,
  Add = 1,
  Subtract = 2,
  Multiply = 3,
  Divide = 4,
  Power = 5,
  Exponential = 6,
  Tanh = 7,
}

const INITIAL_CAPACITY = 1024;

export class NumericTape {
  private values = new Float64Array(INITIAL_CAPACITY);
  private gradients = new Float64Array(INITIAL_CAPACITY);
  private left = new Int32Array(INITIAL_CAPACITY);
  private right = new Int32Array(INITIAL_CAPACITY);
  private opcodes = new Uint8Array(INITIAL_CAPACITY);
  private auxiliary = new Float64Array(INITIAL_CAPACITY);
  private length = 0;

  public get size(): number {
    return this.length;
  }

  public value(node: number): number {
    this.assertNode(node);
    return this.values[node]!;
  }

  public gradient(node: number): number {
    this.assertNode(node);
    return this.gradients[node]!;
  }

  public reset(): void {
    this.length = 0;
  }

  public input(value: number): number {
    return this.append(TapeOpcode.Input, value, -1, -1, 0);
  }

  public constant(value: number): number {
    return this.input(value);
  }

  public add(a: number, b: number): number {
    this.assertNode(a);
    this.assertNode(b);
    return this.append(TapeOpcode.Add, this.values[a]! + this.values[b]!, a, b, 0);
  }

  public subtract(a: number, b: number): number {
    this.assertNode(a);
    this.assertNode(b);
    return this.append(TapeOpcode.Subtract, this.values[a]! - this.values[b]!, a, b, 0);
  }

  public multiply(a: number, b: number): number {
    this.assertNode(a);
    this.assertNode(b);
    return this.append(TapeOpcode.Multiply, this.values[a]! * this.values[b]!, a, b, 0);
  }

  public divide(a: number, b: number): number {
    this.assertNode(a);
    this.assertNode(b);
    if (this.values[b] === 0) throw new RangeError("Division by zero on numeric tape.");
    return this.append(TapeOpcode.Divide, this.values[a]! / this.values[b]!, a, b, 0);
  }

  public power(a: number, exponent: number): number {
    this.assertNode(a);
    if (!Number.isFinite(exponent) || (this.values[a]! < 0 && !Number.isInteger(exponent))) {
      throw new RangeError("Invalid exponent on numeric tape.");
    }
    const value = this.values[a]! ** exponent;
    const derivative = exponent === 0 ? 0 : exponent * this.values[a]! ** (exponent - 1);
    return this.append(TapeOpcode.Power, value, a, -1, derivative);
  }

  public exponential(a: number): number {
    this.assertNode(a);
    const value = Math.exp(this.values[a]!);
    if (!Number.isFinite(value)) throw new RangeError("Numeric tape exponential overflow.");
    return this.append(TapeOpcode.Exponential, value, a, -1, value);
  }

  public hyperbolicTangent(a: number): number {
    this.assertNode(a);
    const value = Math.tanh(this.values[a]!);
    return this.append(TapeOpcode.Tanh, value, a, -1, 1 - value * value);
  }

  /** Accumulate reverse-mode derivatives from the selected scalar output. */
  public backward(output: number, seed = 1): void {
    this.assertNode(output);
    if (!Number.isFinite(seed)) throw new RangeError("Gradient seed must be finite.");
    this.gradients.fill(0, 0, this.length);
    this.gradients[output] = seed;

    for (let node = this.length - 1; node >= 0; node--) {
      const gradient = this.gradients[node]!;
      if (gradient === 0) continue;
      const a = this.left[node]!;
      const b = this.right[node]!;
      switch (this.opcodes[node]) {
        case TapeOpcode.Input:
          break;
        case TapeOpcode.Add:
          this.gradients[a] = this.gradients[a]! + gradient;
          this.gradients[b] = this.gradients[b]! + gradient;
          break;
        case TapeOpcode.Subtract:
          this.gradients[a] = this.gradients[a]! + gradient;
          this.gradients[b] = this.gradients[b]! - gradient;
          break;
        case TapeOpcode.Multiply:
          this.gradients[a] = this.gradients[a]! + gradient * this.values[b]!;
          this.gradients[b] = this.gradients[b]! + gradient * this.values[a]!;
          break;
        case TapeOpcode.Divide: {
          const denominator = this.values[b]!;
          this.gradients[a] = this.gradients[a]! + gradient / denominator;
          this.gradients[b] = this.gradients[b]! - gradient * this.values[a]! / (denominator * denominator);
          break;
        }
        case TapeOpcode.Power:
        case TapeOpcode.Exponential:
        case TapeOpcode.Tanh:
          this.gradients[a] = this.gradients[a]! + gradient * this.auxiliary[node]!;
          break;
        default:
          throw new Error("Unknown numeric tape opcode.");
      }
    }

    for (let i = 0; i < this.length; i++) {
      if (!Number.isFinite(this.gradients[i]!)) throw new RangeError("Numeric tape produced a non-finite gradient.");
    }
  }

  private append(opcode: TapeOpcode, value: number, a: number, b: number, auxiliary: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(auxiliary)) {
      throw new RangeError("Numeric tape values and derivatives must be finite.");
    }
    this.ensureCapacity(this.length + 1);
    const node = this.length++;
    this.values[node] = value;
    this.gradients[node] = 0;
    this.left[node] = a;
    this.right[node] = b;
    this.opcodes[node] = opcode;
    this.auxiliary[node] = auxiliary;
    return node;
  }

  private ensureCapacity(required: number): void {
    if (required <= this.values.length) return;
    let capacity = this.values.length;
    while (capacity < required) capacity *= 2;
    const copy = <T extends Float64Array | Int32Array | Uint8Array>(buffer: T): T => {
      const next = new (buffer.constructor as { new(length: number): T })(capacity);
      next.set(buffer);
      return next;
    };
    this.values = copy(this.values);
    this.gradients = copy(this.gradients);
    this.left = copy(this.left);
    this.right = copy(this.right);
    this.opcodes = copy(this.opcodes);
    this.auxiliary = copy(this.auxiliary);
  }

  private assertNode(node: number): void {
    if (!Number.isSafeInteger(node) || node < 0 || node >= this.length) {
      throw new RangeError("Numeric tape node index is out of range.");
    }
  }
}
