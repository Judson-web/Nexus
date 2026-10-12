import assert from "node:assert/strict";
import test from "node:test";
import { Value } from "./autodiff.js";

test("reverse mode accumulates gradients across shared graph nodes", () => {
  const x = Value.of(3);
  const y = x.mul(x).add(x.mul(2));
  y.backward();
  assert.equal(y.data, 15);
  assert.equal(x.grad, 8);
});

test("numeric constants do not change gradients for scalar arithmetic", () => {
  const x = Value.of(4);
  const y = x.add(3).sub(2).mul(5).div(2);
  y.backward();
  assert.equal(y.data, 12.5);
  assert.equal(x.grad, 2.5);
});

test("value operands retain gradients on both sides of arithmetic", () => {
  const x = Value.of(6);
  const y = Value.of(3);
  const result = x.div(y).add(x.mul(y)).sub(y);
  result.backward();
  assert.equal(result.data, 17);
  assert.equal(x.grad, 1 / 3 + 3);
  assert.equal(y.grad, -6 / 9 + 6 - 1);
});

test("reverse mode differentiates composite smooth expressions", () => {
  const x = Value.of(1.7);
  const y = x.pow(3).div(2).add(x.exp()).sub(x.tanh());
  y.backward();
  const expected = (3 * 1.7 ** 2) / 2 + Math.exp(1.7) - (1 - Math.tanh(1.7) ** 2);
  assert.ok(Math.abs(x.grad - expected) < 1e-10);
});

test("reverse mode matches central finite differences on a scalar objective", () => {
  const objective = (x: number): number => Math.exp(x * 0.7) + (x - 2) ** 4;
  const x = Value.of(0.4);
  const y = x.mul(0.7).exp().add(x.sub(2).pow(4));
  y.backward();
  const epsilon = 1e-5;
  const numerical = (objective(0.4 + epsilon) - objective(0.4 - epsilon)) / (2 * epsilon);
  assert.ok(Math.abs(x.grad - numerical) < 1e-5);
});

test("autodiff rejects invalid operations and non-finite operands", () => {
  assert.throws(() => Value.of(1).div(0), /division by zero/i);
  assert.throws(() => Value.of(0).log(), /positive/i);
  assert.throws(() => Value.of(1000).exp(), /overflow/i);
  assert.throws(() => Value.of(Number.NaN), /finite/i);
  assert.throws(() => Value.of(1).add(Number.NaN), /finite/i);
  assert.throws(() => Value.of(1).mul(Number.POSITIVE_INFINITY), /finite/i);
});

test("reverse mode handles deep graphs without recursive call-stack overflow", () => {
  const x = Value.of(1);
  let result = x;
  for (let i = 0; i < 20_000; i += 1) result = result.add(1);
  result.backward();
  assert.equal(result.data, 20_001);
  assert.equal(x.grad, 1);
});
