import assert from "node:assert/strict";
import test from "node:test";
import { NumericTape } from "./numeric-tape.js";

test("numeric tape computes values and gradients through shared parents", () => {
  const tape = new NumericTape();
  const x = tape.input(3);
  const twice = tape.add(x, x);
  const square = tape.multiply(twice, twice);
  const output = tape.add(square, tape.constant(2));
  assert.equal(tape.value(output), 38);
  tape.backward(output);
  assert.equal(tape.gradient(x), 24);
  assert.equal(tape.gradient(twice), 12);
});

test("numeric tape supports arithmetic and unary derivatives", () => {
  const tape = new NumericTape();
  const x = tape.input(2);
  const y = tape.input(4);
  const difference = tape.subtract(y, x);
  const ratio = tape.divide(difference, y);
  const squared = tape.power(ratio, 2);
  const exp = tape.exponential(x);
  const tanh = tape.hyperbolicTangent(y);
  const output = tape.add(tape.add(squared, exp), tanh);
  tape.backward(output);
  assert.ok(Math.abs(tape.gradient(x) - (Math.exp(2) - 0.25)) < 1e-10);
  assert.ok(Math.abs(tape.gradient(y) - (0.125 + (1 - Math.tanh(4) ** 2))) < 1e-10);
});

test("numeric tape resets between graphs and clears gradients on each backward pass", () => {
  const tape = new NumericTape();
  const x = tape.input(5);
  const y = tape.multiply(x, x);
  tape.backward(y);
  assert.equal(tape.gradient(x), 10);
  tape.backward(x);
  assert.equal(tape.gradient(x), 1);
  tape.reset();
  assert.equal(tape.size, 0);
  const next = tape.input(7);
  const out = tape.power(next, 2);
  tape.backward(out);
  assert.equal(tape.gradient(next), 14);
});

test("numeric tape validates indices, division, exponents, and finite gradients", () => {
  const tape = new NumericTape();
  assert.throws(() => tape.value(0), /index/i);
  const x = tape.input(1);
  const zero = tape.constant(0);
  assert.throws(() => tape.divide(x, zero), /division by zero/i);
  assert.throws(() => tape.power(x, Number.NaN), /exponent/i);
  assert.throws(() => tape.backward(99), /index/i);
  const invalid = tape.input(1e308);
  assert.throws(() => tape.exponential(invalid), /overflow/i);
});

test("numeric tape grows typed buffers without losing node references", () => {
  const tape = new NumericTape();
  const x = tape.input(1.5);
  let current = x;
  for (let i = 0; i < 3000; i++) current = tape.add(current, x);
  tape.backward(current);
  assert.equal(tape.size, 3001);
  assert.equal(tape.value(current), 4501.5);
  assert.equal(tape.gradient(x), 3001);
});
