# Typed-array gradient backend: implementation plan

## Goal

Replace the allocation-heavy scalar-object autodiff graph used by the Transformer trainer with a deterministic, contiguous numeric tape. Keep the existing scalar autodiff implementation as the correctness oracle until the new backend proves equivalent.

This is a training-kernel project, not a model-architecture change. Do not change tokenizer, data splits, optimizer defaults, checkpoint format, or production serving while introducing it.

## Current bottleneck

`src/model/transformer-backprop.ts` creates one `Value` object per scalar operation and per parameter, then `Value.backward()` allocates an object stack, a `Set<Value>`, and a topological-order array. The trainer repeats this for every sampled target. Target stride reduces graph count, but does not remove per-graph object allocation.

## Proposed design

Add a separate `NumericTape` implementation using structure-of-arrays storage:

- `Float64Array` for forward values and accumulated adjoints.
- Integer typed arrays for operation codes and parent node indices.
- Compact opcodes for add, subtract, multiply, divide, power, exp, tanh and the operations actually needed by the Transformer loss.
- Reverse traversal by descending node index when the tape guarantees append-only topological order; otherwise explicitly retain a validated topological schedule.
- Parameter nodes map directly to checkpoint weight indices, allowing gradients to be written into a contiguous gradient buffer without materializing `Value[]`.
- Reuse tape buffers between examples where safe, with an explicit reset boundary so gradients and stale graph nodes cannot leak across examples.

Avoid object-per-node wrappers in the hot path. A typed-array tape that still creates one JS object per operation would miss the main goal.

## Migration stages

1. **Characterize reference behavior.** Add deterministic small-model fixtures and record scalar loss, gradient vectors, parameter layout and optimizer updates.
2. **Tape primitives.** Test every opcode's forward value and reverse derivative, including shared parents, constants, repeated edges, zero/near-zero inputs, and non-finite rejection.
3. **Tiny graph parity.** Compare tape and reference outputs and every gradient component against the scalar autodiff engine within explicit absolute/relative tolerances.
4. **Transformer loss parity.** Run both implementations on the same tiny configurations and contexts; compare loss and all gradients, including finite-difference checks on selected parameters.
5. **Opt-in backend.** Add a trainer option defaulting to the reference engine. Run all existing tests against both paths. Do not switch defaults yet.
6. **Benchmark fairly.** On the same machine, corpus, seed, model config and sampled targets, report median wall time, targets/second, peak memory where available, final loss and gradient error across repeated runs.
7. **Promote only with evidence.** Require parity tests, the full CI suite, and a material measured throughput or memory improvement before changing the default. Preserve the reference path for debugging.

## Acceptance criteria

- No changes to the mathematical model or checkpoint weight ordering.
- Same deterministic sampled target indices and same optimizer update semantics.
- Finite-difference and reference-gradient tests pass.
- Full TypeScript checks, all tests, smoke train/generate and build pass.
- Benchmark report records Node version, platform, CPU, corpus SHA-256, model config, seed, optimizer, target stride, repeat count and both backend identifiers.
- No unsupported claim of speedup before results exist.

## Explicit non-goals

- No third-party inference API or training dependency.
- No deployment, migration of live data, or automatic replacement of existing checkpoints.
- No dropping the scalar reference implementation until the numeric backend is established.
- No copying external implementation code; use external projects only as documented research references.
