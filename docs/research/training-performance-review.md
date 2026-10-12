# Training performance research notes

Review date: 2026-10-12

## External implementations reviewed

### karpathy/nanoGPT

- Repository: https://github.com/karpathy/nanoGPT
- License: MIT (see repository LICENSE).
- Relevant design lessons: contiguous token batches, vectorized tensor operations, a compact training loop, evaluation on held-out data, and benchmark-oriented iteration.
- Important caveat: nanoGPT is implemented in Python with PyTorch and optimized tensor kernels. Copying its training loop into this TypeScript scalar-autodiff implementation would not provide those vectorized kernels; a faithful port would be a separate backend project, not a safe drop-in patch.
- Status: use as architectural reference; no source code copied.

### karpathy/minGPT

- Repository: https://github.com/karpathy/minGPT
- License: MIT (see repository LICENSE).
- Relevant design lessons: minibatching across examples and sequence positions is a major efficiency lever; cleanly separate model, trainer, and data interfaces.
- Status: reference only; no source code copied.

### karpathy/build-nanogpt

- Repository: https://github.com/karpathy/build-nanogpt
- Relevant design lessons: fuse Q/K/V projection work, use vectorized attention kernels where available, and benchmark training steps rather than assuming an optimization helps.
- Status: reference only. Its PyTorch implementation and GPU-oriented optimizations do not transfer directly to Nexus's current TypeScript CPU reference path.

## Current Nexus bottleneck

`src/model/transformer-backprop.ts` constructs a scalar `Value` graph for each next-token target. The manifest trainer applies an optimizer update after processing a document. This is useful as an inspectable reference implementation, but graph allocation and reverse traversal per target are expensive. A major eventual speedup likely requires a fused numerical backend or manually derived gradients with finite-difference checks—not only smaller model settings.

## Implemented bounded optimization

The training workflow uses deterministic target stride 2 for the current bounded CPU run. Within each capped training sequence, it selects target positions at a fixed interval and always includes the final token. This roughly halves the number of gradient-graph builds for the selected targets, at the cost of fewer supervised positions per document. The full validation/test evaluator remains unstrided so held-out scores continue to cover all tokens.

The checkpoint provenance records `targetStride`, `maxTokensPerExample`, and `truncatedExamples`. Compare results only alongside those settings; do not claim this setting preserves model quality without a controlled comparison.

## Next high-value engineering step

Build a benchmark harness that reports targets per second, elapsed training time, parameter count, peak memory when measurable, and fixed-seed loss on a synthetic task. Then prototype a typed-array/manual-gradient backend behind the existing model interface. Keep the current autodiff implementation as the correctness oracle and require numerical-gradient checks before making a faster backend the default.
