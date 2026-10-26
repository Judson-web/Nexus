# Open LLM training and data reuse review

Review date: 2026-10-26

## Repositories reviewed

### `karpathy/nanochat` (MIT)

- Repository: https://github.com/karpathy/nanochat
- License: MIT. Preserve its copyright and license text if substantial source code is copied into a derivative.
- Relevant components: separate base-pretraining and supervised-fine-tuning (SFT) stages, response-aware chat evaluation, dataset/task adapters, and `tasks/smoltalk.py` for Smol-SmolTalk.
- Nexus adaptation: keep the conversational prompt as context and apply training loss to the assistant continuation, not the user prompt. This principle is now implemented as an independent TypeScript adapter; the PyTorch implementation was not copied.

### `karpathy/nanoGPT` (MIT)

- Repository: https://github.com/karpathy/nanoGPT
- License: MIT.
- Useful lessons: simple reproducible training loops, explicit held-out evaluation, and keeping experiments inspectable.
- Limitation: nanoGPT relies on PyTorch tensor kernels. Copying its model loop into Nexus's scalar TypeScript autodiff would not provide equivalent throughput. A tensorized backend is a separate engineering task.

### `Lightning-AI/litgpt` (Apache-2.0)

- Repository: https://github.com/Lightning-AI/litgpt
- License: Apache-2.0.
- Useful lessons: explicit pretraining, fine-tuning, evaluation and deployment recipes; implementation patterns for checkpoint conversion, memory-aware runs, and controlled experiments.
- Limitation: its optimized implementation depends on Python/PyTorch and accelerator-oriented kernels. Reusing the recipes is practical; transplanting them into Nexus's scalar TypeScript autodiff is not.

### `karpathy/llm.c` (MIT)

- Repository: https://github.com/karpathy/llm.c
- License: MIT.
- Useful lessons: a readable C CPU reference path, CUDA training kernels, small debug-state fixtures and tests comparing logits/loss/gradients against a framework implementation. These are good references for a future native or WebGPU compute backend.
- Limitation: the performant path is CUDA/C and the CPU path is explicitly a teaching/reference implementation. Porting its numerical kernels or a compatible architecture is a separate project. Preserve the MIT notice if source code is copied.

### Optional local inference baseline: SmolLM2-135M-Instruct (Apache-2.0)

- Model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- The official card declares Apache-2.0 and documents a `@huggingface/transformers` / Transformers.js path for local text generation. This could provide an on-device, no-inference-API comparison baseline while Nexus's from-scratch model remains research-only.
- This is a different track from training Nexus from scratch: it reuses a pretrained checkpoint, tokenizer, and chat template. Keep the model ID, pinned revision, license, weight checksums, hardware requirements, and user-data boundary explicit. Evaluate browser WebGPU/CPU memory and cold-start behavior before choosing it for the deployed app.

## Dataset review

### OpenAssistant OASST1 (Apache-2.0)

- Official card: https://huggingface.co/datasets/OpenAssistant/oasst1
- Keep it as the first human-annotated conversation source. Nexus already pins the revision and license, filters English and high-risk/personal-information labels where available, splits at conversation-tree level, and records source hashes and attribution.
- The first 500-tree cap yielded 1,079 accepted pairs. This is a useful smoke corpus, not sufficient evidence of general-purpose conversational ability.

### Smol-SmolTalk and SmolTalk

- Official cards: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk and https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Smol-SmolTalk is targeted at models below 1B parameters; its card reports 460,341 training rows and 24,229 test rows and declares Apache-2.0. It contains a `source` field with multiple source classes, so the entire blend must **not** be blindly treated as one license. The adapter now accepts alternating user/assistant multi-turn histories with an optional leading system instruction and trains only against the final assistant reply, using earlier turns as context.
- The SmolTalk card states that new subsets Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, and Smol-summarize are Apache-2.0, while existing public datasets keep their original licenses.
- Follow-up ingestion should pin an immutable Hub revision and filter to the concise Smol-Magpie-Ultra source (`smol-magpie-ultra-short`) or another individually verified Apache-2.0 subset. Preserve source IDs, content hashes, source/test separation and license evidence. Do not use the `all` mixture by default.

### Explicitly excluded pending better license evidence

- `HuggingFaceH4/no_robots`: the official card declares CC-BY-NC-4.0. Exclude from a project that may be used commercially unless the project deliberately accepts a non-commercial restriction.
- `Open-Orca/OpenOrca`, the `all` SmolTalk mixtures, and mixed-source OpenHermes derivatives: constituents have differing or unclear source terms. Review source-level records rather than assuming the wrapper repository's license overrides upstream terms.

## Inference-format contract

Manifest-trained checkpoints record their serving format in the checksum-covered checkpoint JSON. Chat-SFT checkpoints use `nexus-chat-sft-v1`; the CLI and local API wrap a plain user prompt as `User: … Assistant: ` and return only the newly generated continuation. Older checkpoints without this field default to legacy causal-language-model behavior. The format tag is validated at verified load time rather than being inferred from prompt strings at runtime.

## Training objective change

Nexus previously serialized conversations as `User: … Assistant: …` but applied next-token loss to the entire sequence. That spent scarce CPU gradient work teaching the model to predict the user prompt as well as the answer. The trainer now encodes prompt context separately and applies loss only to assistant continuation tokens and EOS for user/assistant pairs. Plain-text examples keep the normal full-sequence next-token objective.

Held-out loss/perplexity now measures assistant continuation targets for conversation pairs; these scores are **not directly comparable** with earlier full-sequence metrics. The objective is recorded in provenance and resumable-state metadata. A resume checkpoint trained under the legacy objective is rejected so the run cannot silently mix training semantics.

## Next steps

1. Run the masked-target unit tests, CI, and the bounded corpus workflow; inspect continuation samples and response-only held-out metrics.
2. Add a separate immutable SmolTalk adapter filtered to explicitly Apache-2.0 subsets, then compare OASST-only, Smol-only, and deliberately mixed runs on the same evaluation protocol.
3. Replace scalar-per-target reverse-mode graph training with a tested batched tensor backend before scaling corpus size or model width.
4. Keep the production model unconfigured until response quality, safety, held-out benchmarks, and checkpoint integrity all pass.

## Sources

- https://github.com/karpathy/nanochat/blob/master/LICENSE
- https://github.com/karpathy/nanochat/blob/master/scripts/chat_sft.py
- https://github.com/karpathy/nanochat/blob/master/tasks/smoltalk.py
- https://github.com/karpathy/nanoGPT
- https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/HuggingFaceH4/no_robots
