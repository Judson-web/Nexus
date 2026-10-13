# Nexus English-language curriculum

This is a training plan for the in-house model, not a claim that the current checkpoint understands English. Keep all candidate checkpoints separate from production until blind held-out generation review passes.

## Data policy

Every source must have recorded provenance, an explicit license or public-domain basis, a source hash, and a source-level split. Do not scrape arbitrary web pages or mix validation/test passages into training. Prefer clear, varied English examples over simply increasing the number of repeated paragraphs.

## Curriculum stages

1. **Sentence foundations:** clean English sentences covering common words, punctuation, articles, pronouns, prepositions, singular/plural agreement, verb tense, negation, and questions. Use licensed/public-domain sources or manually authored examples. Deduplicate and keep whole sources separated across splits.
2. **Everyday prose and dialogue:** varied sentence lengths, contractions, turns of dialogue, instructions, descriptions, and explanations. Preserve paragraph boundaries when preparing sequences.
3. **Paragraph coherence:** stories and expository writing with references across sentences, transitions, cause/effect, and consistent subjects. Longer context is required here.
4. **Reading and explanation:** short passage followed by a grounded summary or explanation. Treat this as a later supervised objective; don't assume next-token pretraining alone teaches instruction following.
5. **Refinement:** balanced mixture of the earlier stages with held-out evaluation and generation samples after each change.

## Evaluation gate

Maintain a fixed, source-disjoint evaluation set that never enters the tokenizer or training process. Score:
- grammar and sentence completion on unfamiliar prompts;
- coherence across at least 3-5 generated sentences;
- spelling, punctuation, and token-decoding integrity;
- repetition, premature EOS, and malformed UTF-8 rates;
- validation/test negative log-likelihood and perplexity.

Review generated outputs blindly against the previous checkpoint. A lower perplexity is useful but insufficient. Promotion requires human-readable, coherent samples and no held-out leakage.

## Context and token coverage experiment

The current reference Transformer is CPU-oriented and uses exact scalar autodiff, so long contexts and dense target coverage are computationally expensive. This experiment raises the model context limit and uses a 64-token context, 512-token sequence cap, and target stride 2. These are still modest lengths relative to paragraph-level language modeling; profile throughput before increasing further. Never silently truncate: retain and report truncation counts, and eventually split long documents into overlapping windows so all text can contribute while each target has preceding context.

## Next engineering milestones

- Add a windowing dataset loader that turns long passages into overlapping token windows, preserves EOS at document boundaries, and reports total/used/skipped tokens.
- Add deterministic curriculum mixing and per-stage provenance.
- Add generation-quality tests that inspect outputs rather than merely checking output length.
- Optimize training throughput (mini-batches/vectorized gradients) before expanding context to hundreds or thousands of tokens.
- Keep the current production checkpoint unchanged until the quality gate passes.
