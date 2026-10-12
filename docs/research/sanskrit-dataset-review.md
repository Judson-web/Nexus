# Sanskrit dataset source review

Review date: 2026-10-12

## Candidate A: CLTK Sanskrit-English parallel corpus

- Repository: https://github.com/cltk/sanskrit_parallel_gitasupersite
- Source revision to pin before any use: resolve and record the exact commit SHA at acquisition time; do not rely on a moving branch.
- Declared contents: Sanskrit-English parallel corpus for the Bhagavad Gita and Valmiki Ramayana.
- Repository README states that the files are public domain: https://github.com/cltk/sanskrit_parallel_gitasupersite/blob/master/README.md
- Repository LICENSE.md uses the Creative Commons Public Domain Mark 1.0, which says the work is identified as having no known copyright restrictions, while noting jurisdictional caveats: https://github.com/cltk/sanskrit_parallel_gitasupersite/blob/master/LICENSE.md

### Assessment

Promising first candidate for a *small tokenization audit*, not yet approved as a general pretraining corpus.

Before acquiring it:
1. Pin the repository commit and record per-file SHA-256 hashes.
2. Inspect the exact files and their source/edition notes; verify that the license statement applies to the actual Sanskrit and English text files, not merely repository code or metadata.
3. Preserve source links and attribution/context in a manifest.
4. Check whether the English text is an older public-domain translation or a newer copyrighted translation; do not assume the original Sanskrit text's age clears translation rights.
5. Keep this classical literary sample separate from any contemporary prose benchmark. A tokenizer result on scripture/epic verse does not establish token efficiency on modern technical or conversational Sanskrit.

### Other candidate considered: Sāmayik

- Dataset card: https://huggingface.co/datasets/acomquest/Saamayik
- The mirrored dataset card declares MIT and describes 43,493 train, 2,416 validation, and 2,417 test pairs, plus a separate out-of-domain split.
- It aggregates content from multiple sources including Bible translations, Gitasopanam, educational materials, and technical tutorials. The repository-level MIT declaration alone is not sufficient evidence that every upstream component can be redistributed and used under that license.

**Decision:** do not ingest Sāmayik into training until upstream rights are resolved. It may be reconsidered after a source-by-source rights audit.

## Gate

No corpus is copied into the repository or used for model training as part of this review. Approval requires a pinned revision, file-level hashes, source/edition review, a clear rights basis for each included component, and a documented acquisition manifest.
