#!/usr/bin/env python3
"""Fine-tune a pretrained instruction model with LoRA on a Nexus dataset manifest.

Uses TRL prompt-completion datasets so loss is computed on assistant completions,
not on user/system prompts. Keeps validation and test splits isolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROLE_MARKER = re.compile(r" (Assistant|User): ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/nexus-qwen-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional deterministic cap; 0 uses the full training split.")
    args = parser.parse_args()
    if not 0 < args.epochs <= 100:
        parser.error("--epochs must be in (0, 100].")
    if not 0 < args.learning_rate <= 0.01:
        parser.error("--learning-rate must be in (0, 0.01].")
    if not 1 <= args.batch_size <= 64:
        parser.error("--batch-size must be in [1, 64].")
    if not 1 <= args.gradient_accumulation <= 256:
        parser.error("--gradient-accumulation must be in [1, 256].")
    if not 128 <= args.max_seq_length <= 16384:
        parser.error("--max-seq-length must be in [128, 16384].")
    if not 1 <= args.lora_rank <= 128:
        parser.error("--lora-rank must be in [1, 128].")
    if args.max_train_examples < 0:
        parser.error("--max-train-examples cannot be negative.")
    return args


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("formatVersion") != 1:
        raise ValueError("Unsupported dataset manifest format; expected formatVersion=1.")
    metadata = manifest.get("sourceMetadata")
    if not isinstance(metadata, dict):
        raise ValueError("Manifest is missing sourceMetadata provenance.")
    for field in ("name", "url", "license", "attribution"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Manifest sourceMetadata.{field} is required.")
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Manifest examples must be an array.")
    seen: set[str] = set()
    counts = {"train": 0, "validation": 0, "test": 0}
    for item in examples:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("Manifest contains an invalid example.")
        split, digest = item.get("split"), item.get("sha256")
        if split not in counts or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Every example needs a valid split and SHA-256.")
        if digest_text(item["text"]) != digest:
            raise ValueError("Example text hash does not match manifest SHA-256.")
        if digest in seen:
            raise ValueError("Manifest contains exact duplicate examples.")
        seen.add(digest)
        counts[split] += 1
    for split, count in counts.items():
        if count != manifest.get("splits", {}).get(split):
            raise ValueError(f"Manifest {split} count does not match its examples.")
        if count == 0:
            raise ValueError(f"Manifest has no {split} examples.")
    return {
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "metadata": metadata,
        "examples": examples,
        "counts": counts,
    }


def parse_serialized_conversation(text: str) -> list[dict[str, str]]:
    """Parse the adapter's escaped User:/Assistant: format without guessing roles."""
    if not text.startswith("User: "):
        raise ValueError("example does not start with a user message")
    body = text[len("User: "):]
    markers = list(ROLE_MARKER.finditer(body))
    first_end = markers[0].start() if markers else len(body)
    first = body[:first_end].strip()
    messages: list[dict[str, str]] = []
    system_prefix = "System instruction: "
    system_separator = " User asks: "
    if first.startswith(system_prefix) and system_separator in first:
        system_text, first_user = first[len(system_prefix):].split(system_separator, 1)
        if not system_text.strip() or not first_user.strip():
            raise ValueError("system/user instruction prefix is malformed")
        messages.extend([
            {"role": "system", "content": system_text.strip()},
            {"role": "user", "content": first_user.strip()},
        ])
    else:
        messages.append({"role": "user", "content": first})

    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        content = body[marker.end():end].strip()
        role = marker.group(1).lower()
        expected = "assistant" if messages[-1]["role"] == "user" else "user"
        if role != expected or not content:
            raise ValueError("role markers do not alternate or a turn is empty")
        messages.append({"role": role, "content": content})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("conversation must end with an assistant answer")
    return messages


def chat_token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def fit_prompt_completion(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, Any] | None:
    """Preserve the final answer; prune old turns and then shorten user context."""
    completion = dict(messages[-1])
    prompt = [dict(message) for message in messages[:-1]]
    if not prompt or prompt[-1]["role"] != "user":
        return None
    target_length = chat_token_length(tokenizer, [completion])
    if target_length > max_length - 32:
        return None

    def total_length() -> int:
        return chat_token_length(tokenizer, prompt + [completion])

    while total_length() > max_length:
        system_count = 1 if prompt and prompt[0]["role"] == "system" else 0
        if len(prompt) - system_count >= 3:
            del prompt[system_count:system_count + 2]
        else:
            break
    while total_length() > max_length:
        user = prompt[-1]
        ids = tokenizer.encode(user["content"], add_special_tokens=False)
        if len(ids) <= 16:
            return None
        keep = max(16, int(len(ids) * 0.8))
        user["content"] = tokenizer.decode(ids[-keep:], skip_special_tokens=True).strip()
        if not user["content"]:
            return None
    return {"prompt": prompt, "completion": [completion]}


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    source = load_manifest(args.manifest)
    examples = source["examples"]
    metadata = source["metadata"]
    training = sorted((e for e in examples if e["split"] == "train"), key=lambda e: e["id"])
    validation = [e for e in examples if e["split"] == "validation"]
    testing = [e for e in examples if e["split"] == "test"]
    if args.max_train_examples:
        training = training[:args.max_train_examples]
    if not training or not validation or not testing:
        raise ValueError("Training, validation, and test examples are all required.")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError("Missing dependencies. Install with: pip install -r requirements-training.txt") from error

    print(
        f"Loading {args.base_model}; manifest_sha256={source['raw_sha256']}; "
        f"source={metadata['name']}; license={metadata['license']}",
        file=sys.stderr,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Base tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rejected: dict[str, int] = {}
    def prepare(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in rows:
            try:
                messages = parse_serialized_conversation(row["text"])
                record = fit_prompt_completion(tokenizer, messages, args.max_seq_length)
                if record is None:
                    key = "too-long-" + split
                    rejected[key] = rejected.get(key, 0) + 1
                    continue
                records.append(record)
            except (ValueError, TypeError, KeyError) as error:
                key = "malformed-" + split
                rejected[key] = rejected.get(key, 0) + 1
                print(f"Skipping example {row.get('id', '<unknown>')}: {error}", file=sys.stderr)
        return records

    train_records = prepare(training, "train")
    validation_records = prepare(validation, "validation")
    test_records = prepare(testing, "test")
    if not train_records or not validation_records or not test_records:
        raise ValueError("No usable prompt/completion rows remain for one split. Increase --max-seq-length or inspect data.")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    model.config.use_cache = False
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=bool(torch.cuda.is_available()),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(validation_records),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate(
        eval_dataset=Dataset.from_list(validation_records),
        metric_key_prefix="validation",
    )
    # Test is evaluated only after selection, never used for training or tuning.
    test_metrics = trainer.evaluate(
        eval_dataset=Dataset.from_list(test_records),
        metric_key_prefix="test",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    report = {
        "format": "nexus-pretrained-lora-run",
        "version": 1,
        "baseModel": args.base_model,
        "adapterType": "LoRA PEFT",
        "manifestPath": str(args.manifest),
        "manifestSha256": source["raw_sha256"],
        "source": metadata,
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "learningRate": args.learning_rate,
            "batchSize": args.batch_size,
            "gradientAccumulation": args.gradient_accumulation,
            "maxSequenceLength": args.max_seq_length,
            "loraRank": args.lora_rank,
            "lossPolicy": "prompt-completion-only",
            "requestedSplitCounts": source["counts"],
            "usableSplitCounts": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "rejectedCounts": rejected,
            "trainResult": train_result.metrics,
        },
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
        "testUsedForModelSelection": False,
        "licenseReminder": "Preserve source attribution and review both the base-model and data licenses before redistribution.",
        "note": "This is a LoRA adapter over an existing pretrained model, not from-scratch pretraining.",
    }
    (args.output_dir / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    card = (
        "# Nexus LoRA adapter\n\n"
        "Fine-tuned with Hugging Face TRL SFTTrainer and PEFT LoRA. The base model is required for inference.\n\n"
        f"- Base model: {args.base_model}\n"
        f"- Source dataset: {metadata['name']}\n"
        f"- Source license: {metadata['license']}\n"
        f"- Attribution: {metadata['attribution']}\n"
        f"- Source URL: {metadata['url']}\n"
        f"- Manifest SHA-256: {source['raw_sha256']}\n"
        "- Metrics and exact run settings: training-report.json\n\n"
        "Review base-model and dataset licenses before sharing or deploying this adapter. "
        "This adapter is not an independently pretrained language model.\n"
    )
    (args.output_dir / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps({
        "outputDir": str(args.output_dir),
        "baseModel": args.base_model,
        "adapterType": "LoRA PEFT",
        "usableSplitCounts": report["training"]["usableSplitCounts"],
        "validationLoss": validation_metrics.get("validation_loss"),
        "testLoss": test_metrics.get("test_loss"),
        "testUsedForModelSelection": False,
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
