import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_pretrained_lora as training


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Model role/template overhead plus roughly one token per four chars.
        size = 2 + sum(3 + (len(message["content"]) + 3) // 4 for message in messages)
        return list(range(size))


def example(text, split):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"id": digest[:24], "text": text, "sha256": digest, "split": split}


class SerializedConversationTests(unittest.TestCase):
    def test_parses_single_turn_conversation(self):
        parsed = training.parse_serialized_conversation(
            "User: Explain gravity simply. Assistant: Gravity attracts objects with mass."
        )
        self.assertEqual([m["role"] for m in parsed], ["user", "assistant"])
        self.assertEqual(parsed[-1]["content"], "Gravity attracts objects with mass.")

    def test_parses_system_instruction_and_multi_turn_history(self):
        parsed = training.parse_serialized_conversation(
            "User: System instruction: Be concise. User asks: Explain plants. "
            "Assistant: Plants use sunlight. User: What does chlorophyll do? "
            "Assistant: It absorbs light for photosynthesis."
        )
        self.assertEqual([m["role"] for m in parsed], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(parsed[0]["content"], "Be concise.")
        self.assertEqual(parsed[1]["content"], "Explain plants.")
        self.assertEqual(parsed[-1]["content"], "It absorbs light for photosynthesis.")

    def test_escaped_role_markers_are_not_treated_as_turn_boundaries(self):
        parsed = training.parse_serialized_conversation(
            "User: Explain the literal Assistant : marker. "
            "Assistant: Assistant : is ordinary text here."
        )
        self.assertEqual(len(parsed), 2)
        self.assertIn("Assistant :", parsed[0]["content"])
        self.assertIn("Assistant :", parsed[1]["content"])

    def test_rejects_malformed_role_order_and_missing_answer(self):
        with self.assertRaises(ValueError):
            training.parse_serialized_conversation("Assistant: No user prompt.")
        with self.assertRaises(ValueError):
            training.parse_serialized_conversation("User: Hello. User: Second user without assistant.")
        with self.assertRaises(ValueError):
            training.parse_serialized_conversation("User: Missing answer marker")

    def test_prompt_completion_retains_final_answer_and_trims_history_first(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "old question " * 30},
            {"role": "assistant", "content": "old answer " * 30},
            {"role": "user", "content": "middle question " * 30},
            {"role": "assistant", "content": "middle answer " * 30},
            {"role": "user", "content": "latest user question " * 30},
            {"role": "assistant", "content": "Keep this exact final answer intact."},
        ]
        fitted = training.fit_prompt_completion(tokenizer, messages, 80)
        self.assertIsNotNone(fitted)
        self.assertEqual(fitted["completion"][0]["content"], "Keep this exact final answer intact.")
        self.assertEqual(fitted["prompt"][-1]["role"], "user")
        self.assertNotIn("old question", " ".join(m["content"] for m in fitted["prompt"]))

    def test_rejects_answer_that_cannot_fit_without_silent_target_truncation(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "answer " * 500},
        ]
        self.assertIsNone(training.fit_prompt_completion(tokenizer, messages, 128))


class ManifestValidationTests(unittest.TestCase):
    def test_requires_provenance_and_verifies_hashes_and_split_counts(self):
        rows = [
            example("a train sample", "train"),
            example("a validation sample", "validation"),
            example("a test sample", "test"),
        ]
        manifest = {
            "formatVersion": 1,
            "splits": {"train": 1, "validation": 1, "test": 1},
            "sourceMetadata": {
                "name": "fixture",
                "url": "https://example.invalid/data",
                "license": "test-only",
                "attribution": "fixture author",
            },
            "examples": rows,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = training.load_manifest(path)
            self.assertEqual(loaded["counts"], {"train": 1, "validation": 1, "test": 1})
            self.assertEqual(len(loaded["raw_sha256"]), 64)

            manifest["examples"][0]["text"] = "tampered train sample"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                training.load_manifest(path)

    def test_rejects_missing_source_license_or_attribution(self):
        manifest = {
            "formatVersion": 1,
            "splits": {"train": 1, "validation": 1, "test": 1},
            "sourceMetadata": {"name": "fixture", "url": "https://example.invalid/data"},
            "examples": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sourceMetadata.license"):
                training.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
