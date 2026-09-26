import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("mlx_gemma4_stage_bench.py")
SPEC = importlib.util.spec_from_file_location("mlx_gemma4_stage_bench", MODULE_PATH)
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class FakeMx:
    def __init__(self):
        self.evaluations = 0

    def eval(self, _value):
        self.evaluations += 1


class StageBenchTests(unittest.TestCase):
    def test_canonical_plan_covers_all_lengths_modes_and_stages(self):
        cases = bench.planned_cases(bench.DEFAULT_LENGTHS)
        self.assertEqual(len(cases), 5 * 2 * len(bench.STAGES))
        self.assertEqual({case.prompt_or_context_tokens for case in cases}, {256, 512, 1024, 2048, 4096})
        self.assertEqual({case.mode for case in cases}, {"prefill", "decode"})
        self.assertEqual({case.category for case in cases}, {
            "embedding", "qkv", "attention_core_sdpa", "o_projection",
            "ffn_gate_up_activation", "ffn_down", "rmsnorm_residual", "lm_head",
        })

    def test_protocol_is_exactly_five_plus_one_hundred_synchronized_calls(self):
        fake = FakeMx()
        value = bench.time_operator(fake, lambda: object())
        self.assertEqual(fake.evaluations, 105)
        self.assertGreaterEqual(value["total_ms"], 0)
        self.assertAlmostEqual(value["mean_ms"], value["total_ms"] / 100)

    def test_lengths_reject_duplicates_and_non_positive_values(self):
        self.assertEqual(bench.parse_lengths("256,512"), (256, 512))
        for invalid in ("", "256,256", "0,256", "abc"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bench.parse_lengths(invalid)

    def test_model_identity_distinguishes_affine_and_bf16_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "model.safetensors.index.json").write_text("{}")
            (root / "config.json").write_text(json.dumps({"model_type": "gemma4", "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}))
            identity = bench.model_identity(root, 4)
            self.assertEqual(identity["config_exposed_weight_bits"], 4)
            with self.assertRaises(ValueError):
                bench.model_identity(root, 16)
            (root / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
            self.assertEqual(bench.model_identity(root, 16)["config_exposed_weight_bits"], 16)

    def test_strict_json_rejects_non_finite_receipts(self):
        with self.assertRaises(ValueError):
            json.dumps({"duration": float("nan")}, allow_nan=False)

    def test_plan_only_smoke_emits_complete_strict_json_without_loading_mlx(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw)
            (model / "config.json").write_text(
                json.dumps({"model_type": "gemma4", "quantization": {"bits": 8}})
            )
            (model / "model.safetensors.index.json").write_text("{}")
            args = SimpleNamespace(
                mlx_source="/mlx",
                mlx_lm_source="/mlx-lm",
                model=str(model),
                weight_bits=8,
                lengths="256,512,1024,2048,4096",
                plan_only=True,
            )
            with patch.object(bench, "mlx_protocol_source_identity", return_value={"commit": bench.PINNED_MLX_COMMIT}), patch.object(
                bench,
                "mlx_lm_source_identity",
                return_value={"commit": bench.PINNED_MLX_LM_COMMIT},
            ):
                receipt = bench.benchmark(args)
            encoded = json.dumps(receipt, allow_nan=False)
            self.assertEqual(receipt["status"], "planned_not_measured")
            self.assertEqual(len(receipt["planned_cases"]), 110)
            self.assertTrue(encoded.startswith("{"))

    def test_protocol_provenance_seals_core_mlx_path_and_commit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            referenced = root / bench.PINNED_MLX_TIME_UTILS_PATH
            referenced.parent.mkdir(parents=True)
            referenced.write_text("def time_fn(): pass\n")
            completed = SimpleNamespace(
                stdout=bench.PINNED_MLX_COMMIT + "\n", stderr="", returncode=0
            )
            clean = SimpleNamespace(stdout="", stderr="", returncode=0)
            with patch.object(
                bench.subprocess, "run", side_effect=[completed, clean]
            ):
                identity = bench.mlx_protocol_source_identity(root)
            self.assertEqual(identity["commit"], bench.PINNED_MLX_COMMIT)
            self.assertEqual(identity["referenced_path"], bench.PINNED_MLX_TIME_UTILS_PATH)
            self.assertEqual(identity["referenced_path_sha256"], bench.sha256_file(referenced))
            self.assertTrue(identity["working_tree_matches_commit"])

    def test_quantized_weight_modules_fail_closed(self):
        class Module:
            def __init__(self, bits):
                self.bits = bits

        bench.validate_weight_modules([Module(4), Module(4)], 4, "qkv")
        bench.validate_weight_modules([object()], 16, "qkv")
        with self.assertRaisesRegex(ValueError, "do not expose requested 4-bit"):
            bench.validate_weight_modules([Module(4), object()], 4, "qkv")
        with self.assertRaisesRegex(ValueError, "do not expose requested 16-bit"):
            bench.validate_weight_modules([Module(8)], 16, "qkv")

    def test_quantization_gate_applies_only_to_weighted_linear_stages(self):
        required = {
            category
            for category, _variant in bench.STAGES
            if bench.category_requires_requested_linear_bits(category)
        }
        self.assertEqual(
            required,
            {"qkv", "o_projection", "ffn_gate_up_activation", "ffn_down"},
        )
        self.assertFalse(bench.category_requires_requested_linear_bits("embedding"))
        self.assertFalse(bench.category_requires_requested_linear_bits("lm_head"))


if __name__ == "__main__":
    unittest.main()
