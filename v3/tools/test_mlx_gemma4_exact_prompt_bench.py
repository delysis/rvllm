import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("mlx_gemma4_exact_prompt_bench.py")
SPEC = importlib.util.spec_from_file_location("mlx_gemma4_exact_prompt_bench", MODULE_PATH)
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class ModelWeightIdentityTests(unittest.TestCase):
    def test_single_safetensor(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw)
            weights = b"single-file model"
            (model / "model.safetensors").write_bytes(weights)
            self.assertEqual(
                bench.model_weight_identity(model),
                {
                    "weight_layout": "single_safetensor",
                    "model_safetensors_sha256": hashlib.sha256(weights).hexdigest(),
                },
            )

    def test_sharded_index(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw)
            index = b"{}"
            (model / "model.safetensors.index.json").write_bytes(index)
            self.assertEqual(
                bench.model_weight_identity(model),
                {
                    "weight_layout": "sharded_index",
                    "model_index_sha256": hashlib.sha256(index).hexdigest(),
                },
            )

    def test_missing_weights_fail_before_timing(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                bench.model_weight_identity(Path(raw))


if __name__ == "__main__":
    unittest.main()
