import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "mlx_gputrace", TOOLS / "mlx_gemma4_gputrace_capture.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CaptureReceiptTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                MODULE.strict_json(path)

    def test_tree_hash_is_deterministic_and_rejects_empty_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "capture.gputrace"
            trace.mkdir()
            with self.assertRaisesRegex(ValueError, "contains no files"):
                MODULE.sha256_tree(trace)
            (trace / "b").write_bytes(b"two")
            (trace / "a").write_bytes(b"one")
            first = MODULE.sha256_tree(trace)
            second = MODULE.sha256_tree(trace)
            self.assertEqual(first, second)
            self.assertEqual(first[1:], (2, 6))

    def test_build_receipt_requires_debug_flag_and_runtime_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mlx"
            source.mkdir()
            core = root / "core.so"
            core.write_bytes(b"runtime")
            receipt = root / "build.json"
            value = {
                "schema": MODULE.BUILD_SCHEMA,
                "status": "built_mlx_metal_debug",
                "mlx_commit": MODULE.PINNED_MLX_COMMIT,
                "cmake_args": MODULE.BUILD_FLAGS,
                "mlx_source": str(source.resolve()),
                "core_path": str(core.resolve()),
                "core_sha256": MODULE.sha256_file(core),
            }
            receipt.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                MODULE.validate_build_receipt(receipt, core, source), value
            )
            value["cmake_args"] = []
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MLX_METAL_DEBUG"):
                MODULE.validate_build_receipt(receipt, core, source)


if __name__ == "__main__":
    unittest.main()
