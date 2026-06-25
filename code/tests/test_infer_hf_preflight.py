import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "pipeline" / "infer_hf.py"


class InferHfPreflightTests(unittest.TestCase):
    def _fixture(self, root: Path):
        base = root / "base"
        adapter = root / "adapter"
        base.mkdir()
        adapter.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        eval_file = root / "eval.jsonl"
        eval_file.write_text(json.dumps({"query": "q1", "user_prompt": "u", "answer": "a"}) + "\n",
                             encoding="utf-8")
        return base, adapter, eval_file

    def _run(self, root: Path, base: Path, adapter: Path, eval_file: Path, out: Path):
        env = os.environ.copy()
        env["ZHJG_WORK_DIR"] = str(root / "work")
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--base", str(base), "--adapter", str(adapter),
             "--eval_file", str(eval_file), "--out", str(out)],
            text=True, capture_output=True, env=env, timeout=20,
        )

    def _write_fake_model_stack(self, root: Path):
        (root / "torch.py").write_text(
            "bfloat16 = object()\n"
            "class inference_mode:\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *args): return False\n",
            encoding="utf-8",
        )
        (root / "transformers.py").write_text(
            "class _Ids:\n"
            "    shape = (1, 1)\n"
            "class _Inputs(dict):\n"
            "    input_ids = _Ids()\n"
            "    def to(self, device): return self\n"
            "class AutoTokenizer:\n"
            "    eos_token_id = 0\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *args, **kwargs): return cls()\n"
            "    def apply_chat_template(self, *args, **kwargs): return 'prompt'\n"
            "    def __call__(self, *args, **kwargs): return _Inputs(input_ids=_Ids())\n"
            "    def decode(self, *args, **kwargs): return '<think>自然推理</think><answer>免征</answer>'\n"
            "class _Param:\n"
            "    device = 'cpu'\n"
            "class AutoModelForCausalLM:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *args, **kwargs): return cls()\n"
            "    def eval(self): return self\n"
            "    def parameters(self): return iter([_Param()])\n"
            "    def generate(self, **kwargs): return [[10, 20]]\n",
            encoding="utf-8",
        )
        (root / "peft.py").write_text(
            "class PeftModel:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, model, *args, **kwargs): return model\n",
            encoding="utf-8",
        )

    def test_complete_output_exits_before_heavy_model_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, adapter, eval_file = self._fixture(root)
            out = root / "out.jsonl"
            out.write_text(json.dumps({"query": "q1", "answer": "a"}) + "\n", encoding="utf-8")
            result = self._run(root, base, adapter, eval_file, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already complete", result.stdout)

    def test_malformed_resume_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, adapter, eval_file = self._fixture(root)
            out = root / "out.jsonl"
            out.write_text("{broken\n", encoding="utf-8")
            result = self._run(root, base, adapter, eval_file, out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("已有输出第 1 行损坏", result.stderr + result.stdout)

    def test_pending_item_writes_compatible_row_and_finishes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_fake_model_stack(root)
            base, adapter, eval_file = self._fixture(root)
            out = root / "nested" / "out.jsonl"
            result = self._run(root, base, adapter, eval_file, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(row["query"], "q1")
            self.assertEqual(row["think"], "自然推理")
            self.assertEqual(row["answer"], "免征")
            self.assertTrue(row["format_ok"])
            self.assertIn("rows=1 empty_answers=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
