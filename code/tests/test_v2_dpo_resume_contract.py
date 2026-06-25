import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[1]


class V2DpoResumeContractTests(unittest.TestCase):
    def _env(self, root: Path):
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(CODE),
            "ZHJG_WORK_DIR": str(root),
            "ZHJG_OUTPUT_DIR": str(root / "output"),
            "ZHJG_CKPT_DIR": str(root / "ckpts"),
            "ZHJG_MODEL_DIR": str(root / "models"),
            "ZHJG_LOG_DIR": str(root / "logs"),
            "ZHJG_ENV": str(root / "env"),
            "VLLM_ENV": str(root / "vllm_env"),
            "V2_TAG": "derag2",
            "KIMI_MODEL": "kimi/kimi-k2.6",
            "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
        })
        return env

    def test_paths_exactly_match_original_run_v2_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            code = (
                "import json,run_v2_dpo_resume as r; "
                "print(json.dumps({"
                "'tag':r.TAG,'served':r.SERVED_NAME,'base':str(r.RFT_BASE),"
                "'canonical_lora':str(r.CANONICAL_LORA),'merged':str(r.DPO_MERGED),"
                "'infer':str(r.INFER),'scores':str(r.SCORES),'report':str(r.REPORT),"
                "'summary':str(r.SUMMARY),'logs':str(r.LOG_ROOT)}))"
            )
            cp = subprocess.run([sys.executable, "-c", code], cwd=CODE, env=self._env(root),
                                text=True, capture_output=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            d = json.loads(cp.stdout)
            self.assertEqual(d["tag"], "v2-2s-2s-2s")
            self.assertEqual(d["served"], "v2_2s_2s_2s")
            self.assertEqual(Path(d["base"]).name, "v2-rft-2sigma-2s-merged")
            self.assertEqual(Path(d["canonical_lora"]).name, "v2-dpo-2sigma-2s-2s-lora")
            self.assertEqual(Path(d["merged"]).name, "v2-dpo-2sigma-2s-2s-merged")
            self.assertEqual(Path(d["infer"]).name, "v2-2s-2s-2s_infer.jsonl")
            self.assertEqual(Path(d["scores"]).name, "v2-2s-2s-2s_scores.jsonl")
            self.assertEqual(Path(d["report"]).name, "v2-2s-2s-2s_report.md")
            self.assertEqual(Path(d["summary"]).name, "v2-2s-2s-2s_summary.json")
            self.assertEqual(Path(d["infer"]).parent.name, "derag2")
            self.assertEqual(Path(d["logs"]).name, "v2_dpo_resume")

    def test_downloaded_adapter_is_atomically_published_to_canonical_lora(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            downloaded = root / "ckpts" / "v2-dpo-derag2-lora" / "checkpoint-108"
            downloaded.mkdir(parents=True)
            (downloaded / "adapter_config.json").write_text("{}", encoding="utf-8")
            (downloaded / "adapter_model.safetensors").write_bytes(b"adapter-test")
            code = (
                "import run_v2_dpo_resume as r; r.LOG_ROOT.mkdir(parents=True,exist_ok=True); "
                "r.publish_canonical_lora(); "
                "print(r.CANONICAL_LORA); print((r.CANONICAL_LORA/'.done').exists()); "
                "print((r.CANONICAL_ADAPTER/'adapter_model.safetensors').read_bytes().decode())"
            )
            cp = subprocess.run([sys.executable, "-c", code], cwd=CODE, env=self._env(root),
                                text=True, capture_output=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            lines = cp.stdout.strip().splitlines()
            self.assertIn("v2-dpo-2sigma-2s-2s-lora", lines[-3])
            self.assertEqual(lines[-2], "True")
            self.assertEqual(lines[-1], "adapter-test")

    def test_launcher_forces_derag2_and_does_not_embed_a_new_output_tag(self):
        launcher = (CODE / "scripts" / "run_v2_dpo_resume.sh").read_text(encoding="utf-8")
        self.assertIn("export V2_TAG=derag2", launcher)
        self.assertIn('export ZHJG_V2_OUTPUT_DIR="$ZHJG_OUTPUT_DIR/derag2"', launcher)
        self.assertIn('export ZHJG_LOG_FILE="$ZHJG_LOG_DIR/v2_dpo_resume/pipeline.log"', launcher)
        self.assertIn("flock -n 9", launcher)
        self.assertIn('V2_DPO_GPU_STABLE_SAMPLES', launcher)
        self.assertIn('export VLLM_BASE_URL="http://127.0.0.1:8000/v1"', launcher)
        self.assertIn('export KIMI_MODEL="kimi/kimi-k2.6"', launcher)
        self.assertIn("v2-dpo-2sigma-2s-2s-merged", launcher)
        self.assertNotIn("v2-dpo-derag2-merged", launcher)
        self.assertNotIn("derag2_dpo_summary", launcher)

    def test_v2_grpo_launcher_uses_swift_colocate_and_derag2(self):
        launcher = (CODE / "scripts" / "run_v2_grpo_online.sh").read_text(encoding="utf-8")
        swift = (CODE / "swift" / "grpo_on_model.sh").read_text(encoding="utf-8")
        self.assertIn('export V2_TAG="${V2_TAG:-derag2}"', launcher)
        self.assertIn('export GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"', launcher)
        self.assertIn('export VLLM_TP="${VLLM_TP:-8}"', launcher)
        self.assertIn('export V2_GRPO_WARMUP_STEPS="${V2_GRPO_WARMUP_STEPS:-30}"', launcher)
        self.assertIn('export V2_GRPO_MAIN_STEPS="${V2_GRPO_MAIN_STEPS:-90}"', launcher)
        self.assertIn("run_v2_grpo_online.py", launcher)
        self.assertIn("--rlhf_type grpo", swift)
        self.assertIn("--use_vllm true", swift)
        self.assertIn("--vllm_mode colocate", swift)
        self.assertIn("--external_plugins", swift)
        self.assertIn("--reward_funcs \"$REWARD_FUNC\"", swift)
        self.assertIn("--move_model_batches 16", swift)
        self.assertIn("--offload_model true", swift)

    def test_infer_contract_rejects_alignment_and_empty_generation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            program = r'''
import json
import run_v2_dpo_resume as r

r.V2_EVAL.parent.mkdir(parents=True, exist_ok=True)
eval_rows = []
for i in range(500):
    q = f"冻结问题{i}"
    eval_rows.append({"qid": r.qid_of(q), "query": q, "user_prompt": f"提示{i}",
                      "answer": f"金标{i}", "reasoning": f"推理{i}", "split": "eval"})
support = [{"qid": x["qid"], "v1_answers": [x["answer"]]} for x in eval_rows]
for i in range(1739):
    q = f"池问题{i}"
    support.append({"qid": r.qid_of(q), "v1_answers": [f"池答案{i}"]})
for path, rows in ((r.V2_EVAL, eval_rows), (r.V2_V1_SUPPORT, support)):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
infer = [{"query": x["query"], "user_prompt": x["user_prompt"], "gold_answer": x["answer"],
          "gen_text": "<think>有效推理</think><answer>有效答案</answer>",
          "think": "有效推理", "answer": "有效答案",
          "format_ok": True, "format_reason": "ok"} for x in eval_rows]
with r.INFER.open("w", encoding="utf-8") as f:
    for row in infer:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(r.infer_status(r.INFER)["ok"])
infer[7]["user_prompt"] = "错位"
with r.INFER.open("w", encoding="utf-8") as f:
    for row in infer:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(r.infer_status(r.INFER)["ok"])
infer[7]["user_prompt"] = eval_rows[7]["user_prompt"]
infer[8]["gen_text"] = "<think>达到长度后截断"
infer[8].update(r.parse_think_answer_diagnostic(infer[8]["gen_text"]))
with r.INFER.open("w", encoding="utf-8") as f:
    for row in infer:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(r.infer_status(r.INFER)["ok"], r.infer_status(r.INFER)["format_failures"])
infer[9]["answer"] = ""
with r.INFER.open("w", encoding="utf-8") as f:
    for row in infer:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(r.infer_status(r.INFER)["ok"])
'''
            cp = subprocess.run([sys.executable, "-c", program], cwd=CODE, env=self._env(root),
                                text=True, capture_output=True, timeout=30)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertEqual(cp.stdout.strip().splitlines(), ["True", "False", "True 1", "False"])

    def test_approved_existing_infer_is_adopted_without_regeneration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            program = r'''
import json, os
import run_v2_dpo_resume as r

r.LOG_ROOT.mkdir(parents=True, exist_ok=True)
r.V2_EVAL.parent.mkdir(parents=True, exist_ok=True)
eval_rows = []
for i in range(500):
    q = r.EXPECTED_FORMAT_FAILURE_QUERY if i == 436 else f"冻结问题{i}"
    eval_rows.append({"qid": r.qid_of(q), "query": q, "user_prompt": f"提示{i}",
                      "answer": f"金标{i}", "reasoning": f"推理{i}", "split": "eval"})
support = [{"qid": x["qid"], "v1_answers": [x["answer"]]} for x in eval_rows]
for i in range(1739):
    q = f"池问题{i}"
    support.append({"qid": r.qid_of(q), "v1_answers": [f"池答案{i}"]})
for path, data in ((r.V2_EVAL, eval_rows), (r.V2_V1_SUPPORT, support)):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
rows = []
for i, src in enumerate(eval_rows):
    raw = "<think>循环直到截断" if i == 436 else "<think>有效推理</think><answer>有效答案</answer>"
    think, answer = (("", raw) if i == 436 else ("有效推理", "有效答案"))
    rows.append({"query": src["query"], "user_prompt": src["user_prompt"],
                 "gold_answer": src["answer"], "gen_text": raw,
                 "think": think, "answer": answer})
with r.INFER.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
source_sha = r.sha256(r.INFER)
os.environ["V2_DPO_APPROVE_FORMAT_ACCOUNTING"] = "1"
os.environ["V2_DPO_EXPECT_INFER_SHA256"] = source_sha
r.infer_identity = lambda: {"test_identity": "fixed"}
r.adopt_approved_existing_infer()
fixed = r.read_jsonl(r.INFER)
print(r.infer_trusted())
print(fixed[436]["think"], repr(fixed[436]["answer"]), fixed[436]["format_ok"])
print(r.infer_status(r.INFER)["format_failures"])
print(len(list(r.INFER.parent.glob(r.INFER.name + ".pre_format_accounting-*"))))
print(r.read_json(r.INFER_FORMAT_MANIFEST)["generation_changed"])
'''
            cp = subprocess.run([sys.executable, "-c", program], cwd=CODE, env=self._env(root),
                                text=True, capture_output=True, timeout=30)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            lines = cp.stdout.strip().splitlines()
            self.assertEqual(lines[-5:], ["True", "循环直到截断 '' False", "1", "1", "False"])


if __name__ == "__main__":
    unittest.main()
