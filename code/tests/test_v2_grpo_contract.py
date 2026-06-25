import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[1]


class V2GrpoContractTests(unittest.TestCase):
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

    def test_paths_are_new_grpo_derag2_paths_and_do_not_overwrite_dpo(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            code = (
                "import json,run_v2_grpo_online as r; "
                "print(json.dumps({"
                "'eval_tag':r.EVAL_TAG,'served':r.SERVED_NAME,'base':str(r.BASE_MODEL),"
                "'warmup_lora':str(r.WARMUP_LORA),'lora':str(r.FINAL_LORA),"
                "'merged':str(r.FINAL_MERGED),'data':str(r.GRPO_DATA),'logs':str(r.RAW_DIR)}))"
            )
            cp = subprocess.run([sys.executable, "-c", code], cwd=CODE, env=self._env(root),
                                text=True, capture_output=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            d = json.loads(cp.stdout)
            self.assertEqual(d["eval_tag"], "v2-2s-2s-2s-grpo")
            self.assertEqual(d["served"], "v2_2s_2s_2s_grpo")
            self.assertEqual(Path(d["base"]).name, "v2-dpo-2sigma-2s-2s-merged")
            self.assertEqual(Path(d["warmup_lora"]).name, "v2-grpo-warmup-2s-2s-2s-lora")
            self.assertEqual(Path(d["lora"]).name, "v2-grpo-2sigma-2s-2s-2s-lora")
            self.assertEqual(Path(d["merged"]).name, "v2-grpo-2sigma-2s-2s-2s-merged")
            self.assertEqual(Path(d["data"]).name, "70_grpo_data.v2-2s-2s-2s.jsonl")
            self.assertEqual(Path(d["logs"]).name, "v2_grpo_online")
            self.assertNotEqual(Path(d["merged"]).name, "v2-dpo-2sigma-2s-2s-merged")

    def test_launcher_uses_old_swift_colocate_recipe_and_v2_online_reward(self):
        launcher = (CODE / "scripts" / "run_v2_grpo_online.sh").read_text(encoding="utf-8")
        self.assertIn('export V2_TAG="${V2_TAG:-derag2}"', launcher)
        self.assertIn("v2-dpo-2sigma-2s-2s-merged", launcher)
        self.assertIn("V2_GRPO_WARMUP_STEPS", launcher)
        self.assertIn("V2_GRPO_MAIN_STEPS", launcher)
        self.assertIn("V2_GRPO_REWARD_AUDIT", launcher)
        self.assertIn("V2_GRPO_KIMI_SMOKE", launcher)
        self.assertIn("V2_GRPO_SWIFT_SMOKE", launcher)
        self.assertIn("GRPO_V2_KIMI_LOCK", launcher)
        self.assertIn('export GRPO_ENV="${GRPO_ENV:-/home/nvme02/conda/grpo_env}"', launcher)
        self.assertIn('export VLLM_TP="${VLLM_TP:-8}"', launcher)
        self.assertIn("run_v2_grpo_online.py", launcher)

        grpo = (CODE / "swift" / "grpo_on_model.sh").read_text(encoding="utf-8")
        self.assertIn("--vllm_mode colocate", grpo)
        self.assertIn("--move_model_batches 16", grpo)
        self.assertIn("--offload_model true", grpo)

        runner = (CODE / "run_v2_grpo_online.py").read_text(encoding="utf-8")
        self.assertIn("step_v2_grpo_reward_audit.py", runner)
        self.assertIn("kimi_smoke()", runner)
        self.assertIn("swift_v2_grpo_smoke2", runner)

    def test_grpo_data_pool_includes_canonical_gold_and_sample_answers(self):
        sys.path.insert(0, str(CODE))
        from pipeline.step_v2_build_grpo_data import build_rows

        rows = build_rows(
            [{"qid": "q1", "query": "q", "user_prompt": "u", "split": "train",
              "gold_answer": "本月销售额9万元，未超过10万元，可以免征增值税。"}],
            {"q1": {
                "v1_canonical_answer": "本月销售额9万元，未超过10万元，可以免征增值税。",
                "gold_answer": "销售额9万元低于10万元标准，因此免征增值税。",
                "v1_answers": ["本月9万元没有超过10万元，可以免税。", "本月9万元没有超过10万元，可以免税。", ""],
            }},
            shuffle_seed=None,
        )
        pool = json.loads(rows[0]["v1_answers_json"])
        self.assertEqual(pool, [
            "本月销售额9万元，未超过10万元，可以免征增值税。",
            "销售额9万元低于10万元标准，因此免征增值税。",
            "本月9万元没有超过10万元，可以免税。",
        ])
        self.assertEqual(rows[0]["v1_answer_pool_trainable"], True)
        self.assertEqual(rows[0]["gold_answer"], "本月销售额9万元，未超过10万元，可以免征增值税。")

    def test_grpo_data_filters_untrainable_vague_answer_pool(self):
        sys.path.insert(0, str(CODE))
        from pipeline.step_v2_build_grpo_data import build_rows

        rows = build_rows(
            [{"qid": "q1", "query": "q", "user_prompt": "u", "split": "train", "gold_answer": "需结合实际判断。"},
             {"qid": "q2", "query": "q2", "user_prompt": "u2", "split": "train",
              "gold_answer": "本月销售额9万元，未超过10万元，可以免征增值税。"}],
            {
                "q1": {"v1_answers": ["需结合实际情况判断。", "请按政策规定处理。"]},
                "q2": {"v1_answers": ["本月销售额9万元，未超过10万元，可以免征增值税。"]},
            },
            shuffle_seed=None,
        )
        self.assertEqual([r["qid"] for r in rows], ["q2"])

    def test_grpo_data_rejects_eval_qid_overlap(self):
        sys.path.insert(0, str(CODE))
        from pipeline.step_v2_build_grpo_data import build_rows

        with self.assertRaises(SystemExit):
            build_rows(
                [{"qid": "q_eval", "query": "q", "user_prompt": "u", "split": "train", "gold_answer": "gold"}],
                {"q_eval": {"v1_answers": ["answer"]}},
                eval_qids={"q_eval"},
                shuffle_seed=None,
            )


if __name__ == "__main__":
    unittest.main()
