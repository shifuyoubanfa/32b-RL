import unittest
from unittest.mock import patch

from pipeline.reward import parse_think_answer_diagnostic
from pipeline.v2_paths import qid_of
from pipeline import step_v2_eval


class EvalFormatAccountingTests(unittest.TestCase):
    def test_truncated_think_is_preserved_and_never_masquerades_as_answer(self):
        raw = "<think>部分推理反复循环，直到达到最大生成长度"
        got = parse_think_answer_diagnostic(raw)
        self.assertEqual(got["think"], "部分推理反复循环，直到达到最大生成长度")
        self.assertEqual(got["answer"], "")
        self.assertFalse(got["format_ok"])
        self.assertEqual(got["format_reason"], "missing_think_close+empty_answer")

    def test_format_failure_forces_rule_and_answer_failure_but_keeps_kimi_score(self):
        query = "测试题"
        qid = qid_of(query)
        rec = {
            "query": query,
            "user_prompt": "【参考问答对】参考【问题】测试题",
            "think": "没有任何规则关键词的残缺推理",
            "answer": "",
            "format_ok": False,
            "format_reason": "missing_think_close+empty_answer",
        }
        support = {qid: {"v1_answers": ["应当缴纳3%的税款"]}}
        with patch.object(step_v2_eval, "score_think_eval",
                          return_value={"clean_score": 4.0, "n": 3}), \
             patch.object(step_v2_eval, "score_think_rule",
                          return_value={"has_rag_style": False, "n_traces": 0}):
            got = step_v2_eval.score_record(rec, support)
        self.assertEqual(got["clean_score"], 4.0)
        self.assertEqual(got["clean_n"], 3)
        self.assertFalse(got["has_rag_style"])
        self.assertFalse(got["rule_pass"])
        self.assertTrue(got["rule_forced_failure"])
        self.assertFalse(got["in_pool"])
        self.assertEqual(got["answer_reason"], "empty_answer")


if __name__ == "__main__":
    unittest.main()
