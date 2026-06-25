import unittest

from pipeline.rules_v6 import answer_in_v1_pool


class AnswerInV1PoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            "本月销售额9万元，未超过10万元，免征增值税。",
            "销售额为9万元，可以享受免税政策。",
            "本月9万元，无需缴纳增值税。",
        ]

    def test_matching_answer_passes(self):
        result = answer_in_v1_pool("本月销售额9万元，免征增值税。", self.pool)
        self.assertTrue(result["in_pool"])
        self.assertTrue(result["comparable"])

    def test_empty_answer_fails(self):
        result = answer_in_v1_pool("", self.pool)
        self.assertFalse(result["in_pool"])
        self.assertEqual(result["reason"], "empty_answer")

    def test_factless_generic_answer_preserves_legacy_metric_but_is_uncomparable(self):
        result = answer_in_v1_pool("建议结合实际情况处理。", self.pool)
        self.assertTrue(result["in_pool"])
        self.assertFalse(result["comparable"])
        self.assertEqual(result["reason"], "no_comparable_facts")

    def test_new_number_fails(self):
        result = answer_in_v1_pool("税率为13%，应缴增值税。", self.pool)
        self.assertFalse(result["in_pool"])
        self.assertIn("13%", result["drift_facts"])

    def test_no_facts_on_either_side_is_uncomparable(self):
        result = answer_in_v1_pool("建议咨询主管税务机关。", ["请结合实际情况判断。"]) 
        self.assertTrue(result["in_pool"])
        self.assertFalse(result["comparable"])


if __name__ == "__main__":
    unittest.main()
