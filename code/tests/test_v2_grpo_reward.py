import json
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_reward_plugin():
    spec = importlib.util.spec_from_file_location("test_grpo_reward_plugin", ROOT / "swift" / "grpo_reward_plugin.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_plugin = _load_reward_plugin()
V2RuleWarmupReward = _plugin.V2RuleWarmupReward
_v2_reward_one = _plugin._v2_reward_one


V1_ANSWERS = [
    "本月销售额9万元，未超过10万元，可以免征增值税。",
    "销售额9万元低于10万元标准，因此免征增值税。",
]


def wrap(think: str, answer: str) -> str:
    return f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"


def test_v2_reward_hard_gate_order_without_kimi():
    good = wrap(
        "先比较本月销售额和小规模纳税人的免税线：本月9万元，没有超过10万元，所以结论是可以免征；"
        "这一步没有引入新的税率、金额、日期或相反结论。",
        V1_ANSWERS[0],
    )
    rule_bad = wrap(
        "本月销售额9万元，没有超过10万元，所以结论是可以免征；这里故意插入<img src=x>这类图床痕迹，"
        "用于测试规则think失败但答案仍然在V1池内。",
        V1_ANSWERS[0],
    )
    answer_bad = wrap(
        "先比较本月销售额和小规模纳税人的免税线，再判断是否超过标准；这段推理虽然格式完整，"
        "但答案会故意写成应缴，用于测试规则answer硬门。",
        "本月销售额9万元，应缴增值税，税率3%。",
    )
    format_bad = "<think>这段输出一直没有闭合，也没有答案"
    missing_answer_tag = (
        "<think>先比较本月销售额和小规模纳税人的免税线：本月9万元，没有超过10万元，所以结论是可以免征。</think>\n"
        "本月销售额9万元，未超过10万元，可以免征增值税。"
    )

    user_prompt = "【参考问答对】材料【问题】本月9万元是否免税"
    good_r = _v2_reward_one(good, user_prompt, V1_ANSWERS, use_kimi=False)
    rule_bad_r = _v2_reward_one(rule_bad, user_prompt, V1_ANSWERS, use_kimi=False)
    answer_bad_r = _v2_reward_one(answer_bad, user_prompt, V1_ANSWERS, use_kimi=False)
    format_bad_r = _v2_reward_one(format_bad, user_prompt, V1_ANSWERS, use_kimi=False)
    missing_answer_tag_r = _v2_reward_one(missing_answer_tag, user_prompt, V1_ANSWERS, use_kimi=False)

    assert good_r > rule_bad_r > answer_bad_r > format_bad_r
    assert missing_answer_tag_r == format_bad_r
    assert answer_bad_r < 0


def test_v2_rule_warmup_expands_prompt_columns_by_num_generations():
    rewards = V2RuleWarmupReward()(
        [
            wrap(
                "本月9万元没有超过10万元，因此可以免征；这一段把金额和免税线直接比较后得出结论，"
                "没有引用参考资料话术。",
                V1_ANSWERS[0],
            ),
            "<think>坏格式",
        ],
        user_prompt=["【参考问答对】材料【问题】本月9万元是否免税"],
        v1_answers_json=[json.dumps(V1_ANSWERS, ensure_ascii=False)],
    )
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]
