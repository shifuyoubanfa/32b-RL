"""PMI 尺子打分器（RFT 选样 / GRPO 在线奖励共用）。

加载探针选优出的"尺子模型"（output/pmi_ruler.json：clean_base / pi_ref / current；排除原始 V1），
对一段 think 算 s_pmi=-PMI（越大越自然）。尺子冻结、不更新。
"""

import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    OUTPUT_DIR, V1_DIR, CLEAN_BASE_DIR, COLDSTART_LORA_DIR,
    SYSTEM_PROMPT, PMI_MAX_LEN, PMI_THINK_CAP, resolve_adapter,
)
from pipeline import reward
from pipeline.logger import get_logger

log = get_logger("pmi_scorer")

_MODEL = None
_TOK = None
_RULER = None


def ruler_path(name: str, stage_adapter: str = None):
    """候选尺子 -> (base_path, adapter_path or None)。

    clean_base  = 干净 Qwen2.5-32B-Instruct（不带 adapter）。
    stage_model = V1 + 当前阶段冻结 adapter（RFT 传 coldstart、DPO 传 cs_rft、GRPO 传 dpo）。
    """
    if name == "clean_base":
        return CLEAN_BASE_DIR, None
    if name in ("stage_model", "pi_ref", "current"):  # 兼容旧名，统一用 stage adapter
        # resolve_adapter 解析 swift 嵌套 checkpoint，且对已解析路径幂等——一处兜住 step09/11/12/单独运行所有调用
        return V1_DIR, resolve_adapter(stage_adapter or COLDSTART_LORA_DIR)
    raise ValueError(f"未知尺子候选: {name}")


def load_model(base_path: str, adapter_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).eval()
    return model, tok


def qa_user(query: str, answer: str) -> str:
    """Q + 标准答案 作为 PMI 分母上下文。"""
    return f"【参考问答对】\n[已知正确答案：{answer}]\n【问题】\n{query}"


def load_ruler(name: str = None, stage_adapter: str = None) -> str:
    """加载选中的尺子。优先级：显式 name > config.PMI_RULER 覆盖 > output/pmi_ruler.json。

    探针未过 AUC≥0.7（json 里 pmi_ruler=None）时【阻塞报错】，不静默回退 clean_base，
    以贯彻技术方案"先证伪再烧卡、AUC<0.7 不进 RFT"的红线。返回尺子名。
    """
    global _MODEL, _TOK, _RULER
    from config import PMI_RULER as _OVERRIDE
    if name is None:
        if _OVERRIDE:
            name = _OVERRIDE
            log.info("PMI_RULER 环境覆盖尺子=%s", name)
        else:
            p = Path(OUTPUT_DIR) / "pmi_ruler.json"
            chosen = json.loads(p.read_text(encoding="utf-8")).get("pmi_ruler") if p.exists() else None
            if not chosen:
                raise RuntimeError(
                    "PMI 尺子未确定：探针未过 AUC≥0.7（pmi_ruler.json 为空/缺失）。"
                    "按技术方案应阻塞、在 V1 自然种子上重 tune τ/C/W，不应静默用 clean_base。"
                    "如确需强制指定，设环境变量 PMI_RULER=clean_base|stage_model。")
            name = chosen
    base_p, adp = ruler_path(name, stage_adapter)
    log.info("加载 PMI 尺子 [%s] base=%s adapter=%s", name, base_p, adp)
    _MODEL, _TOK = load_model(base_p, adp)
    _RULER = name
    return name


def s_pmi(user_prompt: str, query: str, answer: str, think: str) -> float:
    if _MODEL is None:
        raise RuntimeError("尺子未加载，请先 load_ruler()")
    return reward.pmi_cond(_MODEL, _TOK, SYSTEM_PROMPT, user_prompt, qa_user(query, answer),
                           think, max_len=PMI_MAX_LEN, think_cap=PMI_THINK_CAP)
