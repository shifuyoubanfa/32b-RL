"""V2 训练流程公共件：σ-可分选择函数（查 judgecal 标定表）+ 三分评测口径 + v2 命名。

一切以 v1 重开、跑到 DPO 为止。贯穿原则：answer 锁死 V1 原版、只训 think（见技术方案V2 模块2）。
打分口径必须复用 judgecal 标定时的同一套 Kimi 提示词（judgecal_common.judge_clean_score），
否则下面这张标定 σ 表不作数。
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR, CKPT_DIR
from pipeline import kimi_budget   # 项目围栏：无损去重缓存（同一 ref,think,k>=16 跨阶段只算一次）
from pipeline.judgecal_common import judge_clean_score
from pipeline.rules_v6 import detect_rag_style, answer_in_v1_pool

# ====================== judgecal 标定结果（服务器真 Kimi，2026-06-15）======================
# 6 档：真实照抄档位 -> (Kimi 干净分均值, 该档 k=16 档内标准差 σ)。σ 即"3σ带不相交"判据用的那个。
JUDGECAL_CALIB = [
    # (mean_score, sigma)   由脏到干净
    (0.00, 0.00),   # 完全照抄
    (0.03, 0.06),   # 抄4句
    (1.84, 0.17),   # 抄3句
    (2.64, 0.60),   # 抄2句
    (4.64, 1.63),   # 抄1句
    (7.34, 1.85),   # 没抄
]
SIGMA_JUDGE = 0.74      # 单遍打分噪声（k 可压，σ_judge/√k）
SIGMA_BETWEEN = 1.02    # 样本间方差代表值（k 压不掉）；整体评测 SE=√(σ_between²+σ_judge²/k)/√N


def sigma_of_score(score: float) -> float:
    """按 Kimi 干净分查它落在哪档、给出该档的 σ（分段线性插值）。这是'查那个表'。"""
    pts = JUDGECAL_CALIB
    s = max(pts[0][0], min(pts[-1][0], float(score)))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= s <= x1:
            if x1 == x0:
                return y1
            t = (s - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def confident_cleaner(score_clean: float, score_dirty: float, n_sigma: float = 3.0) -> bool:
    """★ σ-选择函数：判 score_clean 这条 think 是否【按 Nσ 带不相交】地比 score_dirty 那条更干净。

    判据（同 judgecal 报告第4节）：可分 ⟺ gap > N·(σ_clean + σ_dirty)，σ 按各自分数查表。
    用于：冷启动选改写（clean=改写分, dirty=V1该题分）、RFT 选 think、DPO 构对（chosen vs rejected）。
    """
    gap = float(score_clean) - float(score_dirty)
    if gap <= 0:
        return False
    return gap > n_sigma * (sigma_of_score(score_clean) + sigma_of_score(score_dirty))


def cleaner_scores(reference: str, clean_think: str, dirty_think: str) -> tuple:
    """省钱选样：先 k=2 粗筛（均值不更干净直接弃），过了再 k=16 双评。返回 (s_clean, s_dirty)。
    任一为 None 表示被筛掉/打分失败 → 调用方据此判不通过。用于冷启动/RFT/DPO 的 σ 选择前置。"""
    a2 = score_think_kimi(reference, clean_think, k=2)
    b2 = score_think_kimi(reference, dirty_think, k=2)
    if a2["clean_score"] is None or b2["clean_score"] is None or a2["clean_score"] <= b2["clean_score"]:
        return None, None
    a16 = score_think_select(reference, clean_think, k=16)
    b16 = score_think_select(reference, dirty_think, k=16)
    return a16["clean_score"], b16["clean_score"]


def eval_se(n_items: int, k: int) -> float:
    """整体评测：N 题、每题打 k 次，模型平均干净分的标准误。两阶段平均差 > ~3×SE 才算真涨。"""
    sigma_total = (SIGMA_BETWEEN ** 2 + SIGMA_JUDGE ** 2 / max(1, k)) ** 0.5
    return sigma_total / max(1, n_items) ** 0.5


# ====================== 三分评测口径（一个模型在 500 验证上）======================
# think-Kimi：judge_clean_score 整段干净分（评测 k=3）；think-规则：detect_rag_style 检索腔；答案：answer_in_v1_pool。

def score_think_kimi(reference: str, think: str, k: int = 2) -> dict:
    """整段 think 的 Kimi 干净分，打 k 次取平均。底层口径函数；评测用 score_think_eval，选样用 score_think_select。

    全 k 遍都失败 → clean_score=None、n=0（**不返回 0.0**，因 0.0 恰是"完全照抄"档均值、会污染评测平均）。
    调用方必须按 n>0 过滤（对齐 step162 collect_items 的 if not valid: continue）。
    """
    cached = kimi_budget.cache_get(reference, think, k)   # 无损去重：仅 k>=16 选样分命中（k=2 评测/粗筛不缓存）
    if cached is not None:
        return cached
    vals = []
    for _ in range(max(1, k)):                            # k 次仍各自独立打分（降噪不动），只是聚合结果不跨阶段重算
        try:
            vals.append(judge_clean_score(reference, think)["clean_score"])
        except Exception:
            pass
    if not vals:
        return {"clean_score": None, "sd": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    result = {"clean_score": mean, "sd": sd, "n": len(vals)}
    kimi_budget.cache_put(reference, think, k, result)
    return result


def score_think_eval(reference: str, think: str) -> dict:
    """整体评测口径：k=3（judgecal 报告 §1 表：N=500 时 SE≈0.05；瓶颈是 σ_between、再加 k 无益）。
    注意与【选样 k=16】区分：选样要套 σ 标定表必须 k=16，评测只为算整体均值、小 k 即可。"""
    return score_think_kimi(reference, think, k=3)


def score_think_select(reference: str, think: str, k: int = 16) -> dict:
    """逐条选样口径：必须 k>=16。σ 标定表按 k=16 档内 σ；少打则 σ_judge 没被 √16 压掉、
    confident_cleaner 的 3σ 判据会偏松、选进噪声样本（K1）。冷启动/RFT/DPO 选 think 一律走这个入口。"""
    assert k >= 16, "选样必须 k>=16（复用 judgecal 标定 σ 表、套 confident_cleaner 的前提）"
    return score_think_kimi(reference, think, k=k)


def score_think_rule(think: str) -> dict:
    """think-规则分：有没有检索腔表面痕迹（detect_rag_style，确定性、零噪声）。"""
    r = detect_rag_style(think)
    return {"has_rag_style": r["has_rag_style"], "n_traces": r["n"], "spans": r["spans"]}


def answer_drift(answer: str, v1_pool_answers: list[str]) -> dict:
    """答案漂移：模型答案的极性+关键数字在不在该题 V1 池（answer_in_v1_pool）。评测看漂移率 vs 100%。"""
    r = answer_in_v1_pool(answer, v1_pool_answers)
    return {"in_pool": r["in_pool"], "comparable": r["comparable"],
            "reason": r["reason"], "drift_facts": r["drift_facts"]}


# ====================== v2 命名（铁律：一律带 v2，绝不与 V1 同名/同目录）======================
# 版本化（永不覆盖）：默认 OUTPUT_DIR/v2；设 V2_TAG=derag2 → OUTPUT_DIR/derag2，新跑全部数据/进度/评测落新目录、
# 旧目录一律不动。Kimi 计量+无损打分缓存在 OUTPUT_DIR 根(kimi_budget.py)、跨 tag 共享 → 打分文件永不丢、
# v1_think 等同分跨版复用、断点续跑各 tag 各自的 *_progress 续。
V2_OUTPUT_DIR = Path(os.environ.get("ZHJG_V2_OUTPUT_DIR") or str(Path(OUTPUT_DIR) / os.environ.get("V2_TAG", "v2")))
V2_CKPT_DIR = Path(CKPT_DIR)


def v2_lora_dir(stage: str, sigma: int, lineage: str = "") -> Path:
    """LoRA 落点：v2-{stage}-{sigma}sigma[-{lineage}]-lora。stage∈sft/rft/dpo；sigma∈2/3。"""
    tail = f"-{lineage}" if lineage else ""
    return V2_CKPT_DIR / f"v2-{stage}-{sigma}sigma{tail}-lora"


def v2_merged_dir(stage: str, sigma: int, lineage: str = "") -> Path:
    tail = f"-{lineage}" if lineage else ""
    return Path(os.environ.get("ZHJG_MODEL_DIR", str(Path(os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg")) / "models"))) / f"v2-{stage}-{sigma}sigma{tail}-merged"


def v2_eval_paths(tag: str) -> tuple[Path, Path, Path]:
    """某 v2 模型在 500 验证上的 (推理, 三分判定, 报告) 路径。tag 必须带 v2。"""
    assert "v2" in tag, "v2 评测 tag 必须带 v2 标签"
    base = V2_OUTPUT_DIR
    return base / f"{tag}_infer.jsonl", base / f"{tag}_scores.jsonl", base / f"{tag}_report.md"


def v2_summary_path(tag: str) -> Path:
    """三分评测的机读摘要 json（{clean_mean,rule_pass_rate,in_pool_rate,se,...}），剪枝判定用。"""
    assert "v2" in tag, "v2 评测 tag 必须带 v2 标签"
    return V2_OUTPUT_DIR / f"{tag}_summary.json"


if __name__ == "__main__":  # 纯函数自测：选择函数 + 评测SE，不调 Kimi
    print("=== σ-选择函数自测（confident_cleaner，判据：gap > N(σ_clean+σ_dirty)）===")
    cases = [
        ("没抄(7.3) vs 抄4(0.03)", 7.3, 0.03),
        ("没抄(7.3) vs 抄1(4.6)", 7.3, 4.6),
        ("没抄(7.3) vs 抄2(2.6)", 7.3, 2.6),
        ("抄1(4.6) vs 完全(0.0)", 4.6, 0.0),
        ("抄3(1.8) vs 抄4(0.03)", 1.8, 0.03),
    ]
    for name, a, b in cases:
        g = a - b
        s = sigma_of_score(a) + sigma_of_score(b)
        print(f"  {name}: gap={g:.2f} (σ_clean+σ_dirty)={s:.2f} ratio={g/s:.1f}  "
              f"3σ={confident_cleaner(a,b,3)} 2σ={confident_cleaner(a,b,2)}")
    print("=== 整体评测 SE（N=500,k=2 / N=224,k=2 / N=500,k=16）===")
    for n, k in [(500, 2), (224, 2), (500, 16)]:
        print(f"  N={n} k={k}: SE={eval_se(n,k):.3f}  → 两阶段差 >~{3*eval_se(n,k):.2f} 才算真涨")
