"""V2 训练管线共享常量与数据助手（一切带 v2、answer-lock、复用 V1 件）。

命名/路径全部落 v2_common.V2_OUTPUT_DIR(=OUTPUT_DIR/v2)，与 V1/corrected_v*/derag_v* 隔离。
answer-lock 贯穿：训练样本 answer 永远拼 V1 原版（split 里的 V1 贪心 answer），只训 think。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COLDSTART_SYSTEM_PROMPT
from pipeline.v2_common import V2_OUTPUT_DIR

# ====================== 切分 ======================
V2_N_EVAL = int(os.environ.get("V2_N_EVAL", "500"))      # 固定验证集条数（非比例）
V2_SPLIT_SEED = int(os.environ.get("V2_SPLIT_SEED", "42"))

# ====================== 上游数据（全部带 v2）======================
V2_TRAIN = V2_OUTPUT_DIR / "00_data_v2_train.jsonl"          # 1739：qid/query/user_prompt/answer(V1金标准)/reasoning(V1原think)/split
V2_EVAL = V2_OUTPUT_DIR / "00_data_v2_eval.jsonl"           # 500：同字段，全程冻结
V2_PROBLEMS = V2_OUTPUT_DIR / "00_v2_problems_2239.jsonl"   # 建池题集（全 2239）{qid,split,query,user_prompt,gold_answer}
V2_PROBLEMS_TRAIN = V2_OUTPUT_DIR / "00_v2_problems_train.jsonl"  # 仅 1739 train，schema 同上（RFT 自采样池=step151 --problems，含 gold_answer，避免泄漏 eval）
V2_V1_SUPPORT = V2_OUTPUT_DIR / "152_v1_support.v2.jsonl"   # V1 答案池（step152 产，answer_drift 用）

# RFT/DPO 自采样 rollout 候选数（用户方案：RFT 自采 32；可环境变量调小省钱）
V2_RFT_SELFSAMPLE_K = int(os.environ.get("V2_RFT_SELFSAMPLE_K", "32"))
V2_DPO_ROLLOUT_K = int(os.environ.get("V2_DPO_ROLLOUT_K", "16"))

# ====================== 早停"凑够就停"目标（每阶段 2σ 桶攒够就停、不再调 Kimi）======================
# 只控制【处理多少题】，绝不碰每题的 k/n/三道门。0=不限、跑满 1739。
# 3σ 桶是 2σ 的子集、随之累积；早停看 2σ 数（更populated的主桶）。
# 目标分配【凸显 RL】(2026-06-16 用户定)：冷启动(SFT)压低、不封顶、留 headroom 给 RL；
#   RFT 只做小范围验证(RL 前奏)、最低；DPO(主 RL)加量、做重头戏。
V2_COLDSTART_TARGET = int(os.environ.get("V2_COLDSTART_TARGET", "700"))   # SFT：学透去检索腔(防RFT饿死)但不封顶、留~2分headroom给RL（含~10% eval 留出）
V2_RFT_TARGET = int(os.environ.get("V2_RFT_TARGET", "200"))              # RFT 小验证（RL 前奏），每线 2σ；>150 给早停缓冲(干净端产出率低)
V2_DPO_TARGET = int(os.environ.get("V2_DPO_TARGET", "900"))              # DPO 主 RL 真涨主引擎，每线 2σ：把 eval 干净分稳推过 +0.15(3×SE)、不压 MDE 边缘
V2_GATHER_CHUNK = int(os.environ.get("V2_GATHER_CHUNK", "48"))           # 分批粒度（早停最多多跑 1 批）
V2_GATHER_SEED = int(os.environ.get("V2_GATHER_SEED", "42"))            # 洗牌种子：随机抽批(非取前N)；固定值保证续跑接得上，改它=换一批随机样本(会重处理)


def coldstart_train(sigma: int) -> Path:
    return V2_OUTPUT_DIR / f"coldstart_{sigma}sigma_train.v2.jsonl"


def coldstart_eval() -> Path:
    return V2_OUTPUT_DIR / "coldstart_eval.v2.jsonl"        # 自然腔留出 eval（早停/选best用），2σ/3σ 共用


def rft_selfsample(line: str) -> Path:
    return V2_OUTPUT_DIR / f"151_rft_selfsample.{line}.v2.jsonl"


def rft_train(line: str, sigma: int) -> Path:
    return V2_OUTPUT_DIR / f"rft_{line}_{sigma}sigma_train.v2.jsonl"


def dpo_rollout(line: str) -> Path:
    return V2_OUTPUT_DIR / f"60_dpo_rollout.{line}.v2.jsonl"


def dpo_pairs(line: str, sigma: int) -> Path:
    return V2_OUTPUT_DIR / f"dpo_pairs.{line}_{sigma}sigma.v2.jsonl"


# ====================== 通用助手 ======================

def qid_of(q: str) -> str:
    """与 step150.qid_of 同口径：sha1(query)[:12]。"""
    return hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> None:
    """原子落盘（.tmp 写完 replace）：防进程中途被 kill 留半截/截断文件被续跑误判为完整。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def nonempty(path) -> bool:
    """文件存在且非空（>0 字节）。空 σ 桶（合法选不出样本）会写 0 字节文件 → 此函数判 False。"""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def assistant(think: str, answer: str) -> str:
    """V1/swift 一致的 assistant 段拼装（含开头 <think>，模型自吐、训练目标须带上）。"""
    return f"<think>\n{(think or '').strip()}\n</think>\n\n<answer>\n{(answer or '').strip()}\n</answer>"


def sft_row(user_prompt: str, think: str, answer: str, query: str | None = None) -> dict:
    """swift SFT 训练行（answer-lock：answer 永远是 V1 原版，只 think 不同）。system=去检索腔。"""
    return {
        "messages": [
            {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt or ""},
            {"role": "assistant", "content": assistant(think, answer)},
        ],
        "query": query,
    }


def dpo_row(user_prompt: str, chosen_think: str, rejected_think: str, answer: str,
            query: str | None = None, meta: dict | None = None) -> dict:
    """swift rlhf dpo 偏好对行：chosen=messages末assistant、rejected=顶层 rejected_response。
    answer-lock：chosen/rejected 共用【同一份 V1 原 answer】，只 think 不同。"""
    return {
        "messages": [
            {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt or ""},
            {"role": "assistant", "content": assistant(chosen_think, answer)},
        ],
        "rejected_response": assistant(rejected_think, answer),
        "query": query,
        "meta": meta or {},
    }


def index_by_qid(rows: list[dict], key: str = "qid") -> dict:
    return {r[key]: r for r in rows if r.get(key)}


def load_support_index(path=None) -> dict:
    """qid -> V1 答案池记录（含 v1_answers / v1_canonical_think / gold_answer / support）。"""
    return index_by_qid(read_jsonl(path or V2_V1_SUPPORT))


def coldstart_progress() -> Path:
    return V2_OUTPUT_DIR / "coldstart_progress.v2.jsonl"


def rft_progress(line: str) -> Path:
    return V2_OUTPUT_DIR / f"rft_progress.{line}.v2.jsonl"


def dpo_progress(line: str) -> Path:
    return V2_OUTPUT_DIR / f"dpo_progress.{line}.v2.jsonl"


def eval_progress(tag: str) -> Path:
    return V2_OUTPUT_DIR / f"{tag}_score_progress.jsonl"   # 评测三分打分增量落盘（k=3 不进缓存，靠它中断不重烧）


def gather_until(items, fn, *, enough, chunk: int, workers: int, desc: str,
                 progress_path=None, key: str = "qid", seed: int = V2_GATHER_SEED) -> list:
    """固定 seed 洗牌后【分批】跑 fn(item)；每批完用 enough(results)->bool 判够没够，够了就停、返回已处理结果。

    ★早停只决定【处理多少题】；每题怎么处理（k=16/n=32/三道门）全在 fn 内、本函数一概不碰。
    ★洗牌 → 早停拿到的是随机代表子集（不是只取前面那些），同 seed 可复现/续跑一致。
    ★给 progress_path：每批结果【增量追加落盘】，重跑时按 `key`(默认 'qid') 跳过已处理项 →
      中断不丢、续跑接着攒（items 与 fn 的结果都须带字段 `key`）。
    ★enough 恒 False（目标=0）→ 跑满全量，退化成原来的 map_concurrent 行为。
    ★落盘是【整批返回后】才做：若预算围栏中途抛/进程被 kill，正在跑的那一批（≤chunk 题）不进 progress，
      续跑会把这≤chunk 题重处理一次——只重烧便宜的 k=2 粗筛（贵的 k=16 已在 kimi_score_cache 命中、不重烧）；
      想缩小这个窗口就调小 V2_GATHER_CHUNK。
    """
    import random
    from pipeline import vllm_client
    from pipeline.logger import get_logger
    log = get_logger("gather_until")

    done = {}
    if progress_path and Path(progress_path).exists():     # 续跑：读已处理结果（容错跳过崩溃残留的半行）
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get(key) is not None:
                    done[r[key]] = r
    results = list(done.values())
    if results and enough(results):
        log.info("%s 续跑：已有 %d 条达标，整步跳过", desc, len(results))
        return results

    pool = [it for it in items if it.get(key) not in done]  # 只处理没做过的
    random.Random(seed).shuffle(pool)
    n_rem = len(pool)
    if done:
        log.info("%s 续跑：已攒 %d 条，剩 %d 题待处理", desc, len(results), n_rem)
    for i in range(0, n_rem, max(1, chunk)):
        batch = vllm_client.map_concurrent(pool[i:i + chunk], fn, workers=workers, desc=desc)
        if progress_path:
            Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "a", encoding="utf-8") as f:  # 增量落盘（崩/停后这批之前的都不丢）
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results.extend(batch)
        if enough(results):
            log.info("%s 早停：攒够目标（本轮处理 %d/%d 剩余题，共 %d 条）", desc, i + len(batch), n_rem, len(results))
            return results
    log.info("%s 跑满（处理全部 %d 剩余题，共 %d 条）", desc, n_rem, len(results))
    return results
