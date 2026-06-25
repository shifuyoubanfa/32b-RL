"""一次性：用缓存的冷启动改写(coldstart_progress)重挑 SFT 数据为【k2-pass-930】（零额外 Kimi）。

多agent调查结论(2026-06-19，落到历史实验+代码)：
- σ-SFT(472)规则只 8%、干净 flat，根因=【量/总步数不足】(472×5ep≈37步)，不是数据不够干净(改写已 7.11)。
- 历史唯一一次规则真涨=最初冷启动 SFT：1348条/5ep(~110步)，humanness 0.31→0.68、显式痕迹 222→31，一刀压下去；
  之后 RFT/DPO/GRPO 全 flat。结论：规则靠【足量(~1k+)干净 SFT 灌一刀】，不靠 RL、不靠堆更高干净度。
- broad-facts(1168) 掺了 238 条 k2没过(Kimi 没判它比 V1 干净)的样本，会稀释 kimi-think 天花板、拖累 RFT 采样(用户担心成立)。
- 加 epoch 救不了少样本：eval_loss epoch2 即收敛，472 上堆高 epoch 会过拟合被 load_best 回退。旋钮是【量】不是 epoch。

故取【k2-pass-930】= 规则门 ∧ facts门 ∧ k2门(s_clean非空=Kimi判比V1干净)：既有量(翻规则)、又每条都有好 kimi-think
(剔掉 238 条平庸样本)。配 COLDSTART_EPOCHS=7(930×7÷64≈100步，对齐历史~110；930 量足，7ep 过拟合风险低)。

跑法：export V2_TAG=derag2; python rebuild_coldstart_sft.py
  → train2(2s)=k2-pass(~930去eval) 单 SFT 线；train3(3s)=空(sft_node 自动跳过)。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import COLDSTART_EVAL_FRAC
from pipeline.v2_paths import (coldstart_progress, coldstart_train, coldstart_eval,
                               sft_row, read_jsonl, write_jsonl)


def main():
    prog = read_jsonl(coldstart_progress())
    # k2-pass：s_clean 非空(过了 k2 粗筛=Kimi 判改写比 V1 干净) → 隐含也过了规则门∧facts门(它们在 cleaner_scores 之前)
    k2 = [r for r in prog if r.get("s_clean") is not None and r.get("rule_ok")
          and r.get("facts_ok") and r.get("natural") and r.get("qid")]
    k2.sort(key=lambda r: r["qid"])                         # 按 qid 稳定排序：eval 留出确定可复现
    every = max(2, round(1 / COLDSTART_EVAL_FRAC))
    eval_qids = {r["qid"] for i, r in enumerate(k2) if i % every == 0}

    def rows(items):
        return [sft_row(r["user_prompt"], r["natural"], r["answer"], query=r.get("query")) for r in items]

    train2 = [r for r in k2 if r["qid"] not in eval_qids]
    write_jsonl(coldstart_train(2), rows(train2))           # 2s = k2-pass（主线，有量+kimi-think保证）
    write_jsonl(coldstart_train(3), [])                     # 3s = 空（单线：sft_node 见空桶自动跳过）
    write_jsonl(coldstart_eval(), rows([r for r in k2 if r["qid"] in eval_qids]))

    import statistics
    sc = [r["s_clean"] for r in k2]
    print(f"读取 progress {len(prog)} 条")
    print(f"k2-pass(规则∧facts∧k2，每条Kimi比V1干净) {len(k2)} 条，s_clean均值 {statistics.mean(sc):.2f}（应~7）")
    print(f"  → train2(2s)={len(train2)}  eval留出={len(eval_qids)}  train3(3s)=0(单线)")
    print(f"  写出: {coldstart_train(2)} / {coldstart_train(3)}(空) / {coldstart_eval()}")
    print(f"提示：重起前请 export COLDSTART_EPOCHS=7（930×7÷64≈100步，对齐历史成功的~110步）")


if __name__ == "__main__":
    main()
