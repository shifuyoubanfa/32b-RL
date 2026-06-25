"""derag_v5 探针 · 第六步：汇总两个关键数字 + 生死判断。

数字一(X)：RFT 16 遍自采样的自救率（现在就有没有 RL 信号）。
数字二(Y)：Kimi 改写成功率（就算没信号、能不能靠 SFT 造出来）。
判据(以 train+eval 合并的 all 为准，train 偏乐观仅作参考)：
  X≥0.45 → GO_RL：RFT 已有强化学习信号，SFT→DPO→GRPO 整链值得跑。
  X<0.45 ∧ Y≥0.60 → GO_SFT_FIRST：自采样信号弱，先用 Kimi 改写 SFT 搬中心，再续 RL。
  X<0.45 ∧ Y<0.60 → NO_GO：天花板太低，别烧卡，交报告。
"""

import argparse
import json
from pathlib import Path


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rft_headroom", required=True)
    ap.add_argument("--rewrite_headroom", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--run_id", default="")
    args = ap.parse_args()

    rft = load(args.rft_headroom)
    rw = load(args.rewrite_headroom)
    X = (rft.get("all") or {}).get("rescue_rate", 0.0)
    Y = (rw.get("all") or {}).get("rewrite_clean_rate", 0.0)

    if X >= 0.45:
        verdict, why = "GO_RL", "RFT 自采样已稳定出现干净且不漂移的样本，DPO/GRPO 有正样本可学。"
    elif Y >= 0.60:
        verdict, why = "GO_SFT_FIRST", "RFT 自己出不来，但 Kimi 改写能造出干净样本→先 SFT 搬中心再续 RL。"
    else:
        verdict, why = "NO_GO", "病题里既难自采样出干净版、改写也救不动→天花板太低，建议交报告不烧卡。"

    a_rft = rft.get("all", {})
    lines = [
        "# derag_v5 headroom probe summary",
        "",
        f"- run_id: `{args.run_id}`",
        f"- **verdict: {verdict}**",
        f"- reason: {why}",
        "",
        "## 关键数字",
        "",
        "| 指标 | all | eval(没见过) | train(见过,偏乐观) |",
        "|---|---:|---:|---:|",
        f"| 病题数 | {a_rft.get('n_problems','?')} | {rft.get('eval',{}).get('n_problems','?')} | {rft.get('train',{}).get('n_problems','?')} |",
        f"| **X=RFT 自救率** | **{X:.3f}** | {rft.get('eval',{}).get('rescue_rate','?')} | {rft.get('train',{}).get('rescue_rate','?')} |",
        f"| **Y=Kimi 改写成功率** | **{Y:.3f}** | {rw.get('eval',{}).get('rewrite_clean_rate','?')} | {rw.get('train',{}).get('rewrite_clean_rate','?')} |",
        f"| 平均每16遍合格数 | {a_rft.get('mean_pass_per16','?')} | | |",
        f"| 平均每16遍think干净数 | {a_rft.get('mean_clean_per16','?')} | | |",
        f"| 平均每16遍答案在范围数 | {a_rft.get('mean_insupport_per16','?')} | | |",
        "",
        "## 判据",
        "- X≥0.45 → GO_RL（整链值得跑）",
        "- X<0.45 ∧ Y≥0.60 → GO_SFT_FIRST（先 SFT 搬中心再续 RL）",
        "- X<0.45 ∧ Y<0.60 → NO_GO（天花板太低，交报告）",
        "",
        "## 逐题明细",
        "- RFT 自采样逐题：153_rft_headroom.jsonl（每题 16 遍各自 think_clean/answer_in_support/pass）",
        "- Kimi 改写逐题：154_rewrite_headroom.jsonl（before/after 痕迹数 + 是否改干净 + 新 think）",
        "- 病题清单+脏在哪：150_problems.jsonl",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(
        {"run_id": args.run_id, "verdict": verdict, "reason": why,
         "X_rft_rescue_rate": X, "Y_rewrite_clean_rate": Y,
         "rft": rft, "rewrite": rw}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT verdict={verdict} X={X:.3f} Y={Y:.3f} -> {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
