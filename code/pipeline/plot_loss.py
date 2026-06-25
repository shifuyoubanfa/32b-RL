"""画训练曲线：读 swift/transformers 的 trainer_state.json(log_history)。

- 日志里打【文本摘要 + ASCII 曲线】（无依赖，永远能看）；
- 若装了 matplotlib，再存一张 PNG 到 output/<tag>_loss.png；
- GRPO 额外画 reward / kl。

用法: python plot_loss.py --dir <训练输出目录> --tag <cs|rft|dpo|grpo>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR
from pipeline.logger import get_logger

log = get_logger("plot_loss")


def _find_state(d: str):
    """优先 output_dir/trainer_state.json；否则取最新 checkpoint-*/trainer_state.json。"""
    p = Path(d) / "trainer_state.json"
    if p.exists():
        return p
    cks = list(Path(d).glob("**/checkpoint-*/trainer_state.json"))   # swift 存到 v0-时间戳/checkpoint-N/，递归找
    if not cks:
        return None
    return max(cks, key=lambda x: int(x.parent.name.split("-")[-1]))


def _series(lh: list, key: str):
    out = []
    for i, h in enumerate(lh):
        v = h.get(key)
        if isinstance(v, (int, float)):
            out.append((h.get("step", i), float(v)))
    return out


def _spark(vals: list, width: int = 60) -> str:
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    if len(vals) > width:
        n = len(vals)
        vals = [vals[int(i * n / width)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="train")
    args = ap.parse_args()

    st = _find_state(args.dir)
    if not st:
        log.warning("没找到 trainer_state.json in %s（训练未产出？跳过画图）", args.dir)
        return
    lh = json.loads(st.read_text(encoding="utf-8")).get("log_history", [])
    loss, eloss = _series(lh, "loss"), _series(lh, "eval_loss")
    reward, kl = _series(lh, "reward"), _series(lh, "kl")

    log.info("===== [%s] 训练曲线（来自 %s）=====", args.tag, st)
    if loss:
        lv = [v for _, v in loss]
        log.info("[%s] train loss: 首 %.4f → 末 %.4f （最低 %.4f，%d 个记录点）", args.tag, lv[0], lv[-1], min(lv), len(lv))
        log.info("[%s] loss   %s", args.tag, _spark(lv))
    if eloss:
        ev = [v for _, v in eloss]
        log.info("[%s] eval_loss: 首 %.4f → 末 %.4f （最低 %.4f）", args.tag, ev[0], ev[-1], min(ev))
    if reward:
        rv = [v for _, v in reward]
        log.info("[%s] reward 末 %.4f （范围 %.3f~%.3f） %s", args.tag, rv[-1], min(rv), max(rv), _spark(rv))
    if kl:
        kv = [v for _, v in kl]
        log.info("[%s] kl 末 %.4f （峰 %.4f） %s", args.tag, kv[-1], max(kv), _spark(kv))

    # PNG（matplotlib 可选）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        if loss:
            ax.plot([s for s, _ in loss], [v for _, v in loss], color="C0", label="train loss")
        if eloss:
            ax.plot([s for s, _ in eloss], [v for _, v in eloss], "o-", color="C1", label="eval loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right"); ax.set_title(f"{args.tag} training")
        if reward or kl:
            ax2 = ax.twinx()
            if reward:
                ax2.plot([s for s, _ in reward], [v for _, v in reward], color="C2", alpha=0.7, label="reward")
            if kl:
                ax2.plot([s for s, _ in kl], [v for _, v in kl], color="C3", alpha=0.5, label="kl")
            ax2.set_ylabel("reward / kl"); ax2.legend(loc="lower right")
        out = str(Path(OUTPUT_DIR) / f"{args.tag}_loss.png")
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        log.info("[%s] 损失曲线 PNG 已保存 -> %s", args.tag, out)
    except Exception as e:
        log.warning("[%s] PNG 未出（%r）；文本+ASCII 已打印。装 matplotlib 后即可出 PNG。", args.tag, e)


if __name__ == "__main__":
    main()
