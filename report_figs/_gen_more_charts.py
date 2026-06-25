# -*- coding: utf-8 -*-
"""额外图表：m4 两版对比(剔格式失败)、m5 DPO vs GRPO(噪声带)、m6 裁判分辨率阶梯(r01 例子)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, TEAL, PURPLE, RED, GREY = "#2f6fe0", "#1f8a70", "#7a5cc6", "#c0392b", "#9aa0a6"

# ================= m4：剔除格式失败前后（只有 DPO/GRPO 会变；基线/SFT/RFT 无格式失败）=================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
panels = [
    ("kimi think 干净分", [(4.643, 4.637), (4.601, 4.597)], (4.4, 4.75), "{:.3f}", None),
    ("规则think 通过率", [(49.0, 49.1), (50.0, 50.2)], (44, 54), "{:.1f}%", None),
    ("规则answer 在池率", [(84.6, 84.8), (83.8, 84.1)], (80, 88), "{:.1f}%", 85),
]
labels = ["DPO", "GRPO"]
for ax, (title, data, ylim, fmt, hline) in zip(axes, panels):
    x = np.arange(2); w = 0.36
    raw = [d[0] for d in data]; excl = [d[1] for d in data]
    b1 = ax.bar(x - w/2, raw, w, label="原版(含格式失败)", color=BLUE if "kimi" in title else (TEAL if "规则think" in title else PURPLE))
    b2 = ax.bar(x + w/2, excl, w, label="剔除格式失败", color="#b9c7ec" if "kimi" in title else ("#a9d8cb" if "规则think" in title else "#cbbdec"))
    for bars, vals in [(b1, raw), (b2, excl)]:
        for r, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (r.get_x()+r.get_width()/2, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
    if hline:
        ax.axhline(hline, color=RED, ls="--", lw=1.3); ax.text(-0.45, hline+0.15, "85%地板", color=RED, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11); ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#ededed"); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=9, loc="upper right", framealpha=0.9)
fig.suptitle("剔除格式失败前后对比（仅 DPO/GRPO 含格式失败；基线/SFT/RFT 无变化）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("32b强化学习/report_figs/m4_two_version.png", dpi=150); plt.close(fig)

# ================= m5：DPO vs GRPO 头对头 + 噪声带（看是不是统计打平）=================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
m5 = [
    ("kimi think 干净分", 4.643, 4.601, 0.15, (4.3, 4.9), "{:.3f}", "真涨门 3×SE≈0.15"),
    ("规则think 通过率", 49.0, 50.0, 2.2, (44, 56), "{:.1f}%", "二值噪声 SE≈2.2pp"),
    ("规则answer 在池率", 84.6, 83.8, 1.6, (80, 88), "{:.1f}%", "二值噪声 SE≈1.6pp"),
]
for ax, (title, dpo, grpo, noise, ylim, fmt, note) in zip(axes, m5):
    x = np.arange(2)
    bars = ax.bar(x, [dpo, grpo], 0.5, color=[PURPLE, "#e0992f"])
    for r, v in zip(bars, [dpo, grpo]):
        ax.annotate(fmt.format(v), (r.get_x()+r.get_width()/2, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=11, fontweight="bold")
    # 以 DPO 为中心画噪声带
    ax.axhspan(dpo - noise, dpo + noise, color="#cfcfcf", alpha=0.45, zorder=0)
    ax.text(1.5, dpo, "  ±噪声", va="center", fontsize=9, color="#666")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["DPO", "GRPO"], fontsize=11); ax.set_ylim(*ylim)
    ax.set_xlabel(note, fontsize=9.5, color="#444")
    ax.grid(axis="y", color="#ededed"); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.suptitle("DPO vs GRPO：三件套差值全部落在灰色噪声带内 = 统计上打平", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("32b强化学习/report_figs/m5_dpo_vs_grpo.png", dpi=150); plt.close(fig)

# ================= m6：裁判分辨率阶梯（r01 例子的 6 档：Kimi 实测 vs 理想锚点）=================
levels = ["没抄", "抄1句", "抄2句", "抄3句", "抄4句", "完全照抄"]
anchor = [10, 8, 6, 4, 2, 0]
kmean = [7.34, 4.64, 2.64, 1.84, 0.03, 0.00]
kstd = [1.93, 1.69, 0.62, 0.17, 0.05, 0.00]
fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(6)
ax.plot(x, anchor, "--", color=GREY, lw=1.8, marker="s", ms=7, label="理想锚点（裁判该打的分）")
ax.errorbar(x, kmean, yerr=kstd, fmt="-o", color=BLUE, lw=2.4, ms=9, capsize=5,
            label="Kimi 实测均值 ± 标准差（每条打16遍）")
for xi, v in zip(x, kmean):
    ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 13), ha="center", fontsize=11, fontweight="bold")
# 标出"分得开/分不开"
ax.axvspan(-0.4, 2.4, color="#fdecea", alpha=0.7, zorder=0)
ax.text(1.0, 9.2, "干净端：相邻档分不开\n(σ大、Kimi忽高忽低)", ha="center", fontsize=10, color=RED)
ax.text(4.5, 6.2, "重度照抄端：\n砸到0、稳、分得开", ha="center", fontsize=10, color=TEAL)
ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=12); ax.set_ylim(-0.5, 10.5)
ax.set_ylabel("干净分（0–10）", fontsize=12)
ax.set_title("裁判分辨率阶梯（例：增值税小规模免税题，越往右照抄越多）", fontsize=14, fontweight="bold")
ax.grid(axis="y", color="#ededed"); ax.set_axisbelow(True); ax.legend(fontsize=10, loc="center right")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("32b强化学习/report_figs/m6_dirtiness_ladder.png", dpi=150); plt.close(fig)

print("done: m4_two_version / m5_dpo_vs_grpo / m6_dirtiness_ladder")
