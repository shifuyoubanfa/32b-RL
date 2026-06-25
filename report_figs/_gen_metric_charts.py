# -*- coding: utf-8 -*-
"""生成 32B 实验报告 1.1 指标变化 的三张图（三件套：kimi think / 规则think / 规则answer）。
数据=冻结500验收集、Kimi k=3。全部五阶段已就位（含 GRPO）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

STAGES = ["V1 基线", "SFT", "RFT", "DPO", "GRPO"]
X = list(range(len(STAGES)))

def base(ax, title, ylim, ypct=False):
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(X)
    ax.set_xticklabels(STAGES, fontsize=12)
    ax.grid(axis="y", color="#e6e6e6", linewidth=1)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if ypct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")

def draw(ax, ys, color, fmt, ring_idx=3):
    ax.plot(X, ys, "-o", color=color, lw=2.6, ms=9, zorder=3)
    for x, y in zip(X, ys):
        ax.annotate(fmt(y), (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=12, fontweight="bold")
    if ring_idx is not None:  # 给"当前最好"那一点套红圈
        ax.scatter([X[ring_idx]], [ys[ring_idx]], s=320, facecolors="none",
                   edgecolors="#c0392b", linewidths=2.4, zorder=4)

# ① kimi think · 干净分（0-10）—— DPO 4.643 与 GRPO 4.601 统计上持平，红圈给 DPO（最高）
fig, ax = plt.subplots(figsize=(8.8, 4.3))
base(ax, "① kimi think · 干净分（0–10，越高越没有“换词复述照抄”）", (0, 8))
draw(ax, [3.140, 4.408, 4.489, 4.643, 4.601], "#2f6fe0", lambda v: f"{v:.3f}", ring_idx=3)
fig.tight_layout(); fig.savefig("32b强化学习/report_figs/m1_clean_score.png", dpi=150); plt.close(fig)

# ② 规则think · 去检索腔通过率（%）
fig, ax = plt.subplots(figsize=(8.8, 4.3))
base(ax, "② 规则think · 通过率（detect_rag_style 无检索腔表面词的占比）", (0, 100), ypct=True)
draw(ax, [2.6, 45.4, 45.6, 49.0, 50.0], "#1f8a70", lambda v: f"{v:.1f}%", ring_idx=4)
fig.tight_layout(); fig.savefig("32b强化学习/report_figs/m2_rule_pass.png", dpi=150); plt.close(fig)

# ③ 规则answer · 在池率（%）+ 0.85 地板线
fig, ax = plt.subplots(figsize=(8.8, 4.3))
base(ax, "③ 规则answer · 在池率（answer_in_v1_pool，验收要求 ≥ 85%）", (60, 100), ypct=True)
ax.axhline(85, color="#c0392b", lw=1.6, ls="--", zorder=1)
ax.text(-0.35, 85.6, "85% 地板", color="#c0392b", fontsize=11, fontweight="bold")
draw(ax, [93.8, 83.0, 85.0, 84.6, 83.8], "#7a5cc6", lambda v: f"{v:.1f}%", ring_idx=2)
fig.tight_layout(); fig.savefig("32b强化学习/report_figs/m3_in_pool.png", dpi=150); plt.close(fig)

print("done: 5-stage charts (with GRPO)")
