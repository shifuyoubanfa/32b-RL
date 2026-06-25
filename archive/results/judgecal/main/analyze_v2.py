"""judgecal 深化分析：方差(档位×k 分解) + 1/2/3σ 带 + 全档两两可分矩阵。读 161 原始打分。"""
import json, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "161_sentence_judges.jsonl", encoding="utf-8") if l.strip()]

ORDER = ["没抄", "抄1句", "抄2句", "抄3句", "抄4句", "完全照抄"]
EN = ["clean(0)", "copy1", "copy2", "copy3", "copy4", "full"]
ANCHOR = [10, 8, 6, 4, 2, 0]
KS = [1, 2, 4, 8, 16]


def mean(xs): return sum(xs) / len(xs) if xs else 0.0
def var(xs):
    m = mean(xs); return sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0


# 每档：收集每条的 16 分 → item_mean, item_var(内,≈σ_judge²)
lv = {t: {"item_means": [], "item_vars": []} for t in ORDER}
for r in rows:
    sc = [s["clean_score"] for s in r["scores"] if not s.get("error") and s.get("clean_score") is not None]
    if not sc:
        continue
    t = r["true_level"]
    lv[t]["item_means"].append(mean(sc))
    lv[t]["item_vars"].append(var(sc))

stats = {}
for t in ORDER:
    ims = lv[t]["item_means"]
    within_var = mean(lv[t]["item_vars"])          # σ_judge²（单遍打分噪声，k 可压）
    between_var_obs = var(ims)                       # k=16 的档内均值方差（含残余噪声）
    sigma_between2 = max(0.0, between_var_obs - within_var / 16)  # 去掉残余噪声 = 真·样本间方差（k 压不掉）
    sigma_judge = math.sqrt(within_var)
    sigma_between = math.sqrt(sigma_between2)
    spread_k = {k: math.sqrt(sigma_between2 + within_var / k) for k in KS}  # 观测档内std @k
    stats[t] = {"mean": mean(ims), "n": len(ims), "sigma_judge": sigma_judge,
                "sigma_between": sigma_between, "spread_k": spread_k,
                "sd16": spread_k[16]}

# ---------- 图1：观测标准差 vs k（每档一条线） ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for t, en in zip(ORDER, EN):
    ax.plot(KS, [stats[t]["spread_k"][k] for k in KS], marker="o", label=en)
ax.axhline(1.0, ls=":", color="gray"); ax.text(11, 1.05, "half-bin (1.0)", color="gray", fontsize=8)
ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("k (judgments averaged per item)"); ax.set_ylabel("observed within-level std")
ax.set_title("Per-level spread vs k\n(flat tail = sample-to-sample variance, k cannot reduce it)")
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(HERE / "variance_vs_k.png", dpi=130); plt.close()

# ---------- 图2：曲线 + 1/2/3σ 带（k=16） ----------
x = list(range(6))
m = [stats[t]["mean"] for t in ORDER]; s = [stats[t]["sd16"] for t in ORDER]
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(x, ANCHOR, "--", color="gray", marker="s", ms=4, label="ideal (= true level / anchor)")
for mult, alpha, lab in [(3, 0.10, "±3σ"), (2, 0.18, "±2σ"), (1, 0.30, "±1σ")]:
    ax.fill_between(x, [mi - mult * si for mi, si in zip(m, s)],
                    [mi + mult * si for mi, si in zip(m, s)], color="#1f77b4", alpha=alpha,
                    label=lab)
ax.plot(x, m, marker="o", ms=7, lw=2, color="#1f77b4", label="Kimi clean score (mean)")
ax.set_xticks(x); ax.set_xticklabels(EN)
ax.set_xlabel("true copy level"); ax.set_ylabel("Kimi clean score 0-10")
ax.set_title("Kimi clean-score per level, with ±1/2/3σ bands (k=16)")
ax.set_ylim(-2, 11); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig(HERE / "curve_sigma_bands.png", dpi=130); plt.close()

# ---------- 图3：全档两两 σ 距离矩阵 ----------
def pooled(a, b): return math.sqrt((stats[a]["sd16"] ** 2 + stats[b]["sd16"] ** 2) / 2) or 1e-9
D = [[abs(stats[a]["mean"] - stats[b]["mean"]) / pooled(a, b) for b in ORDER] for a in ORDER]
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(D, cmap="RdYlGn", vmin=0, vmax=4)
ax.set_xticks(range(6)); ax.set_xticklabels(EN, rotation=30, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(EN)
for i in range(6):
    for j in range(6):
        d = D[i][j]
        ax.text(j, i, "-" if i == j else f"{d:.1f}", ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")
ax.set_title("Pairwise separation = |Δmean| / pooled_sd  (σ-distance)\n≥2 separable@2σ, ≥3 @3σ; red=can't tell apart")
fig.colorbar(im, ax=ax, label="σ-distance (Cohen's d)")
plt.tight_layout(); plt.savefig(HERE / "separability_matrix.png", dpi=130); plt.close()

# ---------- 写 md ----------
def md_table_header(cols): return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"

var_tbl = md_table_header(["真实档位", "客观锚点", "σ_judge(单遍噪声,k可压)", "σ_between(样本间,k压不掉)"] + [f"k={k}观测std" for k in KS])
for t, a in zip(ORDER, ANCHOR):
    st = stats[t]
    var_tbl += (f"| {t} | {a} | {st['sigma_judge']:.2f} | {st['sigma_between']:.2f} | "
                + " | ".join(f"{st['spread_k'][k]:.2f}" for k in KS) + " |\n")

# 可分矩阵表 + 跨档读出
sep_tbl = md_table_header(["", *EN])
for i, a in enumerate(ORDER):
    sep_tbl += f"| **{a}** | " + " | ".join("—" if i == j else f"{D[i][j]:.1f}" for j in range(6)) + " |\n"

def verdict(d): return "✅可分(≥3σ)" if d >= 3 else ("🟡勉强(2~3σ)" if d >= 2 else "❌分不开(<2σ)")
adj = [(ORDER[i], ORDER[i + 1], D[i][i + 1]) for i in range(5)]
adj_lines = "\n".join(f"- {a}→{b}: σ距离 **{d:.1f}** → {verdict(d)}" for a, b, d in adj)

md = f"""# judgecal 深化分析 · 方差 + σ可分性（按你 4 点重做）

## 0. "客观锚点"是什么意思（先讲清）
**它不是测出来的，是我人为定的"理想分标尺"**：给 6 个真实档位等距赋一个 0-10 分——没抄=10、每多抄一句降 2 分、完全照抄=0。**只用来当"如果 Kimi 完美、它该打的分"那条对照线**（图里的灰虚线）。
**关键：判"两档分不分得开"完全不靠这个锚点**——只比 Kimi 在两档上的实际打分分布（下面第 3 节）。锚点只是让你直观看到"Kimi 比理想偏低多少、在哪压平"。

## 1. 方差：按"档位 × k"拆开（你的第 1 点）
把方差拆成两块：**σ_judge**=同一条反复打的噪声（打几遍 k 能压掉，σ/√k）；**σ_between**=同一档不同样本之间 Kimi 自己就不一致（**k 压不掉、是真天花板**）。观测到的档内 std@k = √(σ_between² + σ_judge²/k)。

{var_tbl}
![每档 std 随 k](variance_vs_k.png)

**读出**：干净端（没抄/抄1）的 σ_between ≈ {stats['没抄']['sigma_between']:.1f}/{stats['抄1句']['sigma_between']:.1f}，**打到 k=∞ 也下不来**——所以顶端的飘不是"打得不够多"，是 Kimi 对干净/轻度样本本身就判得忽高忽低。脏端（抄4/完全）σ 几乎 0（都砸到 0 分）。

## 2. 曲线 + 1/2/3σ 带（你的第 2 点）
![曲线±1/2/3σ](curve_sigma_bands.png)

蓝色三层带 = mean±1σ / ±2σ / ±3σ（k=16）。可以直接看哪两档的带子叠在一起（=分不开）。

## 3. 到底哪档和哪档分得开：全档两两 σ 距离矩阵（你的第 3、4 点）
**判据**：两档可分性 = |两档均值差| ÷ 合并标准差 = **σ距离**（即 Cohen's d）。
- **<2σ**：分不开（带子大面积重叠）；**2~3σ**：勉强；**≥3σ**：干净可分。

{sep_tbl}
![σ距离矩阵](separability_matrix.png)

**相邻档（你问的"1档2档分不开，那1档4档呢"就看这）**：
{adj_lines}

**跨档读出**（矩阵里挑关键的）：
- 没抄 vs 抄2句 = **{D[0][2]:.1f}σ** → {verdict(D[0][2])}；没抄 vs 抄4句 = **{D[0][4]:.1f}σ** → {verdict(D[0][4])}。
- 抄1句 vs 抄2句 = {D[1][2]:.1f}σ（{verdict(D[1][2])}）；抄1句 vs 抄4句 = **{D[1][4]:.1f}σ** → {verdict(D[1][4])}。
- 抄2句 vs 完全照抄 = **{D[2][5]:.1f}σ** → {verdict(D[2][5])}。

## 4. 结论（RL 天花板）
- **相邻档基本都分不开**（只有抄3→抄4 跨过 2σ）；干净端因 σ_between 太大、k 压不掉，**没抄 vs 抄1/抄2 都分不开** → Kimi 会把干净/轻度照抄混作一谈、还误伤干净（阶段7 病）。
- **要拉到隔 2~3 档、且一端是"重度照抄"，才稳过 2σ**（如没抄 vs 抄4、抄2 vs 完全）。
- **对训练**：只能用"基本干净 vs 重度照抄(≥4句)"这种**大间距对子**；0~2 句的细分、以及"干净 vs 轻度"，Kimi 在当前精度下给不出可靠梯度。打更多遍 k 也救不了（瓶颈是 σ_between，不是噪声）。
"""
(HERE / "judgecal_深化分析.md").write_text(md, encoding="utf-8")
print("done: variance_vs_k.png, curve_sigma_bands.png, separability_matrix.png, judgecal_深化分析.md")
