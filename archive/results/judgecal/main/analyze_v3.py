"""judgecal 报告 v3：① 整体评测精度(N×k) ② 每档方差+均值 ③ 均值±1/2/3σ分色带 ④ 可分公式+矩阵 ⑤ 三色热力图。全中文。"""
import json, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / "161_sentence_judges.jsonl", encoding="utf-8") if l.strip()]
ORDER = ["没抄", "抄1句", "抄2句", "抄3句", "抄4句", "完全照抄"]
ANCHOR = [10, 8, 6, 4, 2, 0]
KS = [1, 2, 4, 8, 16]
COLORS = ["#2c7fb8", "#41ab5d", "#d95f0e", "#756bb1", "#dd3497", "#636363"]


def mean(xs): return sum(xs) / len(xs) if xs else 0.0
def var(xs):
    m = mean(xs); return sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0.0


lv = {t: {"means": [], "vars": []} for t in ORDER}
all_item_vars = []
for r in rows:
    sc = [s["clean_score"] for s in r["scores"] if not s.get("error") and s.get("clean_score") is not None]
    if not sc:
        continue
    lv[r["true_level"]]["means"].append(mean(sc))
    lv[r["true_level"]]["vars"].append(var(sc))
    all_item_vars.append(var(sc))

st = {}
for t in ORDER:
    ims, ivs = lv[t]["means"], lv[t]["vars"]
    within = mean(ivs)
    between_obs = var(ims)
    sb2 = max(0.0, between_obs - within / 16)
    st[t] = {"mean": mean(ims), "sj": math.sqrt(within), "sb": math.sqrt(sb2),
             "sd16": math.sqrt(between_obs),
             "spread_k": {k: math.sqrt(sb2 + within / k) for k in KS}}

# 整体评测用的代表性方差(各档均方根)：σ_judge_p、σ_between_p
sj_p = math.sqrt(mean([st[t]["sj"] ** 2 for t in ORDER]))
sb_p = math.sqrt(mean([st[t]["sb"] ** 2 for t in ORDER]))
def sigma_total(k): return math.sqrt(sb_p ** 2 + sj_p ** 2 / k)
NS = [50, 100, 200, 224, 500]
def se(N, k): return sigma_total(k) / math.sqrt(N)

# ---------- 图1：整体评测 SE 随 N、k ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
Nx = list(range(30, 521, 10))
for k, c in [(1, "#d95f0e"), (3, "#41ab5d"), (16, "#2c7fb8")]:
    ax.plot(Nx, [se(N, k) for N in Nx], color=c, lw=2, label=f"每条打 k={k} 次")
ax.axhline(0.1, ls=":", color="gray"); ax.text(420, 0.105, "SE=0.1 参考线", color="gray", fontsize=9)
ax.set_xlabel("评测样本量 N（题数）"); ax.set_ylabel("模型平均干净分的标准误 SE")
ax.set_title("整体评测精度：模型平均干净分的标准误 SE\n(N 越大越准；k 几乎不影响——因为瓶颈是样本间方差 σ_between)")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(HERE / "fig1_整体评测SE.png", dpi=130); plt.close()

# ---------- 图2：每档观测std 随 k（分色） ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for t, c in zip(ORDER, COLORS):
    ax.plot(KS, [st[t]["spread_k"][k] for k in KS], marker="o", color=c, label=t)
ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("打分次数 k（同一条取平均）"); ax.set_ylabel("该档观测标准差")
ax.set_title("每档标准差随 k 的变化\n(尾部压不平 = 样本间方差 σ_between，多打几遍也降不下来)")
ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=9)
plt.tight_layout(); plt.savefig(HERE / "fig2_每档std随k.png", dpi=130); plt.close()

# ---------- 图3：均值 + ±1/2/3σ，每档分色 ----------
fig, ax = plt.subplots(figsize=(9, 5.4))
x = list(range(6))
ax.plot(x, ANCHOR, "--", color="gray", marker="s", ms=5, label="理想线（= 真实档位）")
for i, (t, c) in enumerate(zip(ORDER, COLORS)):
    m, s = st[t]["mean"], st[t]["sd16"]
    ax.plot([i, i], [m - 3 * s, m + 3 * s], color=c, lw=1.2, alpha=0.5)      # ±3σ 细
    ax.plot([i, i], [m - 2 * s, m + 2 * s], color=c, lw=3, alpha=0.5)        # ±2σ 中
    ax.plot([i, i], [m - 1 * s, m + 1 * s], color=c, lw=6, alpha=0.6)        # ±1σ 粗
    ax.plot(i, m, "o", color=c, ms=11, zorder=5, label=f"{t}（均值{m:.1f}）")
    for mult in (1, 2, 3):  # ±Nσ 端点小横线，便于读"X档均值是否落在Y档Nσ内"
        for sign in (1, -1):
            ax.plot([i - 0.12, i + 0.12], [m + sign * mult * s] * 2, color=c, lw=1, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_xlabel("真实照抄档位"); ax.set_ylabel("Kimi 干净分 0-10")
ax.set_title("各档 Kimi 干净分：均值（圆点）+ ±1σ(粗)/±2σ(中)/±3σ(细)\n每档一种颜色；竖条越短=越稳，竖条重叠=分不开")
ax.set_ylim(-3, 11); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="upper right", ncol=2)
plt.tight_layout(); plt.savefig(HERE / "fig3_均值与σ带.png", dpi=130); plt.close()

# ---------- 图4：两两可分 三色热力图（判据=A、B 的 Nσ 带不相交）----------
# 可分(A比B干净) ⟺ |均值_A−均值_B| > N·(σ_A+σ_B)。格子 = |均值差|/(σ_A+σ_B)：≥3 即 3σ 带不相交。
def ratio(a, b):
    denom = st[a]["sd16"] + st[b]["sd16"]
    return abs(st[a]["mean"] - st[b]["mean"]) / denom if denom > 1e-9 else 99.0
D = [[ratio(a, b) for b in ORDER] for a in ORDER]
fig, ax = plt.subplots(figsize=(7.4, 6.2))
cmap = ListedColormap(["#d73027", "#fee08b", "#1a9850"])  # 红/黄/绿
norm = BoundaryNorm([0, 2, 3, 1e9], cmap.N)
ax.imshow(D, cmap=cmap, norm=norm)
ax.set_xticks(range(6)); ax.set_xticklabels(ORDER, rotation=30, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(ORDER)
for i in range(6):
    for j in range(6):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=11); continue
        d = D[i][j]
        word = "3σ可分" if d >= 3 else ("2σ勉强" if d >= 2 else "分不开")
        ax.text(j, i, f"{d:.1f}\n{word}", ha="center", va="center", fontsize=8.5, fontweight="bold")
ax.set_title("两两档位可分性：A、B 的 Nσ 带不相交，即 |均值差| > N×(σ_A+σ_B)\n"
             "格子=|均值差|/(σ_A+σ_B)　绿≥3(3σ带不相交,可分)　黄2~3(只2σ)　红<2(分不开)", fontsize=10)
plt.tight_layout(); plt.savefig(HERE / "fig4_可分矩阵.png", dpi=130); plt.close()

# ---------- 写 md ----------
def hdr(cols): return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"

se_tbl = hdr(["样本量 N", "k=1", "k=3", "k=16"])
for N in NS:
    tag = " (验收集)" if N == 224 else ""
    se_tbl += f"| {N}{tag} | {se(N,1):.3f} | {se(N,3):.3f} | {se(N,16):.3f} |\n"

var_tbl = hdr(["真实档位", "客观锚点", "Kimi均值", "σ_judge(单遍噪声,k可压)", "σ_between(样本间,k压不掉)"]
              + [f"k={k}观测std" for k in KS])
for t, a in zip(ORDER, ANCHOR):
    s = st[t]
    var_tbl += (f"| {t} | {a} | {s['mean']:.2f} | {s['sj']:.2f} | {s['sb']:.2f} | "
                + " | ".join(f"{s['spread_k'][k]:.2f}" for k in KS) + " |\n")

sep_tbl = hdr([""] + ORDER)
sym = lambda d: "✅" if d >= 3 else ("🟡" if d >= 2 else "❌")
symw = lambda d: "3σ可分" if d >= 3 else ("2σ勉强" if d >= 2 else "分不开")
for i, a in enumerate(ORDER):
    sep_tbl += f"| **{a}** | " + " | ".join("—" if i == j else f"{D[i][j]:.1f}{sym(D[i][j])}" for j in range(6)) + " |\n"

md = f"""# judgecal 实验报告 · Kimi 换词复述分辨率（全中文重做）

> 数据：78 条 think（13 个参考 × 6 档，每档 13 条），每条 Kimi 打 0-10 干净分 16 遍。
> "干净分"：10=完全没换词复述照抄，0=整段都是照抄。

## 0. "客观锚点"是什么
人为定的"理想分标尺"：没抄=10、每多抄一句降 2 分、完全照抄=0。**只当对照线**（图里灰虚线），看 Kimi 比理想偏多少。**判两档分不分得开完全不靠它**，只比 Kimi 的实际打分分布（第 3、4 节）。

## 1. 整体评测精度（N×k）—— 以后比模型阶段用
评测一个模型整体干净度 = 让 Kimi 给 N 条各打 k 次、取总平均。这个平均值有多准（标准误 SE）：

> **公式**： SE = σ_total / √N ， 其中 σ_total = √(σ_between² + σ_judge²/k)
> 代入本次实测代表值 σ_between≈{sb_p:.2f}、σ_judge≈{sj_p:.2f}（各档均方根）。

{se_tbl}
**怎么用 / 关键结论**：
- 两个模型阶段的平均干净分，差距要 **> 约 3×SE** 才算真涨（不是噪声）。例：N=224、k=1 时 SE≈{se(224,1):.3f}，所以**两阶段平均差 > ~{3*se(224,1):.2f} 分**才可信。
- **k 几乎不影响整体评测**（N=224 时 k=1→16 只把 SE 从 {se(224,1):.3f} 压到 {se(224,16):.3f}）——因为整体评测的瓶颈是样本间方差 σ_between，它靠 √N 摊薄、不靠 k。**所以整体评测用大 N、小 k（k=1~3）就够，别浪费 16 倍调用。**
- 注意这跟"逐条选样"相反：选 DPO 对子是逐条判，σ_between 没 N 可摊、k 也压不掉（见第 2 节）。

## 2. 每档方差（含均值）+ 随 k 变化
两块方差：**σ_judge**=同一条反复打的噪声（k 能压）；**σ_between**=同档不同样本 Kimi 自己就不一致（**k 压不掉、真天花板**）。观测 std@k=√(σ_between²+σ_judge²/k)。

{var_tbl}
![每档 std 随 k](fig2_每档std随k.png)

**读出**：干净端（没抄/抄1）σ_between≈{st['没抄']['sb']:.1f}/{st['抄1句']['sb']:.1f}，k=1→16 几乎没降——顶端飘是 Kimi 对干净样本本身忽高忽低，多打也救不了。

## 3. 各档均值 + ±1/2/3σ（分色，看重叠）
![均值与σ带](fig3_均值与σ带.png)

每档一种颜色：圆点=均值，竖条=±1σ(粗)/±2σ(中)/±3σ(细)。**判据就是看两档的 ±3σ 竖条相不相交——不相交才算可分**。一眼看：没抄/抄1/抄2 的竖条大片重叠（分不开）；只有跟竖条几乎缩成一点的 抄4/完全照抄 比，才不相交。

## 4. 两两可分：公式 + 矩阵（判据已改严）
> **判据（A 比 B 干净，要两条 3σ 带不相交）**：可分 ⟺ |均值_A − 均值_B| > 3·(σ_A + σ_B)。
> 矩阵格子 = **|均值差| ÷ (σ_A + σ_B)**（σ 取 k=16 档内标准差，即上表 k=16 列）：**≥3 = 3σ 带不相交(可分✅)，2~3 = 只够 2σ(勉强🟡)，<2 = 分不开❌**。
> 注：这比 Cohen's d 严——d 只看均值隔几个合并 σ、3σ 尾巴仍重叠；这里要求两条 3σ 带整段不沾（误判 ~0.1%）。

{sep_tbl}
**你问的"1档2档分不开，那1档4档呢"**：抄1↔抄2 = {D[1][2]:.1f}（{symw(D[1][2])}）；抄1↔抄4 = {D[1][4]:.1f}（{symw(D[1][4])}——只够 2σ、3σ 不够）。
**相邻档里只有 抄3↔抄4 = {D[3][4]:.1f}（{symw(D[3][4])}，因两端 σ 都极小）可分**；没抄↔抄1 {D[0][1]:.1f}、抄1↔抄2 {D[1][2]:.1f}、抄2↔抄3 {D[2][3]:.1f}、抄4↔完全 {D[4][5]:.1f} 全分不开。
**3σ 可分的全靠"抄4/完全(σ≈0)"那端**：没抄↔抄4={D[0][4]:.1f}✅、没抄↔完全={D[0][5]:.1f}✅、抄2↔完全={D[2][5]:.1f}✅、抄3↔完全={D[3][5]:.1f}✅。

## 5. 可分性热力图（三色）
![可分矩阵](fig4_可分矩阵.png)

**红=分不开(<2σ)，黄=勉强(2~3σ)，绿=可分(≥3σ)**。格子里写了 σ距离 和判定。对角线是自己跟自己(—)。

## 6. 结论（RL 天花板，按 3σ 带不相交）
- **相邻档里只有 抄3↔抄4 可分**（两端 σ 都极小）；干净端 没抄↔抄1↔抄2 互相全分不开，没抄连抄3 都只够 2σ。Kimi 还会误伤干净（阶段7 病）。
- **抄1句最废**：σ 太大(1.63)，连跟"完全照抄(0 分)"都不够 3σ（2.85）。
- **3σ 可分、又能当 DPO（chosen 干净 vs rejected 脏）的，基本只剩一种：没抄(chosen,分≥~7) vs 抄4句/完全照抄(rejected,分≈0)**。中间一律不用。
- **整体评测（比阶段）不怕这个**：大 N 摊薄即可、k=1~3 足够（§1）。被 σ_between 卡死的是"逐条选对子"。
"""
(HERE / "judgecal_报告.md").write_text(md, encoding="utf-8")
print("done: fig1_整体评测SE.png fig2_每档std随k.png fig3_均值与σ带.png fig4_可分矩阵.png judgecal_报告.md")
print(f"sigma_between_pooled={sb_p:.3f} sigma_judge_pooled={sj_p:.3f}")
