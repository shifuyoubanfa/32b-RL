# 88 · corrected-v1 失败诊断与 corrected-v2 技术方案（fable5 输出，交 Codex 落地）

> 输入证据：87 最终表、86 delta 统计与失败样本、85 训练日志与 pair/GRPO 数据样本、
> 本地代码 `pipeline/reward.py`（score_rollout/humanness/answer_drift）、`pipeline/step12_build_dpo_pairs.py`、
> `swift/grpo_reward_plugin.py`、`config.py`（REWARD_C_TRACE=0.34, REWARD_C_COPY=1.0, REWARD_TAU_ACC=0.30）。
> 所有新产物统一 `corrected-v2` 命名空间，不覆盖 corrected-v1 与历史目录。

---

## 一、失败原因诊断（按概率排序）

### 先给结论（裁决）

**当前最大问题是「训练信号错位」，且 DPO 与 GRPO 的错位是同一个根源**：两者共用本地代理分
`reward = R_acc · (0.5 + 0.5 · R_human_surface)`，其中

- `R_acc = 0.5·SequenceMatcher(answer, gold) + 0.5·fact_recall`（`reward.py:160-171`）——**一半是和 V1 金标准答案的字符相似度**；
- `R_human_surface = 1/(1 + 0.34·关键词命中 + 1.0·copy_ratio(think vs 参考))`（`reward.py:126-157`）——只看检索腔关键词和 think 对参考的字面照抄。

这个代理在冷启动前后（h 0.23 vs 0.69 的极端对比）有区分力（s_trace AUC 0.99 是在那个区间测的），
但在 **RFT 之后的策略分布上已经饱和失效**：RFT 模型几乎不再写「参考问答对/检索结果」这类触发词，
Kimi 此时扣分的是**篇章结构**——「从资料向答案归纳」「ref_enumeration」「先罗列后给答案」——正则和 5-gram 看不见这些。
于是组内排序被 R_acc（答案文本相似度）主导，**梯度方向变成了"答案更贴金标准原文"，也就是变相奖励照抄与罗列**。
这同时解释了 GRPO-from-RFT 的全部现象：acc +0.013、humanness −0.007、verbatim_copy 60→63（85 §2）。

**不是工程失败、不是 πref 问题、不是评测口径问题、也还没到数据瓶颈。** 按概率排序的 5 个根因如下。

---

### 根因 1（概率 ~85%）：GRPO 在线 reward 与 Kimi humanness 错位，且 acc 项主动泵向照抄

**机制**（代码层面可证）：
- `grpo_reward_plugin.py` 只注册一个 `HumannessReward`，内部就是 `score_rollout` 的总分——训练日志里
  `reward ≡ rewards/HumannessReward/mean` 完全相等（85 §5 每一步都如此），证明在线 reward 没有别的分量。
- gate=ok 后 `reward = R_acc·(0.5+0.5·R_human)`。对 R_acc 的偏导 ∈ [0.5, 1.0]，对 R_human 的偏导 ≤ 0.5·R_acc。
  RFT 后 trace_hits≈0、copy_ratio 中等，R_human 挤在 ~0.65–0.9 的窄带；而组内 K=8 候选的 R_acc（答案相似度+事实召回）
  差异大得多。**组相对优势几乎由"谁的答案最贴金标准原文"决定。**
- 金标准 answer 本身是 V1 的 RAG 腔产物 → 最贴金标准的候选，其 think 往往是"把参考逐条搬进来再收敛"——
  正是 Kimi 判的 `ref_enumeration / verbatim_copy / 从资料向答案归纳`。

**证据**：
- 85 §5 GRPO 日志：`rewards/HumannessReward/mean` 在 0.46–0.80 间震荡无上行趋势，但终评 Kimi h 反而 −0.007；
  训练内分数与目标指标脱钩的直接铁证。
- 87 最终表：GRPO from RFT acc 0.818→0.831（+0.013），h 0.697→0.690；86：acc_up=26 vs acc_down=19，h_down=63。
- 85 §2 报告 traces 计数：verbatim_copy 60→63、ref_enumeration 16→17（GRPO-from-DPO 更糟：verbatim_copy 68）。
- 86 失败样本 204/46/129/213：base think 是"从问题出发推导"，new think 变成"先复述三种准则分录/逐条罗列税率档"，
  h 0.85→0.3 而 acc 不变或上升——**reward 推的方向和 Kimi 罚的方向重合**。
- 样本 134/92/205（acc↑h↓组）：模型靠把更多参考场景塞进 think 换 acc，结构性照抄但不触发关键词。

**反证/边界**：reward 里毕竟有 0.5·R_human 项，为什么没把照抄压住？因为 copy_ratio 只算 think vs 参考的字面 LCS/5-gram，
对"换说法的结构性罗列"无感；且 1/(1+x) 形状在 copy_ratio 0.2→0.4 区间只掉 ~0.1 分，被 R_acc 的收益盖过。

**还需补的诊断**（最小实验 A1 就是干这个）：在已有 4×224 份 Kimi 判分上算
`Spearman(本地 R_human_surface, Kimi humanness)`。预期 <0.3（同一区间内无区分力），即坐实本根因。

### 根因 2（概率 ~80%，与根因 1 同源）：DPO pair 的偏好轴不是 Kimi humanness，margin 主要来自 acc 相似度

**机制**（`step12_build_dpo_pairs.py:68-79` 可证）：
- chosen = 组内综合奖励最高，rejected = 综合奖励最低，margin 卡的是**综合奖励差**。
- 两端都要求 gate=ok 且 R_acc≥floor → rejected 不是"坏样本"，是"答案离金标准原文稍远的样本"。
  综合奖励差完全可以由 R_acc 差撑满（如 0.9 vs 0.65，双双过 floor），R_human 几乎不参与。
- 所以 DPO 实际学的偏好是「答案文本更贴金标准」≫「think 更像人」。它学不出 humanness 提升是**必然**而非偶然。

**证据**：
- 85 §5 DPO 日志：全程 `loss ≈ 0.69 ≈ ln2`，`rewards/margins` 在 −0.014~+0.019 间正负震荡，
  `rewards/accuracies` 只有 0.23–0.41（**低于 0.5**，即训练末期模型给 rejected 的隐式奖励还经常高于 chosen）——
  这批 pair 对当前策略而言基本是不可学习的噪声方向。
- 86 DPO vs RFT：mean Δh=−0.0074、median=0、h_up 58 vs h_down 63——净效果就是抖动。
- 85 §3 pair 审计：只验了结构和长度（chosen 821 vs rejected 882），**从未用 Kimi 验证过 chosen 的 h 高于 rejected**。
- pair 复用旧 `60_dpo_pairs.jsonl`，构对时 PMI=关（step12 日志格式 `PMI=关`），R_human 就是表面项。

**反证/边界**：chosen 比 rejected 略短（821 vs 882），方向上和"少罗列"一致，说明 pair 不是全反的——只是信噪比太低。

**还需补的诊断**（最小实验 A2）：抽 150 对，chosen/rejected 各送 Kimi 判一次 h，统计
`P(h_chosen > h_rejected)` 与 Δh 分布。预期 ≈0.5（纯噪声）。若 >0.65 则本根因降级。

### 根因 3（概率 ~40%，真实存在但属次要放大器）：优化剂量过保守，策略基本没动

**证据**：GRPO `kl ≈ 1e-4`（85 §5 每步如此），50 步 × lr 5e-7 × 新建 LoRA；DPO 23 步 × 1 epoch。
median Δh = 0.0000、大多数样本逐字不变（86 样本 41 的 think 完全相同）——策略只在尾部样本上动了。

**为什么只排第三**：当前 reward 方向是错的，剂量小反而是**止损**——GRPO 在仅 50 步内已把 verbatim_copy 推高 3 个计数。
先修信号再放量；信号修好后，v2 的 GRPO 需要 lr 1e-6~2e-6、100–150 步、KL 监控带 1e-3~1e-2。

### 根因 4（概率 ~25%）：单判分噪声让 ±0.01 量级的结论不可靠（评测口径的次要问题，非主因）

**证据**：86 三组对比 median Δh 全为 0.0000，mean Δh −0.003~−0.007，量级在单次 Kimi 判分的抖动范围内
（base/new think 完全相同的样本也出现 h 0.7→0.75 级别的判分差，如 86 样本 129 两次 base h 分别 0.85/0.7）。
**为什么不是主因**：同一裁判清楚地照出过冷启动 0.233→0.694 和治根后 grounded 0.855，灵敏度足够照出真实提升；
四次评测口径完全一致。结论"无提升"本身可信，但 v2 验收 Δh<0.03 的结论必须双判（两次判分取均值）。

### 根因 5（概率 ~15%）：部分 query 的金标准答案本身是枚举体，acc 与 humanness 在数据层面局部冲突

**证据**：86 样本 129（印花税税目税率表）、42（高新证书填报）——金标准就是表格/条目，
答案要对就得罗列，think 跟着罗列，Kimi 必扣 `ref_enumeration`。这类 query 在 reward 设计里没有特殊处理。
**边界**：这是少数派（traces 统计里 ref_enumeration 只有 16/224），不构成"数据已到瓶颈"——
h_up 样本每组仍有 53–58 个，说明 K=8 rollout 内部存在更自然的候选，是**信号没把它们选出来**，不是池子里没有。

### 明确不是优先级的方向

| 方向 | 为什么不做 |
|---|---|
| 单纯加步数/加 epoch | DPO loss 钉死在 ln2、margins 围 0 震荡 → 信号问题，不是没学够；GRPO 加步数只会沿错误方向走更远（verbatim_copy 已在涨） |
| 单纯调 beta/lr | KL 1e-4 说明剂量确实小，但方向错时放大剂量 = 加速恶化；剂量调整放在信号修复之后 |
| 再换 reference/πref | corrected-v1 已用合并基座+日志双行验证（`policy_start/pi_ref`），语义正确；这条已经排除 |
| 重跑 corrected-v1 同配置求复现 | 三组实验（DPO、双 GRPO）结论一致，复现意义不大 |
| 怀疑评测/换裁判 | 同裁判照出过 0.23→0.69，照不出 ±0.007 是正常的统计行为；只需双判加固，不需换口径 |

---

## 二、corrected-v2 技术路线

**核心原则：把训练信号从"贴金标准原文 + 不写检索词"改成"Kimi 真实偏好的自然推导，且事实闸住"。**
Kimi 从"只做验收"升级为"离线构造训练信号的标注器"（仍不进在线环路）。
优先级：**DPO v2 是主攻**（离线信号可以直接用 Kimi 真分，零代理误差），GRPO v2 是二段（需要把 Kimi 蒸馏成本地 scorer）。

### 2.1 重做 rollout：是，必须

- 从 `RFT merged base`（当前策略起点）重新 rollout：K=8、temperature 1.0、top_p 0.95、`COLDSTART_SYSTEM_PROMPT`。
- query 取自 `70_grpo_data.jsonl` 的 2015 池（**严格排除 224 评测集**），最小实验 120 条，全量 800–1000 条。
- 理由：60_dpo_pairs 的 rollout 来自旧链路，且其 chosen/rejected 轴错位（根因 2）；on-policy 数据同时服务 DPO 构对与 scorer 蒸馏。

### 2.2 重构 DPO pairs：是，规则全换

每条 rollout 候选先过廉价闸（format_ok、空答案剔除），幸存者**全部送 Kimi 离线判分**（一次调用同出 h/g/acc，
复用 step04 rubric）。然后按以下规则构对：

**chosen 条件**：`acc ∈ {correct, partial}` ∧ `g ≥ 0.7` ∧ `h ≥ 0.7` ∧ `copy_ratio(think, refs) ≤ 0.35`
**rejected 条件**：`acc 与 chosen 同档`（优先；至多放宽到差一档且不为 incorrect）∧ `g ≥ 0.6` ∧ `h ≤ h_chosen − 0.25`
**margin**：`Δh ≥ 0.25`（卡 Kimi 真分，不卡本地代理分）
**去混淆**：`|len_c − len_r| / max(len_c, len_r) ≤ 0.5`（防长度轴）；同档 acc 优先（让 humanness 成为 pair 间唯一系统性差异）

**硬负样本（关键增强）**：失败模式本身就是最好的 rejected——同 query 下
「correct 且 h≥0.7 的自然推导」(chosen) vs「correct 但 ref_enumeration/verbatim 的罗列体」(rejected)。
86 的回归样本证明这类候选在 rollout 里天然大量存在。这直接教模型区分"对而自然"与"对但照抄"。

预期产出率需 pilot 实测：若 120 query 产出 <80 对，则放宽 Δh 到 0.2 或加大 K 到 12。

### 2.3 GRPO reward v2 定义

替换 `score_rollout` 的在线版本（新模块 `pipeline/reward_v2.py`，不动 v1）：

```
若 format 不合规:                      R = −1
若 fact_recall(answer, gold关键事实) < 0.6:   R = 0.1 · fact_recall        # accuracy 硬门（带微斜率）
否则:
    R = (0.3 + 0.7 · H_v2(think))                                          # humanness 主项
        × (0.3 if grounded_fail else 1.0)                                  # grounded 闸（数字/实体臆造检测）
        − 1.5 · max(0, copy_ratio(think, refs ∪ gold_answer) − 0.25)       # 照抄惩罚（含对 gold 的照抄！）
        − 0.5 · max(0, enum_density(think) − 0.3)                          # 罗列/模板惩罚
        − 0.2 · max(0, len(think)/len_p75 − 1)                             # 长度惩罚（p75 取自 RFT base 分布）
```

与 v1 的本质区别：
1. **acc 从"渐变相似度"改成"饱和事实召回门"**。`SequenceMatcher(answer, gold)` 整项移出 reward 核心
   （它就是照抄泵，根因 1 的源头）；fact_recall 对 gold 关键事实（数字、文号、科目名，复用 `_facts`）召回到 1 即饱和，
   **多罗列资料不再加分**。
2. **copy_ratio 的对照集从 refs 扩成 refs ∪ gold_answer**——v1 只查 think 抄参考，没查 think 抄金标准答案。
3. **H_v2 不再是关键词正则**，两档实现：
   - 一期（结构特征版，零训练成本）：`enum_density`（列表行/「第X」「另外」连击密度）、推导连接词分布
     （「既然/那/所以」的位置熵）、答案承诺位置（结论出现在 think 前 1/3 罚）、copy_ratio——
     用 2.2 攒下的 Kimi 真分做 logistic 校准，**探针 AUC ≥ 0.75 才准入**（沿用 step09 探针纪律）。
   - 二期（蒸馏版，treat as 主方案）：用 2.2 的 ~8k 条 Kimi 判分 rollout，在 Qwen2.5-7B-Instruct 上 LoRA 蒸馏
     humanness 回归头；held-out Spearman ≥ 0.6 ∧ AUC(h≥0.75 vs h≤0.45) ≥ 0.8 才接入在线 reward。
4. 乘法门保留（accuracy/grounded 是闸不是项），humanness 是唯一渐变主项——组内优势排序按"准确者中谁最自然"。

**防"罗列换准确率"的三重保险**：fact_recall 饱和（罗列无增益）+ copy/enum 显式负项（罗列有代价）+
H_v2 结构敏感（罗列直接压主项）。配合 DPO 硬负样本，从偏好和在线两侧夹击同一失败模式。

### 2.4 训练剂量与守护

- DPO v2：beta 0.1、lr 5e-6、1 epoch（与 v1 同配置，先只换数据，控制变量）。
  训练中监控 `rewards/accuracies`：**末期 ≥ 0.6 是 pair 质量的过程验证**（v1 只有 0.23–0.41）。
- GRPO v2：从 DPO-v2 merged 起，100–150 步、lr 1e-6、beta 0.04、K=8；KL 目标带 1e-3~1e-2
  （v1 的 1e-4 = 没动，>1e-2 = 风险区）；每 25 步存 ckpt 并跑 60 条 mini 评测（Kimi 判 h/acc），出带回退。
- grounded 守护不变：faithfulness 闸 + 评测 grounded 维度已在链路里，v2 只需在 reward 里保留 grounded 闸。

---

## 三、代码与数据改造清单（全部 corrected-v2 命名）

### 新增脚本（`/mnt/pfs/zhjg/code`，本地 `32b强化学习/code/` 同步）

| 文件 | 作用 |
|---|---|
| `pipeline/step90_audit_reward_alignment.py` | 读 80–83 的 infer+judge JSONL，算本地 R_human/R_acc 与 Kimi h/acc 的 Spearman/分桶表 → 根因 1 实证 |
| `pipeline/step91_audit_dpo_pairs_kimi.py` | 抽 150 对 60_dpo_pairs，chosen/rejected 双侧 Kimi 判分，出 P(h_c>h_r) 与 Δh 分布 → 根因 2 实证 |
| `pipeline/step92_rollout_v2.py` | 参数化 on-policy rollout（--model --queries --k --tag），vLLM TP=8 |
| `pipeline/step93_kimi_score_rollouts.py` | 全候选 Kimi 判分（h/g/acc 一次调用；增量落盘+断点续跑，沿用 step01/06/07 模式；JUDGE_WORKERS=3 起步） |
| `pipeline/step94_build_dpo_pairs_v2.py` | 按 §2.2 规则构对（Kimi 真分 margin + 同档 acc + 长度去混淆 + 硬负样本），出对 + 构对报告 |
| `pipeline/reward_v2.py` | §2.3 的 fact_gate / grounded_gate / H_v2(结构特征版) / copy·enum·len 惩罚；不改动 reward.py |
| `swift/grpo_reward_plugin_v2.py` | 注册 `humanness_v2`，复用 v1 插件的列对齐/防静默骨架 |
| `pipeline/step95_distill_humanness_scorer.py` | （二期）Kimi 分蒸馏 7B scorer + 探针报告，AUC 门 ≥0.8 |
| `scripts/run_corrected_v2_minimal.sh` | 最小实验一键：90→91→92(120q)→93→94→mini DPO→eval |
| `scripts/run_corrected_v2_full.sh` | 全量：92(1000q)→93→94→DPO v2→merge→eval→(GRPO v2→eval) |
| `scripts/monitor_corrected_v2.sh` | 沿用 monitor_merged_dpo_grpo.sh 模式，raw log 分离 |

### 产物与目录（服务器）

- 数据：`/home/nvme01/zhjg/output/90_corrected_v2_reward_alignment.{md,jsonl}`、`91_corrected_v2_pair_audit.{md,jsonl}`、
  `92_corrected_v2_rollouts.jsonl`、`93_corrected_v2_rollout_scores.jsonl`、`94_corrected_v2_dpo_pairs.jsonl` + `94_corrected_v2_pair_report.md`
- LoRA：`/home/nvme01/zhjg/ckpts/v1-32b-corrected-v2-dpo-lora`、`v1-32b-corrected-v2-grpo-lora`
- 合并模型：`/home/nvme01/zhjg/models/v1-32b-corrected-v2-dpo-merged`
- 评测报告：`output/96_corrected_v2_dpo_report.md`、`97_corrected_v2_grpo_report.md`、`98_corrected_v2_summary.md`
- 日志：`/home/nvme01/zhjg/logs/corrected_v2/`
- 预检沿用 corrected-v1 纪律：新输出路径含 `corrected-v2` 才放行；合并走 `.partial`→验证→原子发布；`.done` 完成标记。

---

## 四、最小实验计划（半天～一天，先证伪再放量）

### Phase A：纯诊断，无训练（~2–3 小时，可与 B1 并行）

| 步骤 | 内容 | 成本 | 门槛 |
|---|---|---|---|
| A1 | step90：本地代理分 vs Kimi 分相关性（用已有 4×224 判分，零新调用） | CPU 半小时 | Spearman(R_human, kimi_h) < 0.3 → 坐实根因 1；≥0.5 → 根因 3 升级为主因，v2 改为"原 reward + 放量"路线 |
| A2 | step91：150 对 pair 双侧 Kimi 判分（300 次调用） | ~1.5h @3 workers | P(h_c > h_r) ≤ 0.55 → 坐实根因 2；≥0.65 → pair 可部分复用，v2 只做增量清洗 |

### Phase B：mini 闭环（~半天）

| 步骤 | 内容 | 成本 | 门槛 |
|---|---|---|---|
| B1 | step92/93：120 query × K=8 = 960 候选 rollout + Kimi 判分 | rollout ~1h(TP=8)；判分 ~3h @3 workers | 产出 ≥80 对（step94）；不足则 Δh margin 0.25→0.2 |
| B2 | mini DPO v2（同 v1 超参）→ merge → 224 评测（**双判取均值**） | 训练 ~45min + 评测 ~20min×2 | 见下 |

**B2 Go/No-Go（对照 RFT merged base 0.697/0.858/0.818）**：
- **Go（上全量）**：Δh ≥ +0.02 ∧ Δacc ≥ −0.01 ∧ grounded ≥ 0.84，且训练内 `rewards/accuracies` 末期 ≥ 0.6。
- **灰区（扩数据再判）**：0 < Δh < 0.02 → 把 query 扩到 1000、pair 扩到 ≥1.5k 再训一次，不改规则。
- **No-Go**：Δh ≤ 0 或 acc 掉 >0.015 → 停 DPO 路线，转 scorer 蒸馏 + GRPO v2 单路线；若也不行，
  以 RFT merged base 交付并出具"RL 信号已对准但数据上限"的证据链。

### 全量流程（最小实验 Go 之后，~2–3 天含 API 排队）

1. 92/93 全量：1000 query × K=8 = 8000 候选判分（夜间跑，~10h @4 workers；429 严重则降 3）。
2. step94 构对（目标 ≥1.5k 对，含硬负样本占比 ≥30%）→ DPO v2 → merge → 224 双判评测。
   **验收：Δh ≥ +0.04，acc ≥ 0.81，grounded ≥ 0.85**。
3. （达标后可选）step95 蒸馏 scorer（8k 判分数据现成）→ 探针门 → GRPO v2 100–150 步 → 评测。
   **验收：再 +0.02 h，acc 回撤 ≤0.01**。GRPO 是冲刺项，不是交付必需（14B 先例：GRPO 仅 +0.01）。

---

## 五、给组长的解释话术

**corrected-v1 证明了什么（不是白跑）**：
「这一轮我们把 πref 语义修正为冻结合并基座并全链验证，三组 RL（DPO、双起点 GRPO）在完全正确的训练语义下
都没有提升 humanness——这恰恰是有价值的结论：**问题不在训练器，在训练信号**。我们做了归因：
在线 reward 的准确率项是"与金标准答案的文本相似度"，它在组内排序中占主导，而金标准本身是 RAG 腔的，
所以 RL 实际在优化"答案更贴原文"——GRPO 准确率 +1.3 个点、verbatim_copy 上升、humanness 微降，
三个现象同向，证据闭环。DPO 侧，训练日志 loss 全程钉在 ln2、偏好准确率只有 0.23–0.41，
说明旧偏好对里根本没有可学的 humanness 方向。一句话：**旧 RL 信号没有对准目标，而不是 RL 无效**——
同一条链路上，冷启动+RFT 已经把 humanness 从 0.23 提到 0.70 且 grounded 守在 0.86，方法本身被验证过。」

**v2 改什么**：
「把 RL 信号从"守准+不写检索词"换成"在事实闸住的前提下，直接优化验收裁判（Kimi）定义的自然推导"。
具体三件事：① 偏好对重构——on-policy 重采样，chosen/rejected 用 Kimi 真分卡 margin，且两端准确率同档，
让"自然 vs 罗列"成为 pair 间唯一系统差异；② 在线 reward 重构——准确率从相似度改成饱和的关键事实召回门
（罗列资料不再加分），humanness 改用结构敏感的评分（蒸馏自 Kimi），照抄/罗列/超长显式扣分；
③ 过程验证——每个信号接入前先过探针 AUC 门，训练中盯偏好准确率和 KL 带，出带即回退。」

**预期收益 / 风险 / 回退**：
「先跑一天的最小实验拿方向性证据，不先动全量。预期 DPO v2 全量后 humanness +0.04 以上、acc 回撤 ≤0.01、
grounded ≥0.85；GRPO v2 是冲刺项。风险主要是 Kimi 标注预算（约 1–2 万次调用）和构对产出率，
都有 pilot 实测点。回退方案干净：所有产物在 corrected-v2 命名空间，最坏情况下 RFT merged base
（0.697/0.858/0.818）随时可交付，且我们手里多了一份"RL 信号对准与否"的完整归因报告。」

---

## 六、需要从服务器补充的最少信息（给 Codex/另一个终端）

| # | 要什么 | 为什么 | 命令 |
|---|---|---|---|
| 1 | 80–83 的 per-sample judge JSONL 是否齐全（A1 的输入） | step90 要逐样本对齐本地分与 Kimi 分 | `ls -la /home/nvme01/zhjg/output/ \| grep -E "8[0-3]_corrected_v1.*(judge\|infer)"` |
| 2 | `60_dpo_rollout` 的来源与时间戳（确认旧 pair 的 off-policy 程度） | 决定 A2 结论的解释口径 | `ls -la /home/nvme01/zhjg/output/ \| grep -E "60_dpo"; grep -E "DPO_ROLLOUT\|DPO_MARGIN\|RFT_ACC_FLOOR" /mnt/pfs/zhjg/code/config.py` |
| 3 | DashScope 当前限流/余额（v2 需 1–2 万次 Kimi 调用） | 决定 93 的 worker 数与全量排期 | 跑 20 条试探：`cd /mnt/pfs/zhjg/code && python -c "from pipeline import kimi_client; ..."`（或直接看近期 429 频率日志） |
| 4 | 静态 vLLM TP=8 的生成吞吐（92 的排期依据） | 120q×8 与 1000q×8 的耗时估计 | 起 TP=8 服务后对 20 条 query 计时 |
| 5 | `step04` judge rubric 当前版本是否含 h/g/acc 三维一次性输出 | 93 直接复用可省一半调用 | `grep -n "humanness\|grounded" /mnt/pfs/zhjg/code/pipeline/step04*.py \| head -30` |

> 以上 1/2/5 本地 code/ 目录也能查到大半，服务器侧主要是确认实际部署版本与产物时间戳一致。
