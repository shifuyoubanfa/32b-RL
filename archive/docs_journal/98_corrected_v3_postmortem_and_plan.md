# 98 · corrected-v2 mini DPO 失败复盘与 corrected-v3 最小方案（多 agent 复盘终稿）

> 方法：4 个调查 agent（裁判噪声统计 / 数据池审计 / 训练审计 / 验收产品）并行实算本地原始数据
> （95_rejudge.jsonl 50 对×2 判、94_meta.jsonl 112 对、80_judge.jsonl 224 条），再由反方 Reviewer 红队攻击，主审仲裁。
> 所有统计数字均为 Python 实算，非估计。本地工件是首轮快照（112 对/50 复判），终轮（196 对/63 stable）在云机，分布一致（stable 率 34% vs 32%）。

---

## 一、读取的文件与关键事实

读取：94/95/96/97 号报告与原始 jsonl、95 decision、code（run_corrected_v2_mini.py、step93/94/95/95b、judge_common.py、dpo_v2.sh）、背景 85/86/88/89/90/91、80_judge.jsonl。

**实算出的关键事实**（按颠覆性排序）：

1. **Kimi 裁判单次判分噪声 σ_h=0.117，test-retest r=0.434**（100 文本×2 判，judge temp=0.0）。同文本两判 |Δh| 均值 0.150，24% 的文本差 ≥0.30。grounded σ=0.048（可靠），acc_score σ=0.074，acc tier 同文本翻档率 10%。
2. **h 是 ~6 档离散量表且 0.85 封顶**：37% 判分恰为 0.75；base 224 集中 0.75 占 49.1%；判分中 0.9+ 为零。
3. **h 本质是痕迹检测器**：base 224 上 pearson(h, 痕迹总数)=−0.804；OLS：explicit_ref −0.067/policy_source −0.072/verbatim_copy −0.135/ref_enumeration −0.198（截距 0.779）。**无痕迹样本（123 条）的 h 分布 sd 仅 0.055，挤在 0.75↔0.85 一档**——"结构性更像人"在该 rubric 下不可见。整个 de-RAG 路线在此尺子下的理论收益上限 ≈ +0.073（且全部来自去痕迹）。
4. **pair margin 2/3 是选择性回归吃掉的噪声**：构对 margin +0.403 → 复判 +0.128（收缩 −68%）；chosen 0.788→0.701、rejected 0.385→0.573（反向回归，排除裁判漂移）。但符号检验 win31:loss6:tie13（p<1e-4）——**真 margin 为正 ≈+0.13，池子不是空的，只是优势被高估 3 倍**。
5. **direction 59.2%/56% 的失败一半是量化平手**：22 个失败里 10 个仅因复判 h 精确平手（0.75 簇）+ step95 严格 `h_margin>0`；平手计 pass 则 76%。h 方向真翻转主要集中在硬负样本（19 个 h 失败中 14 个；硬负 direction_ok 仅 45.2% vs 普通对 73.7%）。
6. **pair 实际教的不是目标结构**：chosen/rejected think 相似度均值 0.70；62.5% 双方都 correct；真正含 RAG 叙事腔的 rejected 仅 ~5%；表面引用特征（《》/文号/根据）两侧几乎相同。**15/112 chosen 含 `<img>` 图床链接、27/112 含"参考文件"罗列**（step94 的 BAD_CHOSEN_PHRASES 只查"图片链接"字面短语，漏检）——正向样本在反向教 RAG 残留。stable 对里 chosen 系统性更短（15/17，均长 945 vs 1123）。
7. **设计效应量 +0.006**：用 OLS 痕迹系数 × mini 实际达成的痕迹变化（−11/−12/+3/−1）换算，预期 Δh≈+0.0062——比单判 MDE（0.022）低一个量级。**这轮实验开训前就注定读不出结果，与 pair 数量无关。**
8. **"策略动了"本身未过显著性**（红队纠偏）：痕迹计数重测噪声 SE≈6-8/224 集，explicit_ref −11 → z≈−1.75，policy_source −12 → z≈−1.69，均 <2σ。
9. **Δh=−0.009 完全不可判读**：z=−0.67~−0.82，95%CI=[−0.035,+0.017]；Δacc=−0.012 z=−0.46~−1.73；correct −10 在同文本零假设下 z=−2.23、不同文本零假设下 z≈−1.7——**唯一可疑读数，未定罪**，需 per-query McNemar（数据在云机）。
10. **三个工程门失效**：headroom 门 fail-open（step94:292 用 pool query 匹配 224 评测集 baseline，交集恒空 → NA → :300 把 NA 当 pass，该门从设计上就不可能触发）；rejudge NO-GO 被 run_corrected_v2_mini.py:321-322 降级为 WARN 续跑（违反 89 号方案的硬停设计，且门槛 80%→70%→实测 59.2% 还放行）；step95b 在同一份噪声测量上先选样后汇报（mean_h=+0.301 虚高，后验真值 ≈+0.2）。
11. 训练侧：53 对 ×ga2×3epoch = **约 12 个优化步**（89 方案最低剂量的 55%；"<100 对降 ga=1"未实现）；rpo_alpha A/B 未跑；heldout pref-acc 从未计算（脚本不存在，且 n=10 无功效，区分 0.7 vs 0.5 需 n=40）；坍缩监控未实现。
12. 工件版本失控：本地 94/95 是首轮（35 对 NO-GO / 112 对 / 50 复判），97 引用的是终轮（171/196/63）——同名覆盖至少三代，headline 数字本地不可复算。

---

## 二、多 agent 复盘结论（每个 agent 核心判断）

- **裁判噪声统计 Agent**：σ_h=0.117 经两条路径主导失败——构对在噪声尾部做极值选择（margin 真值仅 +0.13），验收单判 MDE≈0.022 读不出任何设计内效应。k=3 取均值可把 margin 噪声 0.165→0.095、方向保持率 62-74%→78-86%、correct 计数 sd 4.49→1.70。归因：噪声及选择效应 50%。
- **数据/候选池审计 Agent**：chosen（best-of-8）复判后 0.701≈base 0.697，池内"表观提升"几乎全是噪声高估；pair 教的是"删触发词+变短"，目标结构对比只占 ~5%；chosen 自带 `<img>`/参考文件残留在反向教学。归因：池上限 40% / 噪声选择 30% / pair 表面化 20%。
- **训练审计 Agent**：不是"没动"而是"学了表面标记但 h 不涨、可能压了正确长答案"（rejected 60-79% 本身 correct、无 NLL 锚、chosen 偏短）；计划内的四道训练保障（ga 降档/rpo A/B/heldout 门/坍缩监控）全部缺失。最小可信训练规模 ≥300 stable 对、≥20 优化步，低于即不开训。
- **验收/产品 Agent**：诚实三态结论是"**无显著变化（读出系统不具备判别力）**"；h 实测 sd=0.142、unpaired MDE≈0.038；裁判 r=0.43 的尺子既不能认证 pair 也不能读模型差；headroom 门 fail-open 是"唯一能提前发现根因的闸被设计成静默通过"。归因：测量/验收体系 55%。
- **反方 Reviewer（关键纠偏）**：①"痕迹下降"本身只有 1.7σ，全员奠基的"硬事实"未定罪；②"headroom≈0"是跨人群过度结论，符号检验证明真 margin 为正；③ h 量表 0.85 封顶 + 去痕迹后 sd=0.055——**rubric 天花板才是被低估的主因（35%）**；④ 设计效应 +0.006 < 一切 MDE，"53 对太少"接近伪命题；⑤ 成对判分有长度偏差（chosen 88% 更短）/位置偏差/与绝对分体系不可比三重风险，采用前必须做负对照。

---

## 三、统一归因排序（仲裁后）

| 排名 | 根因 | 权重 | 决定性证据 |
|---|---|---|---|
| 1 | **目标定义/rubric 测量天花板**：h=痕迹检测器（r=−0.804），0.85 封顶、去痕迹后只剩一档分辨率；"结构性像人"不可见；设计效应 +0.006 < MDE | ~30% | 事实 3/7 |
| 2 | **裁判噪声 + 噪声上的极值选择**：σ_h=0.117、r=0.43；margin 2/3 是噪声；direction 失败半数是量化平手；95b 双重浸入虚报 +0.301 | ~25% | 事实 1/4/5/10 |
| 3 | **pair 内容错位**：教"删触发词+变短"，~5% 才是目标结构对比；chosen 带 `<img>`/参考文件反向教学；62.5% both-correct 近重复对压制正确答案 | ~20% | 事实 6 |
| 4 | **验收统计功效不足**：单判 MDE 0.022-0.038，Δh=−0.009/correct−10 全部不可判读 | ~15% | 事实 9 |
| 5 | **训练剂量与保障缺失**：12 步、无 NLL 锚、门控旁路；但"53 对太少"不是主因——效应量瓶颈在上游 | ~10% | 事实 11 |
| — | 候选池上限：真 margin +0.13 存在（不是空池），但最优候选挤在裁判量表天花板——**用当前尺子不可判定**，需 v3 对照 B 拆解 | 待测 | 事实 4 + 红队 ③ |

对用户六个假设的直接回答：候选池=部分成立但被尺子遮蔽不可判定；pair 构造=成立（教错了东西+漏检残留）；裁判噪声=成立且是放大器；DPO 训练=次要（剂量低但效应量瓶颈在上游）；验收统计=成立（单判读不出）；目标定义=**被低估的头号问题**（rubric 看不见目标）。

---

## 四、corrected-v3 最小可执行方案（零训练，先证伪）

**核心假设（要拆开的三岔口）**：当前失败到底是 ①rubric 看不见目标（h 可被字符串手术 Goodhart / 在 0.85 封顶）、②候选池产不出结构性更像人的样本、还是 ③只是测量噪声淹没了真实进展。三个探针各砍一支。

| 探针 | 内容 | 调用量 | 判定 |
|---|---|---|---|
| **E0 测量定标 + mini 终审** | 先从云机拉 96 per-query infer/judge、196 对复判全量、63 stable meta、93 scores、trainer 日志；对 base 与 mini-DPO 的 224 输出各补判 k=2（合计 k=3）→ paired Δh bootstrap CI、correct 翻转 McNemar、痕迹 paired 矩阵（同时定标痕迹计数噪声） | 896 | correct −10 定罪/赦免；mini 终态三选一；产出 σ 的 eval 人群版 |
| **E1 安慰剂字符串手术**（攻 rubric） | 60 条带痕迹 base 输出，纯机械删 explicit_ref/policy_source 标记词（零内容改动），k=3 判 | 180 | 若 Δh ≥ OLS 预测的 60%（约 +0.04）→ h 可被表面手术 Goodhart，**rubric 必须重定义，停止优化 h 均值** |
| **E2 重写天花板探针**（攻 pool ceiling） | 50 条 base think 由强模型做"从问题推导"式受控重写（保事实，过 faithfulness/fact_recall 闸），k=3 判 + 20 条人工盲评 A/B | ~300 | 若重写 k=3 均值 ≥0.80（> 无痕迹基线 0.770）∧ 人工盲评 ≥70% 偏好 → **池上限实锤，v3.1 数据源改受控重写**；若仍卡 0.75/0.85 档 → rubric 表达不了结构人化 |
| **E3 成对判分负对照**（采用前提） | base-vs-base 不同 seed 50 组（位置偏差/虚假果断率应 ≈50:50）+ 同内容长/短改写 30 组（长度偏差） | ~200 | 任一偏差超 55:45 → 成对判加长度归一/位置惩罚或弃用 |

预算：**Kimi ~1,600 次（约 2h @3workers），GPU 0 小时，无训练**。一天内收口。

**探针结果 → 分支**：
- E1 Goodhart 实锤 → 目标重定义分支：要么诚实把验收主指标改为"paired 痕迹消除 + 人工盲评"（痕迹本来就是可优化且原则上可测的东西），要么重写 rubric（锚定样例、把"结构推导"与"痕迹"拆成两个维度）并重锚 0.697 基线。
- E2 GO → **v3.1 重写对 DPO mini**：chosen=受控重写、rejected=原输出（margin 可控、对比度即目标结构、不依赖 8 选 1 撞运气）；300 query；禁 both-correct 近重复对；长度 |ratio−1|≤0.10；chosen 净化正则（`<img`/`https?://`/参考文件）；构对与验收全部 k=3；train ≥260 对 / ga2 / 2-3 epoch / rpo_alpha=1.0；预算 ≈4,500 调用 + 2h GPU。
- E1 否 ∧ E2 否 → 路线诚实终结：RFT merged base（0.697/0.858/0.818）即当前尺子下的最优，交付证据链（噪声模型 + 天花板证明 + 两个证伪探针），叙事是"度量上限"而非"工程失败"。

---

## 五、统计判定框架（落地版）

1. **方差分解**：σ²_total = σ²_between(样本间) + σ²_judge(判内)。实测（pair 文本人群）σ_judge：h=0.117、g=0.048、acc_score=0.074；E0 补 eval 人群版。
2. **k 次判分**：h 用 k=3 取均值（σ→0.067，paired MDE 0.022→0.013）；acc tier 用 k=3 多数票（二值翻转 9%→1.3%，correct 计数 sd 4.49→1.70）；grounded k=1 够用。
3. **验收主读数 = 逐题 paired Δ + bootstrap 95% CI**：保存 (qid, h_base^k3, h_new^k3, tier_base, tier_new)；correct 用 McNemar（数 b/c 不一致格）。禁止用单判均值差下结论。
4. **构对**：margin 门建立在 k=3 均值上（margin≥0.25 的假方向率从 4-20% → <0.6%）；方向复核优先用成对判（E3 通过后），平手不算失败。
5. **反双重浸入纪律**：任何过滤后集合的指标，必须用一次未参与选样的独立判分汇报。
6. **本轮判定**：Δh=−0.009 → 噪声（CI 跨零，z≈−0.8）；Δacc=−0.012 → 噪声边缘；correct −10 → 可疑未定罪（E0 McNemar 定夺）；mini 结论 = "无显著变化，且当前读出系统对 ≤0.03 的变化天然失明"。

---

## 六、GO/NO-GO 标准（v3 各阶段）

**探针阶段**（E1/E2/E3 判定见 §四表格）。**硬停门（写成 raise，不许 WARN 续跑）**：headroom NA 或 <0.05；独立复判方向率 CI 下界 <60%；chosen 复判 h≥0.75 保持率 <80%；paired correct 净翻转 ≤−8。

**v3.1 训练 GO（promote 到 full 的标准）**：
- 效果：k=3 paired Δh bootstrap 95%CI 下界 >0 ∧ 点估计 ≥+0.02
- acc 守护：c+p% 下降 ≤1pt ∧ McNemar 净新增 incorrect ≤4
- grounded 守护：均值 Δg ≥−0.01 ∧ g<0.6 条数增加 ≤3
- 过程：独立判分下 pair 方向 ≥80%；heldout pref-acc n≥40 且 ≥26/40
- 旁证：痕迹 paired 下降 ≥20%（k≥2 多数票口径）∧ verbatim_copy 不升；30 条人工盲评胜率 ≥60%

**开训前置闸（新增）**："设计效应值"预测：用 OLS 痕迹系数 × pair 实际教的变化预测 Δh，**预测值 < 当轮读出 MDE 则不开训**（v2 的 +0.006 vs 0.022 本可拦下这次空跑）。

---

## 七、明确不建议做的事

1. 不推 full DPO/GRPO——mini 未给出任何正向信号，且尺子未修。
2. 不在同一把尺子下扩量重训（53→200 对只会复现本轮：同样的噪声选 pair、同样读不出差异的验收）。
3. 不再降 stable filter/pair 数门槛续命——v2 把 80%→70%→实测 59.2% 还放行是流程事故，v3 一律硬停。
4. 不只调 lr/beta/steps——设计效应 +0.006 与训练剂量无关。
5. 不把 win-rate 当主验收读数（绝对分 0.697 基线体系不可丢），成对判只用于构对方向。
6. 不在选样用的测量上汇报指标（95b 教训）。
7. 不再盲目扩 best-of-8 池判分——真 margin ≈+0.13 已测得，瓶颈不是判分量。
8. 修复测量前，不相信任何 |Δh|≤0.03 的读数（包括"提升"）。

---

## 八、Codex 落地清单（文件级）

**数据回拉**：`scripts/pull_v2_artifacts.sh`——从云机拉 96_corrected_v2_mini_dpo_{infer,judge}.jsonl、95b_corrected_v2_pair_rejudge_all.jsonl（196 对）、95b stable meta（63 对）、93_corrected_v2_rollout_scores.jsonl、trainer 日志/trainer_state.json 到本地审计目录。

**新增脚本**（输出统一 `output/corrected_v3/<run_id>/`，文件名带 `corrected_v3` 前缀；decision json 记输入文件 hash+行数）：
- `pipeline/step100_judge_noise_calibration.py` → `100_corrected_v3_noise_calibration.{jsonl,md}`（E0：k 判、方差分解、痕迹计数重测）
- `pipeline/step101_paired_eval_stats.py` → `101_corrected_v3_mini_paired_readout.md`（paired Δ+bootstrap CI+McNemar+痕迹矩阵；先回溯用于 mini 终审）
- `pipeline/step102_placebo_trace_surgery.py` → `102_corrected_v3_placebo_report.md`（E1）
- `pipeline/step103_rewrite_ceiling_probe.py` → `103_corrected_v3_rewrite_probe.{jsonl,md}` + 人工盲评表（E2）
- `pipeline/step104_pairwise_judge_controls.py` → `104_corrected_v3_pairwise_controls.md`（E3）

**修改**：
- `judge_common.py`：加 `judge_text_k(text, k, aggregate='median')`（独立会话）与 `pairwise_judge(a,b)`（A/B 换位 2 票+加赛）
- `step94_build_dpo_pairs_v2.py`：headroom NA→NO-GO（fail-closed）；BAD_CHOSEN 改正则（`<img`、`https?://`、`参考文件`、answer 区文号罗列结尾）；逐条件淘汰漏斗计数写入 decision json；禁 both-correct 且 think 相似度 ≥0.5 的对；长度 |ratio−1|≤0.10
- `step95_rejudge_pairs_v2.py`：平手不算失败；方向判定接 k=3 或成对判；阈值由噪声模型反推
- `run_corrected_v2_mini.py` → 新建 `run_corrected_v3.py`：所有 NO-GO 一律 raise 硬停（不留 WARN 旁路）；run_id 子目录防同名覆盖
- `dpo_v2.sh`（v3.1 才用）：实现 "<100 对 ga=1" 分支；rpo_alpha 默认 1.0；保存 trainer_state.json

**预算**：探针阶段 Kimi ≈1,600 次、GPU 0；v3.1（若 GO）≈4,500 次 + 2h GPU。
