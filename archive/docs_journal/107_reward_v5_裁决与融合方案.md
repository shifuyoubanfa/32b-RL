# 107 · 奖励函数 v5 技术裁决与融合方案（Fable5 终裁，交 Codex 落地）

> 评审：证据审计 / V1 可行性 / 224 防泄漏 / Kimi 角色与规则稳定性 / DPO-GRPO 设计 / 红队 六 agent 并行实算（合计亲读 100+ 样本、重算 224×3 模型推理、K16 采样池、bootstrap 功效）。
> 所有结论带行级/数字证据。本文给唯一方案，无分叉。

---

## 0. 一句话终裁

**两套方案都不能照搬，必须融合，但融合之前有一个比"选哪套 reward"更致命的前提必须先解决：当前被 RL 优化的指标 ~90% 是假阳（合法税法引用被误判为痕迹），真痕迹 headroom 只有约 22/224 题（9.8%）——任何 reward 设计都推不动一个已近地板的指标。** 所以 v5 的第一动作不是训练，是**修指标 + 修评测 + 一天的零 GPU 证伪实验**；融合方案以 **Codex 的词典序不可补偿约束为骨架**，吸收**用户的规则稳定性审计与分阶段推进**，但 **V1 logprob 降级为辅助监控带（绝不进 reward）、Kimi 降级为离线复核（绝不进硬门/在线 reward）**。

---

## 1. 最致命的实算证据（决定一切的三件事）

1. **被优化的指标几乎是空的**：frozen trace_total=127 里 policy_source=76（60%）。证据审计 agent 人读+正则双判 96 条 policy 命中样本，**0 条含真 RAG 痕迹、97-100% 是合法法条引用**（"按照《印花税法》第四条规定…"），且 `r(policy, answer_score)≈+0.03~+0.05`——**引用法条的回答反而更对，压 policy = 压正确推理，方向是反的**。扣掉 policy 假阳 + 税务事实掩码后的 verbatim（26→14 真），s1 上携带真痕迹的题只剩 **22/224=9.8%**，其中无歧义的 explicit_ref（"参考问答对"）只集中在 **8 题**。**RL 天花板被这 22 题钉死。**

2. **四轮 RL 净变化≈0 是统计噪声，"指标迁移"叙事被逐题证伪**：clean_rate 轨迹 0.241→0.237→0.245，而 80% 功效的 MDE=0.116（差一个数量级）；McNemar net 全部不显著（s2 +1 p=0.85、s3 −5 p=0.30）；红队逐题比对：s1→s2 丢 explicit 的 4 题与涨 verbatim 的 7 题**零重叠**——不是"拿 explicit 换 verbatim"的耦合交易，是两个独立随机游走。DPO/GRPO 三阶段 answer/fact 的 95%CI 全部过零。

3. **clean_rate=0.24 是误导性看板**：它是 DPO-chosen 资格率（burden==0 ∧ masked≤0.30 ∧ fact≥0.75 ∧ answer≥0.55），不是无痕率。真实无痕率 frozen 口径 0.74、v4 算子 0.89。"76% 输出仍有 trace"是错的，真痕迹率约 10%。

**但 headroom 在采样分布里真实存在**（红队 K16 池实算）：57% 的 query 同时有干净和脏候选、池内平均干净率 0.221——模型本就会干净，是 greedy 解码坐在总体均值上 + 224-greedy-eval + frozen 计数器把信号淹没了。**所以 RL 不是没救，是被错的指标和错的评测淹没了。**

---

## 2. 逐问裁决（A-E）

### A. V1 自一致基线：作 reward 不成立，作辅助守门有条件成立，必须降级
- **V1 logprob 代表 V1 风格/自我偏好密度，不是正确性**：自信的错答案 logprob 同样高（self-preference 偏置）。**否决作 reward 项。**
- **条件混淆已实测**（V1 agent）：去 RAG 的目的就是改 think，think 改了→`logP(answer|query,RAG,think)` 的条件就漂移；12.5% 的题 think 大改而 answer 几乎不变，logprob 差分无法区分"答案变了"还是"think 变了"。
- **σ-band 是噪声地板**：per-query σ_q≈0.121，比真实有害退化信号（0.011）宽约 11 倍，μ−kσ 带形同虚设。
- **裁决**：V1 **不做答案质量 reward、不做硬门**。降级为：(a) answer-only、固定 conditioning（不含 think）、长度归一的 logprob，作 per-query 分布漂移的**弱召回监控带**（k=2），仅在 DPO 筛选阶段与 Codex A/B **AND 串联**做拒绝（任一判退化即拒），不补偿 think 分；(b) 在线训练对 answer span 加轻 KL 护栏 β_ans。**真正的答案守门用确定性 fact-floor**（见 D）。V1 自评打分（方案 0A）= 同款 LLM-judge，balanced_acc 0.633，**否决**。

### B. 224 防泄漏：224 已是被污染的开发集，必须从 2015 切新 sealed
- **合法（不污染）**：在任意集上跑 V1/模型收集其自身分布统计（μ/σ、logprob 直方图）——这是测量 V1 本身。
- **非法（污染）**：任何用"让验收集指标更好看"为目标去选阈值/选提示/选 checkpoint 的动作（125a 已发生且失败：把 Kimi 门拟合到评测相邻数据，balanced_acc 0.633）。
- **裁决三集**（query sha256 两两不相交，落 `split_manifest.json`）：
  - **TRAIN_POOL = 2015 − 500 = 1515**：SFT/RFT/DPO/GRPO 全部采样只在此。
  - **CALIB = 从 2015 切 200**：只用于调规则阈值、定 NI margin、Kimi 校准、规则子项稳定性审计；冻结后才碰 sealed。
  - **SEALED_FINAL = 从 2015 切 300（全新）**：永不进训练/采样/校准，全程只判一次，整个 v5 周期 sealed 候选 ≤3 个 ckpt（Holm 校正控 FWER）。
  - **DEV = 沿用现有 224**：可反复看、定方向、调超参、看趋势；但**其数字一律标注 development-only，永不写入"RL 有效性结论"**（保留与 v1-v4 历史可比性）。
- 不切 224 本身（切了也不能去污染）；切 300 是"答案非劣侧充分 + 训练池可承受 + derag 侧用连续指标可表达"的最小可行点。

### C. Kimi 角色：相对/元判断可用，绝对/在线/唯一校准尺不可用
| 角色 | 裁决 | 条件 |
|---|---|---|
| 判 B 相对 A 退化 | 有条件 | 必须成对（给 A 和 B 问 B 是否丢结论/税种/税率/金额/限定），k=3 多数票，平票→UNKNOWN→mask；绝不绝对二值判 |
| 判规则扣分是否合理（元判断） | 最适合 | k=3，是规则审计的输入之一——但**不作唯一校准尺**（见下） |
| 改写 think 不改 answer | 适合 | 离线 SFT 数据生成（历史 0.23→0.68 唯一大杠杆），必须过 answer-lock + 去公文腔后处理（phrase_gate 已红：'综上' 0.289/千字 vs 阈 0.15） |
| 在线唯一 reward / 硬门 | **否决** | balanced_acc 0.633、bad_pass 0.433，大 K 下误放 1−(1−0.433)¹⁶≈1.0 |
- **关键修正（红队）**：规则子项稳定性审计**不能只用 Kimi 认可率**（用 0.633 的尺校准另一把尺=双重不可靠）。改为：**bootstrap 符号一致性为主（2015 池 1000 次重采样符号一致率 ≥0.95）+ Kimi 元判断 k=3 为辅证**，两者都达标的子项才进 GRPO penalty。

### D. DPO：answer-lock + 单边长度门 + 确定性 margin（融合两套）
- **π_ref = RFT merged**（被优化策略的同分布起点；V1 与待训策略分布差太大会使 KL 失真）。**V1 只做筛选门 + answer span KL 护栏，不做 π_ref、不进 reward。**
- **"V1 作 answer-ref、RFT 作 think-ref" 的实现**：answer 侧由 V1 logprob band（弱召回）+ 确定性 fact-floor（主）守；think 侧 RFT 作下界锚（think_len∈[120,1800]，防雕成空洞）。
- **pair 三门（全确定性，Kimi 仅离线复核）**：
  - 门1 answer 安全：chosen 与 rejected 各自 `fact_recall(answer,gold)≥0.80 ∧ introduced_nums=∅ ∧ grounding_floor_ok ∧ 极性词集合无翻转 ∧ V1_logprob ≥ μ_q−1.0σ_q`。
  - 门2 answer 等价（堵混杂——实测当前 chosen 比 rejected answer +0.121/fact +0.111，答案质量与风格被绑死）：`|Δfact_recall|≤0.05 ∧ answer_score_ch ≥ answer_score_rj−0.05 ∧ 关键槽位 facts() 签名相等`。
  - 门3 trace margin（纯风格轴）：`chosen.burden=0 ∧ chosen.masked_copy≤0.30 ∧ (Δreal_burden≥2 ∨ (Δreal_burden≥1 ∧ Δmasked_copy≥0.12))`。**real_burden 用剔除 policy_source 的 real_trace（见 §3）。**
- **answer-lock（堵 138 的 answer↓根因）**：chosen 与 rejected **共享同一 answer 文本**（取 chosen 的 answer 拼到两条），DPO logratio 100% 来自 think；做不到则 answer span loss_mask=0。think span 权重 1.0、answer span 权重 0.1。
- **长度门改单边**：仅当 `len(rj_think)<0.8·len(ch_think)` 才丢弃（堵"chosen 写成空 stub"），允许 rejected 更长（复制天然更长）。实测产量 r0.3=183→单边=216 对，越过 min_pairs=160。
- **诚实记账**：ΔDPO = DPO − RFT（π_ref），带 sealed paired bootstrap CI。

### E. GRPO：词典序硬门 + 最差维度聚合 + 剔 policy_source + 从 SFT 直起
- **reward（词典序、不可补偿）**：
  ```
  R = −1                              若 L0 失败(format/extreme_degen/introduced_nums/explicit_ref>0/ref_enum>0/raw_copy>0.55/img_trace)
  R = −1                              若 answer-DEGRADE(fact_recall<0.80 或 answer_score<baseline−0.08 或 极性翻转 或 forbidden_claim)
  R = mask(advantage=0)               若 UNKNOWN(0.75≤fact_recall<0.80 边界)
  R = 0.3 + 0.7·S_derag               若 NO_DEGRADE
  ```
- **S_derag 用最差维度聚合（不是加和，防迁移）**：`S_derag = 1 − max(norm_explicit_ref, norm_ref_enum, norm_masked_copy_excess)`，`norm_masked_copy_excess = clip((masked_copy−0.25)/0.30, 0, 1)`。**加和允许 explicit↓换 verbatim↑（已被实证钻空），最差维度聚合堵死这条路。policy_source 完全移出 S_derag，仅作监控。**
- **组内门**：每题 K≥8，组内 NO_DEGRADE<2 → 整组 mask；advantage z-score 只用 NO_DEGRADE 子集（−1 不进方差，避免正样本 advantage 塌缩）。
- **起点与分账（堵 138 的"GRPO 只修 DPO 伤害"）**：GRPO **从 SFT 直起，π_ref=SFT，与 DPO 并列两条独立支线**，各自相对 SFT 评测，最后取 sealed 上同时满足三门的那条。ΔGRPO = GRPO − SFT（不是 GRPO − DPO）。
- 超参：lr 5e-7、β(KL) 0.05（偏紧防漂移）、K 8-16、每 50 step 在 dev 跑指标迁移审计（explicit↓但 masked↑则该 step masked penalty×1.5 并回滚）。

---

## 3. v5 必须先做的两件事（在任何训练之前）

**(1) 修指标**——把 RL 优化目标从 frozen trace_total 切到：
```
real_trace_units = explicit_ref + ref_enumeration + verbatim_real + label_line
verbatim_real = 1[ masked_copy(mask_tax_facts(think), mask_tax_facts(refs)) ≥ 0.40 ]
```
policy_source / standalone_citation **退出一切 reward 与 DPO 筛选项，仅作监控**。主评测指标改为**连续 per-item burden 均值 + 配对 bootstrap**（MDE 比二值 clean_rate 敏感 3-5×），不再用 trace_total/clean_rate 二值做主门。

**(2) 修评测 + 一天证伪实验（0 GPU，红队主张，采纳）**——烧 GPU 前先确认 headroom 与可测性：
- 实验1 真痕迹上界：`real_trace 携带题数/224`，若 <8% 判定 ceiling 成立、trace 不是可优化目标 → 不训练，先重做冷启动 SFT 搬中心（见 §4）。
- 实验2 采样头室：K16 池 `frac(query 同时有 clean&dirty 候选)`，实测 0.57；≥0.40 则 DPO 有真实 within-query 对比可学。
- 实验3 评测敏感度：在 s1/s2/s3 上回算 burden 均值 + 配对 t，看历史四轮是否有被二值门吃掉的连续信号。
- 三个实验全在本地/dev，约 1 天、0 GPU。**通过才进训练，不通过先修数据。**

---

## 4. 最终融合方案（阶段图，定死）

| 阶段 | 输入 | 模型/起点 | 用 224? | 用 2015? | 调 Kimi? | 调 V1? | 输出/门 | 失败回退 |
|---|---|---|---|---|---|---|---|---|
| **S0 修指标+证伪** | 128/129/132 推理、K16 池 | — | DEV 算敏感度 | — | 否 | 否 | real_trace 指标 + burden 评测器 + 三实验报告 | ceiling<8% → 转 §4-S1' 重做冷启动不做 RL |
| **S0b 校准** | CALIB 200 | V1×6 采样 | 否 | CALIB | Kimi 元判断 k=3 校准三态裁判+规则审计 | 测 V1 μ_q/σ_q、定 k/margin | 冻结阈值+sha256；规则子项 bootstrap 一致率≥0.95 才入 GRPO | Kimi balanced_acc<0.85 → 裁判降级离线复核 |
| **S1 冷启动 SFT（搬中心，最大杠杆）** | TRAIN_POOL 1515 带痕迹题 Kimi 改写（answer-lock+去公文腔） | RFT merged + fresh LoRA | DEV 验收 | TRAIN_POOL | 改写 think | answer fact-floor | think real_trace 降 + answer fact 不降 + phrase_gate 不红 | <400 改写 → 回改写 prompt |
| **S2 DPO（支线A）** | S1 自采样 K16 | S1_merged，π_ref=S1 | DEV 验收 | TRAIN_POOL | 离线 pair 灰区双序复核 | logprob band 筛选门 | §2-D 三门 pair ≥160；ΔDPO=DPO−RFT | answer 非劣不过→回退 S1 |
| **S3 GRPO（支线B，与S2并列）** | S1 自采样 K8-16 | S1_merged，π_ref=S1 | DEV 验收 | TRAIN_POOL | 离线边界复核 | fact-floor 硬门 | §2-E 词典序 reward；ΔGRPO=GRPO−S1 | 指标迁移审计触发→回滚 step |
| **S4 sealed 终裁** | SEALED 300 | S2/S3/S1 各 ckpt | 否 | SEALED | 否（冻结门） | 冻结 μ_q/σ_q | 三门乘法：ANSWER_NI ∧ DERAG_GAIN ∧ NO_MIGRATION，paired bootstrap CI | 全失败→交付 RFT base 0.697 + 诚实报告 |

**最终证明"think 去 RAG 提升 ∧ 答案不降"的口径（sealed 300，一次性，预注册）**：
- ANSWER_NI（不可补偿硬门）：`LCB95(Δfact_recall) ≥ −0.03` 且关键槽位等价率非劣；
- DERAG_GAIN：`LCB95(Δreal_trace_burden 改善) ≥ δ_min`（δ_min 在 CALIB 估的 sd 反推，≥0.5·sd）；
- NO_MIGRATION：explicit/verbatim/enum 各子项 `LCB95(Δ) ≥ −0.5·sd`（防 explicit↓verbatim↑）；
- NO_NEW_TIC：phrase_gate 不变红。
- 四门全真才算"提升且不退化"；任一不过判 FAIL 或 inconclusive，绝不只报点估计。

---

## 5. 终裁结论（用户六问）

1. **是否采用 Codex 方案**：采用其**骨架**（词典序不可补偿约束、A/B 三态退化门、UNKNOWN-mask、防迁移）。**修正**：锚点 A 不用 answer_score（仅相似度，非正确性），用 gold critical-slot 等价 + 确定性 fact-floor + Kimi A/B 复核三源合一；A/B 的语义判断 k=3 多数票，不依赖单次 Kimi。
2. **是否采用用户方案**：采用其**分阶段推进**与**规则稳定性审计思想**。**否决** V1 logprob/自评作答案质量 reward 或硬门（降级为弱召回监控带）；规则审计改用 bootstrap 一致性为主、Kimi 元判断为辅。
3. **是否融合**：是。
4. **完整方案**：§4 阶段图。
5. **必须废弃的旧逻辑**：① frozen trace_total 作主指标；② policy_source/standalone_citation 进 reward 或 DPO 筛选；③ answer_score（SequenceMatcher 相似度）当正确性证明；④ clean_rate=0.24 当去痕看板；⑤ V1 logprob/自评作 reward 或硬门；⑥ Kimi 作硬门/在线 reward/唯一规则校准尺；⑦ 二值全局 clean_rate/McNemar 作主验收（功效不足）；⑧ DPO 用 answer_score≥0.55 主筛 + 对称长度门；⑨ GRPO 从 DPO 串联起跑（改并列、从 SFT 起）；⑩ S_derag 各子项加和（改最差维度聚合）。
6. **Codex 下一步代码顺序**：
   1. `reward_v3.py`：新增 `real_trace_units`（剔 policy_source）、`S_derag` 最差维度聚合、词典序 `derag_reward_v5`；policy/citation 降为监控字段。
   2. 新 `pipeline/step140_metric_audit.py`：§3 三个证伪实验 + burden 连续评测器（0 GPU）。
   3. 新 `pipeline/step141_split_manifest.py`：从 2015 切 TRAIN_POOL/CALIB/SEALED，落 sha256，全脚本加 `assert qid not in sealed_ids`。
   4. `step125_*`（改写门）：answer-lock（共享 answer）+ 去公文腔后处理 + phrase_gate 作硬监控。
   5. `step127/step124_build_dpo_pairs`：换 §2-D 三门 + 单边长度门 + 确定性 margin（用 real_burden）。
   6. `dpo_v2.sh`：answer span loss_mask、think/answer 权重 1.0/0.1、π_ref=S1。
   7. `grpo_on_model.sh` + reward 插件：词典序 reward、组内门、从 S1 起、指标迁移审计 hook、ΔGRPO=GRPO−S1。
   8. 新 `step142_sealed_eval.py`：§4 四门乘法验收 + paired bootstrap CI + Holm 校正 + 冻结清单 sha256。
   - **顺序铁律**：1-3 先做（修指标+证伪+切集），三实验通过才做 4-7（训练），最后 8（一次性 sealed）。

---

## 6. 给组长的诚实话术

"四轮 RL 净变化≈0 的真相不是 reward 不够好，是我们一直在优化一个 ~90% 是假阳的指标——所谓 RAG 痕迹里 60% 是合法税法引用（'按照《印花税法》第四条'），而且引用法条的回答反而更准。扣掉假阳后，真痕迹只在约 10% 的题上，这就是天花板。好消息是：模型在采样时本就有 57% 的题能产出更干净的版本，headroom 在分布里真实存在，只是被错的指标和不敏感的评测（224 上二值 clean_rate 的最小可检出量是 0.116，比信号大一个数量级）淹没了。v5 先用一天零 GPU 修指标、修评测、证伪 headroom，再按 Codex 的'答案不可退化'骨架 + 我们验证过的冷启动 SFT 大杠杆重做：先把 think 的中心搬干净，再用 DPO/GRPO 在确定性 margin 上收尾，每阶段相对自己起点在新切的 sealed 集上带 CI 验收。预期是把那约 22 道真带痕迹的题改干净、同时证明答案一道不退化——而不是再追一个测不出的全局数字。"
