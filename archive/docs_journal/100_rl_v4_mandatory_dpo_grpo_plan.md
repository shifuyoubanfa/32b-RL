# 100 · RL 必走通约束下的 derag_v4 全链路方案（注入→DPO→GRPO，五角色评审定稿）

> 约束：RL 路线不可推翻，必含 DPO、尽量含 GRPO、有可验收 RL 增益；SFT/重写只作前置；目标固定 trace_free+grounded+accuracy。
> 评审：目标裁判 / DPO 数据 / GRPO 奖励 / 训练工程 / 风险验收 五角色，全员服从约束完成设计。
> 命名空间：`derag_v4`，云机 output=/home/nvme01/zhjg/output/derag_v4/<run_id>/，ckpts=/home/nvme01/zhjg/ckpts/derag_v4/{s1_sft,s2_dpo,s3_grpo}/<run_id>/，文档编号 120-139。

---

## 1. 一句话结论

**按 14B 已验证的顺序重建 32B 链路：先用受控重写做"分布注入"SFT-lite（RL 前置，不计 RL 账），再在注入后模型的 on-policy 采样池上构对跑 DPO，最后从 DPO ckpt 用本地确定性 reward 跑 GRPO——每阶段相对自己起点 ckpt 的 paired Δ 诚实记账，RL 增益 = ΔDPO + ΔGRPO。**

三轮失败证明的不是"RL 不行"，是顺序错了：在没有正样本的分布上先跑了 RL。14B 是阳性对照（注入后 DPO +0.056、GRPO +0.01）。

## 2. 对 99 号文件的纠偏

**撤回三条**：
- "放弃自采样 RL 主线"→ 过度外推。三连零证明的是**未注入分布上**池空（best-of-8 0.701≈base 0.697）；注入后的 on-policy 池是否有正样本是**可测的**（G1-1 池密度门），不是信仰问题。
- "GRPO 永久出局"→ 依据失效。旧依据针对"结构性像人"不可本地计算；新目标 4 类痕迹中 explicit_ref/ref_enumeration/policy_source 可正则、verbatim≈copy_ratio（reward.py 现成），110 号实算 134 处痕迹约 8 成本地可度量；在线 reward 全本地化后"Kimi 吞吐 19 次/min"的否决也不再适用。
- "SFT 自蒸馏作主交付"→ 降级为 Stage1 注入前置，增益记蒸馏账，永不计入 RL。

**保留全部工程裁决**：旧池不可用、旧 humanness 永久退役、DERAG 新口径、judge 三 bug 必修、pair 确定性闸先行+k≥2、answer byte-equality、照抄泵禁令（删一切与 gold 字符相似度正项）、门 raise 硬停、224 评测 query 禁入训、95b 反双重浸入、成对判分弃硬门。

**修正一处**："k=3 temp0 伪重复"表述不准——temp0 下 r12=0.634<1 说明 API 本就非确定；真问题是重复间系统相关，用三仪器去相关协议（temp0 标准序 / temp0.3 / 字段序变体）解决，历史读数不作废。

## 3. 新技术路线总览（10 行）

```
Stage0  修裁判（4 bug + judge prompt v3 证据先行 + 政策引用判别 + 再校准门）＋ reward_v3 本地化（修文号正则毒点）＋ 预飞行单测
Stage1  注入 SFT-lite：greedy 重推 2015 题 → 确定性预筛带痕迹题(~780) → Kimi 受控重写+门控(≥400条) + replay(~200) → LoRA 88步 → merge s1
        ↳ G1-1 池正样本密度门（RL 资格总闸）：注入后 200 带痕迹题×K8 本地判，≥50% 题有干净候选 ∧ ≥40% 有对比对
Stage2  on-policy DPO：s1 模型 K=8@temp0.9 采样 ~1100 题 → 确定性主导构对(chosen=自产干净/rejected=自产带痕迹,margin 确定性) ≥300-800 对
        → rpo_alpha=1.0、beta0.1、lr5e-6、≥38-76 步 → heldout pref-acc n≥40 ≥26/40 选 ckpt → merge s2
Stage3  GRPO：从 s2，πref=s2，本地确定性 reward（fact_recall 闸×数字守恒闸×痕迹罚项），lr1e-6、beta0.04、120步、
        max_completion 1792、每 25 步 60 题离线 Kimi 探针防 hacking（发散 raise 回滚）
验收    每阶段相对自己起点 paired k=3 Δ + 确定性痕迹计数 McNemar 双轨；ΔRL=ΔS2+ΔS3；望远镜核账 |端到端−Σ阶段|≤0.01
```

## 4. 方案 A（稳妥 RL 路线）

- **数据来源**：重推 greedy（部署解码 max_new=1536，不用 temp0.9 旧 rollout）；60_dpo_rollout 仅作零成本测量（标注 1024 截断偏置），训练禁用；允许受控重写（仅 Stage1 注入与 ≤30% 锚定对）；Stage2 候选全部新采样自注入后模型（on-policy）。
- **链路**：Stage0→1→2→3 如上。注入是给 RL 制造学习信号的前置，不是终点。
- **预期与记账**：总余量 ≈+0.072（87/224 带痕迹题 × 0.185 分差）。注入吃 +0.03~0.05（蒸馏账），DPO 余量 +0.015~0.035，GRPO +0.005~0.015。**不能拿 14B 的 +0.056 直接外推 32B DPO——起点 0.775 余量小一个量级，预期管理写进终报。**

## 5. 方案 B（更纯 RL 路线，Kimi 不当 teacher）

正样本不靠 Kimi 重写，靠三路自产挖掘：
1. **prompt 引导采样（context distillation 式）**：rollout 时系统提示追加明令（禁止"参考资料/问题N/文号清单"字眼、给 1-2 条干净风格示例），温度 1.0-1.1、K=16；**训练 messages 用部署 prompt**，chosen 文本来自引导采样——仍是模型自己的分布在不同条件下的样本。
2. **机械手术增广**：模型自产文本的规则编辑版作 chosen 候选，**编辑帽：删除 ≤15% 字符、插入仅限 8 词连接词闭集 ≤3 处、missing_nums=∅**（≥85% 逐字保留才可声称"还是模型的文本"；E2 已证此类编辑 +0.038 且守护不崩）。
3. **高温大 K 暴力挖掘**：对带痕迹题 K=16-32。
- **"模型自产"的可证伪定义**：`chosen_nll_ratio` 门——chosen 在部署上下文下的 NLL/token ≤1.15× 模型 greedy NLL，超出即判"离流形文本"剔除。
- **风险与最低验证**：产率未知是主风险 → pilot 120 题试产，pair 产率 ≥30% 放量、15-30% 只许 K 8→16 一次、<15% 转方案 A。成本：rollout GPU ~1-2h + 判分 ~1.5k 调用，1 天内可裁决。
- 定位：A 的消融对照（全量版第 7 天可跑），证明"RL 增益不依赖外部 teacher 注入"时叙事价值最大。

## 6. DPO 具体方案

| 项 | 值 | 理由 |
|---|---|---|
| 起点/πref | s1_merged + fresh LoRA / s1_merged（disable adapter） | dpo_v2.sh 已验证的无歧义语义 |
| 采样池 | 注入后模型，痕迹倾向题 ~780 + 干净题 300，K=8 temp0.9 top_p0.95 max_new1536（TP=8 ~1.5h） | on-policy；池子构成保证组内对比 |
| chosen | 自产：trace_hits=0 ∧ copy_ratio≤0.30 ∧ fact_recall≥0.7 ∧ introduced/missing_nums=∅ ∧ Kimi k=2 min(grounded)≥0.8 ∧ acc 多数 correct | 确定性闸先行，Kimi 只兜灰区 |
| rejected | 同题自产：trace_hits≥2 ∨ copy_ratio≥0.40；acc tier ≤ chosen 且 fact_recall 差 ≤0.05；**≥90% 须有确定性痕迹证据** | margin 不押裁判噪声（v2 的 −68% 收缩教训）；fact_recall 差约束堵 accuracy 混淆 |
| margin | **确定性**：trace_hits 差 ≥2 ∨ copy_ratio 差 ≥0.2；Kimi 只确认不创造 margin | 98 号教训制度化 |
| 长度 | pair 级 \|len_c/len_r−1\|≤0.3 ∧ **数据集级均长比 ∈[0.9,1.1]** | 裁判长文偏置 76.7%，集级平衡防系统性长度学习 |
| 配比 | 自产对 ≥70%，锚定对+重写对 ≤30%；不足 400 对用手术增广补 | 保持 on-policy 主体 |
| 超参 | beta=0.1、lr=5e-6、**rpo_alpha=1.0（swift 不支持即 raise，不许静默 skip）**、epoch=2、warmup 0.1 | NLL 锚防 chosen 概率同跌 |
| 步数 | P=600→76 步 / P=300→38 步；P<320 自动 ga=1；**P<160 raise NO-GO** | v2 的 12 步不可判读 |
| 开训前置闸 | 设计效应（OLS 痕迹系数×pair 痕迹差）≥ MDE 0.013；headroom NA=NO-GO | v2 的 0.006<0.022 教训制度化 |
| 训中监控 | rewards/accuracies@step20≥0.55（v1 的 0.23-0.41=不可学）；rewards/chosen<−0.3 连续 10 步停；每 10 步 step94b heldout pref-acc，连续 2 降取峰值 | |
| 选模 | heldout 偏好集 n≥40（query 级不相交、分层抽样、fresh k=2 复验、翻转率>25% 整池可疑 raise），pref-acc ≥26/40（二项 p=0.04） | 98 号欠账 step94b 本次必落地 |
| 验收 | 相对 s1：paired k=3 Δtrace_free ≥+0.015 ∧ CI 下界>0；或本地痕迹题净转化 ≥+12/87 ∧ McNemar p<0.05。守护：Δacc≥−0.01 ∧ c+p% 降≤1.5pt ∧ Δgrounded CI 下界≥−0.02 ∧ len p50≤+20% ∧ 5-gram 重复≤base+2% | 双轨：裁判分数 + 裁判噪声免疫的确定性计数 |

## 7. GRPO 具体方案

- **从 DPO 后模型开始**：s2_merged + fresh LoRA，**πref=s2_merged**（锚 RFT base 会把策略拽回带痕迹分布；锚 s1 会惩罚 DPO 位移）。
- **开跑资格门**：S2 相对 RFT base 累计 Δtrace_free ≥+0.05 且守护过——GRPO 是 on-policy 精修不是分布注入器，前置不成立就顺延修注入。
- **reward v4（全本地确定性，零在线 Kimi）**：
```
R = −1                            若 format 不合法 或 G_degen 触发（断句/碎句/8-gram 自重复≥3/distinct-2<base p5）
R = 0.1×fact_recall               若 fact_recall(answer,gold)<0.8 或极性矛盾（免征↔应缴 等对冲表）   ← 饱和闸，零渐变，根除照抄泵
R ×= 0.2~0.3                      若数字守恒失败：N(think∪answer) ⊄ N(refs∪gold∪query)∪W∪D
                                  W={0..12,100,年份±1}，D=refs∪gold 数字一步四则派生(容差1%)；或 copy_ratio>0.55
                                  反去依据地板：|facts(gold)|≥1 时 facts(think)∩facts(refs∪gold)≥1，否则闸失败  ← 防"删光数字躲守恒"
否则 R = 0.3 + 0.7 × T × (1 − 0.5×l_len)
T = 1/(1 + 0.34·hits_explicit + 1.0·hits_enum + 0.36·policy_pen + 3.0·max(0,copy_ratio−0.30) + 1.5·max(0,enum_density−0.40))
l_len = 带宽外渐罚，ratio=len(think)/L0（L0=该题在 s2 上的 greedy 长度，随数据列下发）
```
  系数与 98 号 OLS 裁判敏感度对齐（ref_enum −0.198 归一 1.0、explicit −0.067→0.34、policy −0.072→0.36、verbatim −0.135→copy 超阈 ×3.0）。
- **政策引用毒点修复（进 reward 前必过对齐门）**：删 reward.py:32 裸 `〔\d+〕号` 正则；"根据(以上|上述)"排除自指。机械罗列才计罚：(a) 清单式 ≥3 文号/《》连排；(b) "政策依据："标签领起；(c) 裸指针句（含文号无任何内容词）≥2 处。**合法行内引用（单文号嵌在 ≥15 字含"规定/按照/明确"的推理句内）零罚**。对齐门：61 痕迹题+50 干净题上 precision≥0.8 ∧ recall≥0.6，20 条含合法引用的干净 think 误报 ≤2——不过线禁止进 reward。
- **"防 sed 式删词"裁决**：机械删词是新目标的**合法解**（E2 +0.038 实证），GRPO 内化它优于 serving 正则（泛化+流式安全）。只防四种副作用：断句→G_degen；丢依据→守恒地板+fact_recall；缩短→L0 带宽 + **grpo.sh max_completion 1024→1792**（v1 隐藏的截断缩短压力，必改）；口癖→探针层语料监控（任一 8-gram 出现于 >10% rollouts 即停）。
- **超参**：K=8、temp1.0、lr=1e-6（v1 的 5e-7→KL 1e-4=没动）、beta=0.04、120 步预算+探针早停、save_steps=25、save_total_limit=8；colocate 显存五连关原样沿用（TP=8/util0.5/maxlen6144/move_model_batches16/sleep+offload/zero3）。KL 行动表：目标带 [1e-3,1e-2]；<5e-4×20 步→lr×2；>3e-2×5 步→SIGTERM 回滚 beta×2 重启；>0.1 单步硬停。
- **防 Goodhart 探针（训练内 raise，不是事后看日志）**：每 25 步 60 题（30 带痕迹/30 干净，训练池内、与 224 不相交）消费 --log_completions 落盘 rollout：本地计数器逐步算 + Kimi derag k=2 异步判（~6.5min 不阻塞训练）。任一触发写 HALT 停训回滚：本地 T 两窗连涨 ≥0.05 而探针 trace_free Δ≤0（代理-裁判脱钩）；探针 grounded 降>0.03；correct 减>3/60；verbatim 计数 vs step0 +20%；copy_ratio↑伴 trace_hits↓；长度漂出带；distinct-2 降>10%。正则覆盖外痕迹兜底：探针上"裁判报痕迹 ∧ 本地 hits=0"比率升>10pt → 收割该批文本扩 _TRACE_RE，从上一好 ckpt 重启。
- **在线 Kimi reward：否决**（512 判/优化步 ≈27min，比 rollout 慢 50-100×，k=1 噪声吞掉组内差）。**蒸馏 reward model：现在不做**，仅当首轮 GRPO 读出正增益 ∧ 本地/裁判脱钩成为主瓶颈 ∧ 目标升级到正则不可见项，三条同时成立才立项。
- **验收（主仪器换轨——关键设计）**：GRPO 预期 +0.005~0.015 **低于裁判 MDE 0.013**，用 Kimi 判分验收注定假死或假过。主仪器=**确定性痕迹计数 McNemar**（零裁判噪声）：相对 s2 痕迹题净转化 ≥+8（exact p<0.05）∧ 痕迹总计数降 ≥15%；Kimi 降为不回退守门：Δtrace_free CI 下界 ≥−0.005（非劣）∧ McNemar acc 中性 ∧ Δgrounded CI 下界 ≥−0.02。反向闸：确定性计数↓而 Kimi trace_free 同步↓>0.02 = reward hacking，硬停回滚。

## 8. 代码修改清单

| 文件 | 改动 |
|---|---|
| `pipeline/judge_common.py` | ①aggregate 真多数票（round(mean) 废除，平票取保守档）；②缺字段=raise 重试 ≤2 次，3 次标 judge_invalid 剔除（绝不写 0.0）；③reference 截断 3000→6000、think→6000+truncated 标记；④judge prompt v3：证据先行（trace_spans/unsupported_facts/think_answer_conflict 先于分数）+0.5 中段锚点+反印象分条款（给 1.0 须 unsupported_facts=∅）+政策引用判别条款+长度免疫条款+6 条版本化 few-shot（judge_v3.0，跨版本读数禁比）；⑤三仪器去相关 k=3 |
| `pipeline/reward.py` → 新 `pipeline/reward_v3.py` | ①删 answer_drift 的 SequenceMatcher sim 正项（照抄泵）；②修文号正则毒点（裸 `〔\d+〕号` 删除、自指排除、机械罗列三形态正则 policy_pen）；③enum_density、G_degen、L0 带宽、数字守恒+反去依据地板；④reward v4 总公式（§7） |
| `swift/grpo_on_model.sh` | max_completion 1024→**1792**；GRPO_SAVE_STEPS env（默认 25）替换 save_steps=$STEPS；save_total_limit 2→8；--log_completions true；--reward_funcs derag_v4 |
| `swift/dpo_v2.sh` | rpo_alpha 默认 1.0 且不支持即 raise（删静默 skip 分支）；P<320→ga=1 自动分支；env 化 warmup/seed/save_total_limit/LoRA 超参/pdbs；训完归档 trainer_state.json |
| 新增 `pipeline/step94b_heldout_pref_acc.py` | 98 号欠账：n≥40 隐式偏好准确率（β[(logπ−logπref)(chosen)−(rejected)]），26/40 门，2×A800 bf16 |
| 新增 `pipeline/step120_v4_precheck.py` | 所有入口强制：输出目录非空 raise；路径命中历史 ckpt raise；pool∩eval224=∅（原文+规整化双口径）；MANIFEST.json（输入 sha256+行数）；磁盘≥200GB；DPO chosen 全量重过确定性闸 |
| 新增 `pipeline/step121_pool_density_probe.py` | G1-1 池正样本密度门（200 题×K8 本地判，零 Kimi 调用） |
| 新增 `pipeline/step122_reward_preflight.py` | reward 预飞行：spearman(本地T, 110 号 k=3 trace_free)≥0.5（零新调用）；E2 手术文本 reward 必须更高且过 G_degen；10 条手造退化样本 10/10 被闸 |
| 新增 `pipeline/step123_grpo_probe_watcher.py` | 训练内探针+HALT 文件机制（§7 发散判据） |
| 复用改造 | step94（读写/格式/漏斗骨架留，门表全换+headroom fail-closed）、step95b（漏斗报告留，阈值接新噪声模型、三 run_id 分权） |

## 9. 实验运行清单（按今天开工粒度）

**D0（半天，CPU 为主）**：judge 三 bug 修复 + judge v3 + 再校准门（50 题 k=5：trace_sd≤0.07 ∧ 仪器 r≥0.5 ∧ surgery 可读性 CI>0 不退化，~430 调用）｜grounded 扰动测试（40 题×3 变体 k=3，~480 调用；P1 检出≥80%/P2≥70%/P3≥60%，不过线 grounded 降级监控、硬门交确定性守恒）｜reward_v3 实现+预飞行单测（零新调用）｜政策正则对齐门（~130 题判分可复用 110 号现成数据）。
**D1**：greedy 重推 2015（TP=8 1-2h）→确定性预筛→重写 ~780 题+k2 门控（~5.5k 调用过夜）；并行 step94b/120/121 落地。
**D2**：Stage1 SFT 88 步（1.5h）→merge→**G1-1 池密度门**（GPU 1h 零调用）+224 k=3 验收（672 调用）→K=8 采样（1.5h）→构对+灰区判（~800 调用）→DPO 38-76 步（1.5h）→heldout 选 ckpt→merge→S2 验收（672 调用）。
**D3**：GRPO smoke 2 步（组内方差门：≥40% 组 reward std>0）→120 步（~5h，探针随跑）→best ckpt 224 k=3 终审+三阶段记账报告。
**总预算**：3 天名义关键路径，GPU ≈14-16h，Kimi ≈10-13k 调用；含补救缓冲 5-7 日历日硬帽。

## 10. GO/NO-GO 门槛总表（全部 raise 硬停，无 WARN，无降门续跑）

| 门 | 判据 | 失败处置 |
|---|---|---|
| G0-1 裁判再校准 | trace_sd≤0.07 ∧ 仪器间 r≥0.5 ∧ salvage<2% ∧ surgery 可读性不退化 | 修到过，期间一切读数无效 |
| G0-2 grounded 扰动 | P1≥80%/P2≥70%/P3≥60%，假报警≤5% | 不过线→grounded 判分降级监控，硬门=确定性数字守恒（双保险常开） |
| G0-3 政策正则对齐 | precision≥0.8 ∧ recall≥0.6 ∧ 合法引用误报≤2/20 | 不过线禁止进 GRPO reward |
| G0-4 reward 预飞行 | spearman≥0.5 ∧ 手术样本必胜 ∧ 退化样本 10/10 被闸 | 修 reward，不烧 GPU |
| G1-0 数据量 | 过门重写≥400 ∧ replay≥150 | 重写 prompt 迭代一次，仍不足 NO-GO |
| **G1-1 池正样本密度（RL 资格总闸）** | 注入后 200 带痕迹题×K8 本地判：≥50% 题有干净候选 ∧ ≥40% 有对比对 | <30%→补救 1 次→方案 B 引导采样 1 次→仍败则 DPO 退守锚定对并如实标注 |
| G1-2 注入守护 | Δtrace_free≥+0.03 ∧ CI>0 ∧ McNemar 净增 incorrect≤4 ∧ Δg CI 下界≥−0.02 ∧ len p50≤+25% | 补救 1 次（加量/2epoch） |
| G2-0 DPO 开训 | pairs≥160（目标≥400）∧ 设计效应≥0.013 ∧ heldout≥40 ∧ headroom 非 NA | 修 pair 不开训 |
| G2-1 DPO 训中 | accuracies@20≥0.55 ∧ chosen 不下坠 ∧ pref-acc≥26/40 | 补救 2 次（1 超参 1 数据），再败写报告收口 |
| G2-2 DPO 验收 | 相对 s1：Δtrace_free≥+0.015 ∧ CI>0（或净转化≥+12/87 McNemar p<0.05）∧ 守护全过 | Δ<+0.005 ∧ 净转化<+5 = DPO 不行，如实入表 |
| G3-0 GRPO 资格 | S2 相对 base 累计 Δ≥+0.05 ∧ smoke 组内方差门过 | 顺延，先修注入 |
| G3-1 GRPO 验收 | 相对 s2：**确定性净转化≥+8（exact p<0.05）∧ 计数降≥15%** ∧ Kimi 非劣（CI 下界≥−0.005）∧ 守护中性 | 补救 2 次（1 调权 1 KL），再败=GRPO 负结果如实入表，交付 DPO ckpt（链路仍含 GRPO 阶段=约束满足） |
| G3-2 探针哨兵 | §7 发散判据，训练内 HALT | 回滚上一好 ckpt+调权 1 次，再犯止损 |
| G-acct 记账 | \|端到端Δ−Σ阶段Δ\|≤0.01；judge 版本 hash 入全部 decision.json；224 读数次数≤1+补救数；ckpt 选择只用池内 held-in 60 题 | 对不上账=有读数掺假，raise |

## 11. 最低成本版本

**1 天版（回答"注入能否制造 RL 学习信号"+构对机器标定，≈1.9-2k Kimi 调用 + GPU 5-6h）**：
上午=G0 修复+再校准（180）+扰动测试（180）+reward 预飞行（0）∥ base 池零成本探针（200 题×K8 本地判，GPU 1h——若 base 池对比对率已≥40%，DPO 可与注入并行）。下午=mini 注入（300 题重写+k2 门 ≈900 调用 + SFT 19 步 40min）→注入后池探针（GPU 1h 零调用）→224 k=3 终审（672）。交付：G1-1 池密度 GO/NO-GO + 注入蒸馏增益预览 + 修复后裁判的全部标定数。
**3 天版**：上文 D0-D3 完整跑（首个可入账 RL 增益 = DPO 行）。
**最低限度"含 DPO+GRPO 全链路且不自欺"版（5-6 天）**：3 天版 + GRPO 120 步（reward 全本地零调用、哨兵 4×120+终审 672≈1.2k 调用）+1 次补救+报告日。不自欺的定义：①每阶段相对自己起点 paired Δ+CI 预注册；②GRPO 主仪器用确定性计数 McNemar（预期效应<裁判 MDE，用 Kimi 验收才是自欺）；③Δ∈[0,门槛) 如实报"正向但低于分辨率"；④负结果入表不补救第三次。

## 12. 给组长的汇报话术

"前三轮不是 RL 失败，是把测量仪器和学习信号对准目标的必经研发：v1 查出 reward 在奖励照抄（机制级定位），v2 查出裁判噪声淹没效应量（MDE 0.013 vs 设计效应 0.006），v3 证明旧采样池没有可学正样本（best-of-8 复判≈base）。三个根因各有工程修复，且新裁判已被实验证明读得出目标变化（机械去痕迹 +0.038，CI 全正）。14B 是同构阳性对照：注入分布后，同套 DPO +0.056、GRPO +0.01。本轮按正确顺序重建：注入（蒸馏账）→ on-policy DPO（RL 账）→ 本地确定性 reward GRPO（RL 账），每阶段相对自己起点分账并做望远镜核账，保证不把蒸馏收益记成 RL 收益。预期：注入 +0.03~0.05，DPO +0.015~0.035，GRPO +0.005~0.015（GRPO 用确定性痕迹计数验收，因为其量级低于裁判分辨率）。请不要拿 14B 的 +0.056 直接预期 32B：我们起点已是 0.775，余量小一个量级。3 天出首个 RL 入账读数，5-6 天出完整三行记账表。"

## 13. 简历/面试素材（RL 走通叙事）

①分阶段增益归因协议（telescoping paired Δ+CI+McNemar，蒸馏/RL 分账）；②把"RL 无效"拆成三个可测根因并各转成 raise 硬门（池密度/裁判 MDE/reward 错位）；③best-of-K 正样本密度探针作 RL 可行性前置门（零 API 成本）；④"预期效应量<judge MDE"的验收难题→确定性指标 McNemar 主仪器+LLM judge 降守门（评测计量学）；⑤GRPO 反 hacking 哨兵（本地 reward 与外部 judge 符号一致性，训练内 raise）；⑥政策引用"合法 vs 机械"的 reward 正则判别设计（防 reward 教模型丢法律依据）。
