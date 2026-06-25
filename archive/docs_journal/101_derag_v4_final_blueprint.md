# 101 · derag_v4 最终工程图纸（硬继承旧链路，RL 必走通，无开放分叉）

> 本文是 Codex 照图施工的定稿。所有决策已定死。证据来源：旧链路代码亲读（step06/07/08/10/11/12、reward.py、judge_common.py、dpo_v2.sh、grpo_on_model.sh、14B step09/10）、历史报告实测（output/10/20/50/60/70/84）、98/99/100 号复盘、v3.1 标定（110-113）、本轮统计与 DPO 两角色定稿。
> 命名空间：`derag_v4`；云机 output=/home/nvme01/zhjg/output/derag_v4/<run_id>/、ckpts=/home/nvme01/zhjg/ckpts/derag_v4/{s1_sft,s2_dpo,s3_grpo}/<run_id>/、merged=/home/nvme01/zhjg/models/derag_v4_s{1,2,3}_merged_<run_id>；新文件编号 step120-139、报告 120-139 号。

---

## 一、旧链路失败的最终判定（结合全史，行级证据）

**旧链路对的部分（必须继承）**：
1. 冷启动机制有效且是全项目最大单步收益：h 0.233→0.680（20 号报告），explicit_ref 222→31、ref_enumeration 201→20（−90%）、verbatim 174→79。14B 同构（0.21→0.78）。
2. 治根条款有效：step06 REWRITE_TEMPLATE 条款 2/3（"依据口径数字结论必须完全来自参考资料""事实零改动"）+ step07 faithfulness + step08 grounded≥0.7 闸，把 grounded 守在 0.855-0.858。
3. RFT 综合奖励选样修复有效（acc 0.784→0.813 教训已入 step11）。
4. 全部训练工程已验证：dpo_v2.sh πref 语义、GRPO colocate 显存五连关、断点/监控/原子发布机制。
5. 14B 的 DPO margin 用**本地确定性 R_human**（step09_dpo.py:43-47，正则+copy_ratio，零裁判参与）→ +0.056——阳性对照的真正可迁移点。

**旧链路错的部分（各有行级定位，新方案逐一否定）**：
| # | 失败 | 行级证据 | 新方案的否定 |
|---|---|---|---|
| F1 | 改写 prompt 自己引入 policy_source（41→70）：条款 1 只禁显式字眼、条款 2 强制扣依据 → Kimi 改引《政策名》/文号清单；本地 trace 正则对《》罗列/"政策依据："标签全盲（reward.py 只抓〔〕号格式） | step06:31-33 + 10/20 号报告计数 | Stage1 改写 prompt 新增反罗列双条款；G1-2 验收盯 policy_source 计数不升 |
| F2 | 构对期照抄泵：chosen=max(综合reward)，reward 含与 gold 的 SequenceMatcher 正项 | step12:66,77 + reward.py:168,171 | 删除一切 gold 字符相似度正项；chosen 选取禁用任何分数排序（确定性键） |
| F3 | 两代 margin 都押在噪声上：v1 用综合 reward 差≥0.05（config.py:201），v2 用 Kimi h 差≥0.25 → 复判收缩 −68% | step12:79、step94:120,261、98 号实算 | margin 100% 确定性（trace_hits 差≥2 ∨ copy_ratio 差≥0.20） |
| F4 | 未注入分布上先跑 RL：池空（best-of-8 复判 0.701≈base 0.697） | 98 号实算 | Stage1 注入前置 + G1-1 池密度门（纯确定性）作 RL 资格总闸 |
| F5 | 旧池采样器偏置：RL_GEN_MAX_NEW_TOKENS=1024 vs 部署 1536 截断；GRPO max_completion=1024 同病 | config.py:151、grpo_on_model.sh:64 | 全部统一 1536/1792 |
| F6 | 剂量与锚缺失：v2 mini 12 步、rpo_alpha 静默 skip（dpo_v2.sh:31-37）、heldout 门从未实现 | 98 号事实 11 | ga 自适应保 ≥40 步、rpo_alpha=1.0 强制 raise、step94b 落地 |
| F7 | 测量失效：旧 h σ=0.117/r=0.434/0.85 封顶；judge 四 bug；门 fail-open/WARN 旁路；95b 双重浸入 | judge_common.py:130-131,169,238、step94:295-300、step95b:173-176 | judge_v3.0 + 三防线 + 全部 raise 硬停 |

**一句话判定**：冷启动证明了"注入有效"，三轮 RL 证明了"在未注入分布+噪声 margin+照抄泵 reward 下 RL 无信号"。新方案 = 用已成功的注入机制修复其已知缺陷（F1），再把 RL 建立在注入后的分布上，margin 与 reward 全部确定性化（F2/F3），剂量与测量补齐（F5/F6/F7）。

---

## 二、七个问题的定死回答

**Q1（Stage1 vs 旧冷启动）**：见上表 F1 与下方阶段表。差异七项：①目标=残留两类结构痕迹（verbatim 52/policy 61）而非全局风格（旧冷启动已完成全局）；②范围=仅带痕迹题 ~38.8% + replay，非全池 2014；③起点=RFT merged base 非 V1；④改写 prompt=继承 step06 条款 2/3（治根红线原样保留）+新增两条款（"不要机械罗列文件号/政策名清单，如需引用政策必须嵌入含'规定/按照/明确'的推理句，每句最多 1 个文号""不要大段照搬参考原文，用自己的话转述"）+保留"只输出推理过程不输出答案"；⑤门控=确定性全套（含 missing_nums 双向数字守恒——step06 的 introduced_nums 只查新增不查丢失，方向反了一半）+judge_v3.0 k=2 min()，非旧五道闸的单向 facts_ok+旧 h≥0.6；⑥answer=byte-equality 代码断言；⑦验收=新口径 paired CI+痕迹计数（专盯 policy_source 不升）。不是重复第一次失败：第一次失败不在冷启动（它成功了），在 RL；Stage1 重复的是已成功机制、修的是其已知缺陷。

**Q2（起点）**：**定死 = RFT merged base**（/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged）+ fresh LoRA → merge `derag_v4_s1_merged_<run_id>`。πref 链：S2=s1_merged、S3=s2_merged（均 disable-adapter 语义，dpo_v2.sh/grpo_on_model.sh 已验证）。不从 V1 重做：冷启动+RFT 三阶段成果（h +0.46、痕迹 −90%、acc 守 0.818）不可丢弃，重做即重新暴露准确率风险且多烧 2 天。

**Q3（G1-1 定稿）**：
- 干净候选（**纯确定性，零 Kimi**——Kimi 不在场就没有选择性回归）：`format_ok ∧ trace_hits_v3=0 ∧ copy_ratio(think, refs∪gold)≤0.30 ∧ enum_density<0.40 ∧ fact_recall(answer,gold)≥0.75 ∧ introduced_nums=∅ ∧ 反去依据地板(|facts(gold)|≥1 ⇒ facts(think)∩facts(refs∪gold)≥1)`。trace_hits_v3 = 修正正则（删裸〔〕号、排除"根据上述"自指、新增《》清单/标签式/裸指针三形态）。
- 测量协议：step121 对 200 条带痕迹训练题 × s1 的 K=8 temp0.9 max_new1536 采样逐候选算上式（GPU ~1h，n_kimi_calls=0 写入 decision 并 assert）。**注入前基线**：先对本地 60_dpo_rollout.jsonl 同口径实算一次（零 GPU 零 Kimi，今天可跑），记 P_clean_base/P_pair_base 入 MANIFEST（注：该池 1024 截断使其为产率下界）。
- 门值：P_clean（≥1 干净候选的题占比）**≥50%** ∧ P_pair（同题既有干净又有 trace_hits≥2∨copy≥0.40 候选）**≥40%** → GO。
- 处置树（唯一路径）：30-50% → 补救一次=对未达题 K=16 重采 → P_pair≥30% 则以 pair 目标 ≥300 继续；仍 <30% → 引导采样一轮（GUIDED_SUFFIX，K=16）；仍 <30% → DPO 改锚定对+手术对为主（自产下限放宽到 40%），**记账表如实标注"on-policy 挖掘失败"**，且该事实独立成段进终报。禁第三次补救。

**Q4（DPO）**：引导采样**允许、帽 15%**，仅对标准采样无干净候选的题追加；context distillation 论证 + `chosen_nll_ratio = NLL(chosen|部署prompt)/NLL(s1 greedy同题) ≤1.15` 可证伪门（引导桶被砍 >50% = 主张证伪，整桶弃用入报告）。训练 messages **一律部署 prompt**（step12:83/step94:161/14B step09:65 既有事实，零改动继承）。chosen/rejected 条件表、配比（自产≥70%；引导≤15%/手术≤10%/重写≤5%）、margin 100% 确定性、长度双层控制+fact_recall 配平、P≥160（目标 600）、heldout 40 对、rpo_alpha=1.0 强制——全表见 §三 Stage2 行与 §四。防"变短删依据"四道：pair 长度带宽 0.30+集级均长比 [0.9,1.1]+fact_recall(rej)≥fact_recall(cho)−0.05+反去依据地板。**比旧版好的证明**：§一 F2/F3 行级对照 + 14B 确定性 margin 阳性对照 + 设计效应预注册 ≥0.013（v2 的 0.006 开训前即被拦）。

**Q5（GRPO）**：见 §三 Stage3 行与 §四 reward v4。主验收不用 Kimi 均值的算术：GRPO 预期 +0.005~0.015 < paired MDE 0.013（111 号在真零假设上实测 CI 半宽 ±0.0125）——用 Kimi 验收注定假死或假过；主仪器=**确定性痕迹计数 McNemar**（trace_re_v3 口径，重测噪声=0）：相对 s2 净转化 ≥+8（exact p<0.05）∧ 痕迹总计数降 ≥15%；Kimi 仅非劣守门。GRPO 失败 → 预注册负结果段落（模板见 §六），链路仍含 GRPO 阶段+预注册门+诚实读数 = 约束满足。

**Q6（噪声进设计）**：噪声模型定稿表（写入 `120_v4_stat_protocol.json` 后冻结）：旧口径 σ=0.117/r12=0.434/MDE_k1=0.022 永久退役；新口径 trace_sd≤0.07（实测 0.056-0.060）/r12≥0.5（实测 0.63-0.67）/k=3 paired MDE=0.013/acc k=3 真多数票翻转 ≤2%。三防线：①margin 只准确定性；②Kimi 灰区门一律 k=2 **min()** 聚合（噪声只能错杀不能错放）；③构对/验收/选模三批判分 purpose∈{select,report,model_select} 物理分离，step120 assert 同 run_id 双用途即 raise。三仪器去相关：I1=temp0 标准序/I2=temp0.3/I3=temp0 字段序变体，k=3=I1+I2+I3，k=2=I1+I2，k=5 仅再校准。k 值表：G0-1 再校准 k=5（430 调用）｜G0-2 扰动 k=3（480）｜S1 源 acc 门 k=2 两判一致（~2k）｜S1 重写门 k=2 min()（~1.6k）｜S2 chosen 灰区 k=2 min()（≤2.5k 帽）｜S1/S2 224 验收 k=3（672/次）｜S3 哨兵 k=2（480/120 步）｜S3 验收=确定性零调用+Kimi 非劣 k=3（672）。溯源 schema：每行 judge_version/instrument_id/prompt_sha256/run_id/purpose/input_sha256/截断标记；跨版本禁比 assert；判分一次性绑定 cand_id 永不重判。

---

## 三、阶段图与工程表

```
Stage0 修裁判+修reward+预飞行 ──G0门──▶ Stage1 残留痕迹定向修复SFT（蒸馏账）──G1门──▶
Stage2 on-policy DPO（RL账）──G2门──▶ Stage3 本地reward GRPO（RL账）──G3门──▶ 三行记账终报
```

| 阶段 | 输入 | 输出 | 起点 | 训练/主脚本 | 评测 | GO 门 | 失败唯一处置 |
|---|---|---|---|---|---|---|---|
| **S0 修复标定**（D0 半天） | judge_common.py、reward.py、110-113 号现成判分、80_*_judge.jsonl | judge_v3.0、reward_v3.py、`120_v4_stat_protocol.json`、`121_pool_density_baseline.json`（旧池基线，零成本）、L0 锚文件（RFT base 224+池题 greedy think 长度表） | — | step120_precheck、step122_reward_preflight、step121（旧池模式） | G0-1 再校准 50 题 k=5；G0-2 扰动 40 题；G0-3 政策正则对齐（61 痕迹+50 干净+20 合法引用题）；G0-4 reward 预飞行（spearman≥0.5 用 110 号现成数据零调用、手术样本必胜、10 条退化样本全被闸） | 修到过为止，期间一切读数无效 |
| **S1 定向修复 SFT**（D1） | greedy 重推 2015 题（TP=8 1-2h max_new1536）、确定性预筛带痕迹题（~780）、step06 升级版改写 | `124_rewrites.jsonl`、过门重写 ≥400+replay≥150、`derag_v4_s1_merged` | RFT merged base + fresh LoRA r16/α32 | step124_rewrite_residual（继承 step06 治根条款+新增反罗列双条款）、step125_gate_rewrites（确定性全套+missing_nums+byte-equality+k=2 min()）、swift sft（lr5e-5、ep2、warmup0.05、eff batch 16） | 224 paired k=3 + step121（s1 模式） | G1-0 数据量（重写≥400∧replay≥150）；G1-1 池密度（P_clean≥50%∧P_pair≥40%）；G1-2 验收（Δtrace_free≥+0.03∧CI>0∧McNemar 净增 incorrect≤4∧Δg CI≥−0.02∧len p50≤+25%∧**policy_source 计数不升**∧verbatim 计数降） | 各补救 1 次（重写 prompt 迭代/加量）；G1-1 处置树见 Q3；G1-2 再败=以 RFT base 直接进 S2（s1:=RFT base，注入行记 0） |
| **S2 on-policy DPO**（D2） | s1 K=8 temp0.9 采样 ~1100 题（痕迹倾向 780+干净 300）、缺额触发引导采样 | `124_dpo_pairs_{train,heldout,meta}.jsonl`、`124_pair_decision.json`、`derag_v4_s2_merged` | s1_merged + fresh LoRA；πref=s1_merged | step10_rollout（加 --system_suffix/--temperature/--max_new 三参）、step124_build_dpo_pairs_v4、step125_chosen_nll_gate、dpo_v2.sh（四处补丁）、step94b_heldout_pref_acc | 224 paired k=3 双轨 | G2-0a 对数剂量（P≥160，ga 自适应保 ≥40 步）；G2-0b 设计效应 ≥0.013；G2-0c NLL 自产门；G2-0d 配比/长度/fact_recall 集级门；G2-1 训中（accuracies@20≥0.55∧chosen 不下坠∧pref-acc≥26/40）；G2-2 双轨验收（Δ≥+0.015∧CI>0 或净转化 ≥+12/87 McNemar p<0.05 + 守护八条）；G2-3 反双重浸入（fresh 复核保持率 ≥80%） | 补救恰 2 次（A=数据：margin 收紧至 hits 差≥3+引导补量；B=超参：pref-acc<0.65→lr 1e-5 / ≥0.65→beta 0.05，由实测自动判定）；再败 Δ<+0.005∧净转化<+5 → s2:=s1、DPO 行记 0 入表、按 G3 资格门决定 S3 形态 |
| **S3 GRPO**（D3） | s2、痕迹倾向题池 ~1100（70% trace-prone+30% 干净） | `ckpts/derag_v4/s3_grpo/<run_id>/`、`126_grpo_report.md` | s2_merged + fresh LoRA；πref=s2_merged | grpo_on_model.sh（补丁：max_completion **1792**、GRPO_SAVE_STEPS=25、limit=8、--log_completions、--reward_funcs derag_v4）、reward 插件=reward_v3、step123_grpo_probe_watcher | 确定性 McNemar 主审 + 224 k=3 Kimi 非劣 | G3-0 资格（s2 相对 base 累计 Δ≥+0.05∧smoke 2 步组内方差 ≥40% 组 std>0）；G3-1 验收（净转化 ≥+8 p<0.05∧计数降 ≥15%∧Kimi 非劣 CI 下界 ≥−0.005∧Δg CI≥−0.02∧acc McNemar 中性）；G3-2 哨兵（每 25 步，发散 HALT） | 补救恰 2 次（1 调权 1 KL）；再败=负结果按 §六模板入表，交付 s2，链路仍完整；资格门不过=降级 smoke+60 步半预算+负结果协议 |

GRPO 配置定稿：K=8、temp1.0、lr=1e-6、beta=0.04、120 步+探针早停、KL 行动表（带 [1e-3,1e-2]；<5e-4×20 步→lr×2；>3e-2×5 步→SIGTERM 回滚 beta×2；>0.1 单步硬停）、colocate 五连关原样沿用（TP=8/util0.5/maxlen6144/move_model_batches16/sleep+offload/zero3/NCCL_RAS_ENABLE=0）。
reward v4 = §100 号 §7 公式不变，仅两处更新：①policy_pen 用 G0-3 过门的正则 v2（合法行内引用零罚）；②**L0 长度锚冻结为 RFT merged base 的 greedy think 长度**（S1/S2/S3 同一锚，防逐阶段下行棘轮——红队修正）。
哨兵 HALT 全表：本地 T 两窗连涨 ≥0.05 而探针 trace_free Δ≤0；探针 grounded 降>0.03；correct 减>3/60；verbatim 计数 vs step0 +20%；copy_ratio↑伴 trace_hits↓（同义改写逃检签名）；长度漂出带；distinct-2 降>10%；"裁判报痕迹∧本地 hits=0"比率升>10pt（正则盲区扩张→收割文本扩 _TRACE_RE 从上一好 ckpt 重启）。

## 四、代码修改清单（文件级，Codex 照做粒度）

1. **judge_common.py**：①aggregate 真多数票（mode，平票取保守档，废 round(mean)）；②parse 缺字段 raise 重试 ≤2 次后标 judge_invalid 剔除（废 `or 0.0`）；③ref 截断 3000→6000、think→6000+truncated 标记；④DERAG_JUDGE_TEMPLATE→v3：证据先行（trace_spans[{type,quote≤30字,reason}]→unsupported_facts→think_answer_conflict→分数）、0.5 中段锚、反印象分条款（给 1.0 须 unsupported_facts=∅）、政策引用判别条款（合法=嵌入含"规定/按照/明确"推理句每句 ≤1 文号；痕迹=清单式 ≥3 连排/标签式"政策依据："/裸指针 ≥2）、长度免疫条款、6 条版本化 few-shot（非评测题，judge_v3.0）；⑤judge_text_k 三仪器（I1/I2/I3）；⑥每行写溯源 schema 全字段。
2. **新 pipeline/reward_v3.py**（不动 reward.py 原件）：①_TRACE_RE_V3：删裸 `〔\d+〕\s*号`、"根据(以上|上述)"加 `(?!分析|计算|推理|结论|判断)` 负断言、新增 policy_list（≥3 文号/《》连排）/label（"政策依据[:：]"限 think）/bare_pointer（句含文号无内容词 ≥2 计罚）；②enum_density、G_degen（悬空句/碎句 >20%/8-gram 自重复 ≥3/distinct-2<base p5）、L0 带宽（锚=RFT base greedy）；③数字守恒（含一步四则派生白名单容差 1%、反去依据地板）；④fact_recall+极性矛盾对表；⑤reward v4 总公式（删一切 gold 相似度正项）；⑥export 干净候选判据 clean(y) 供 121/124 共用（同文件同阈值，DPO 门与 GRPO reward 同尺）。
3. **新 step120_v4_precheck.py**：输出目录非空 raise；路径命中历史 ckpt raise；pool∩eval224=∅（原文+规整化双口径，含重写产物与探针 60 题三向）；MANIFEST.json（输入 sha256+行数+judge_version+trace_re 版本）；磁盘 ≥200GB；judge run purpose 分离 assert。
4. **新 step121_pool_density_probe.py**：--pool <rollout.jsonl> --mode {old_pool,s1}，逐候选 clean(y)，输出 P_clean/P_pair 分带痕迹题/全题两口径；assert 模块未导入 kimi_client，decision 记 n_kimi_calls=0。
5. **新 step122_reward_preflight.py**：用 110 号 k=3 现成判分算 spearman(本地 T, trace_free)≥0.5（零新调用）；112 号手术对 assert 手术版 reward 更高且过 G_degen；10 条手造退化样本 assert 全被闸。
6. **新 step124_rewrite_residual.py**（step06 改造版）：REWRITE_SYSTEM/TEMPLATE 继承条款 2/3 原文+新增反罗列双条款+删"只输出推理过程"外的歧义；输入=greedy 重推的带痕迹题；保留增量落盘/断点续跑/introduced_nums，**新增 missing_nums**（_nums(original)−_nums(natural) ⊆ 序号白名单）。
7. **新 step125_gate_rewrites.py / step125_chosen_nll_gate.py**：重写门（确定性全套+k=2 min(trace_free)≥0.85∧min(grounded)≥0.85+answer byte-equality assert+长度带宽 [0.6,1.3]×L0+口癖前缀 >5% 重试）；NLL 门（s1_merged bf16 2×A800，比值 ≤1.15，分布入 decision）。
8. **新 step124_build_dpo_pairs_v4.py**（不改 step12/94 原件）：chosen/rejected/margin/配比/长度/fact_recall 配平全按 §二 Q4；逐闸漏斗计数；assert deterministic_margin_evidence 非空、G0-1 PASS 存在、Kimi 分差不在 margin 路径。
9. **dpo_v2.sh 四补丁**：rpo_alpha 无条件传 1.0（不识别即非零退出）；P<400→ga=1 自动分支；DPO_EPOCHS 默认 2；save_total_limit 8+trainer_state 归档。
10. **grpo_on_model.sh 四补丁**：GRPO_MAX_COMPLETION 默认 1792；GRPO_SAVE_STEPS=25；save_total_limit 8；--log_completions true、--reward_funcs derag_v4（插件指 reward_v3）。
11. **新 step94b_heldout_pref_acc.py**：n=40、隐式偏好准确率、26/40 门、qid 清单 sha256 入 decision、与 224/探针三向不相交 assert。
12. **新 step123_grpo_probe_watcher.py**：消费 log_completions，每 25 步本地计数+60 题 Kimi k=2 异步，HALT 文件机制（判据见 §三）。
13. **新 step126_v4_report.py**：三行记账表+望远镜核账+痕迹演化表延长（V1 222/201/174/41 → 冷启动 31/20/79/70 → RFT 11/52/10/61(k3) → s1 → s2 → s3）+漏斗审计附录。

## 五、云机命令与监控

```bash
cd /mnt/pfs/zhjg/code && export DASHSCOPE_API_KEY=...
# D0
python pipeline/step120_v4_precheck.py --run_id $(date +%Y%m%d_%H%M%S)        # 写 MANIFEST，输出 RUN_ID
python pipeline/step122_reward_preflight.py --run_id $RUN_ID                  # 零 GPU 零新调用
python pipeline/step121_pool_density_probe.py --pool output/60_dpo_rollout.jsonl --mode old_pool --run_id $RUN_ID
bash scripts/run_derag_v4.sh --stage s0_judge_calib --run_id $RUN_ID          # G0-1/G0-2/G0-3（~1.1k 调用）
# D1
bash scripts/run_derag_v4.sh --stage s1 --run_id $RUN_ID                      # 重推→预筛→重写(过夜~5.5k调用)→SFT→merge→121 s1模式→224验收
# D2
bash scripts/run_derag_v4.sh --stage s2 --run_id $RUN_ID                      # 采样→构对→NLL门→DPO→94b→merge→224双轨验收
# D3
bash scripts/run_derag_v4.sh --stage s3 --run_id $RUN_ID                      # smoke→GRPO 120步(探针随跑)→McNemar主审+Kimi非劣
# 监控（沿用 monitor 模式：raw log 分离+状态文件）
bash scripts/monitor_derag_v4.sh --run_id $RUN_ID
tail -f /home/nvme01/zhjg/logs/derag_v4/$RUN_ID/state.log                     # 阶段/最新 loss/KL/reward/门状态
```
断点续跑：每 stage 入口检查上游 decision.json 的 PASS 标记；step124 重写/判分增量落盘（继承 step06 模式，done-key=(qid,cand_id)）。
排期：D0 半天（GPU 0）｜D1 一天（GPU ~3.5h+过夜 API）｜D2 一天（GPU ~4h+API ~3.5k）｜D3 一天（GPU ~6.5h+API ~1.2k）。名义 3.5 天，含补救 7 日历日硬帽；Kimi 总账 ≈10-13k 调用。

## 六、结果报告模板（126 号）

```
| 阶段 | 起点→终点 | Δtrace_free (95%CI) | 确定性净转化/87 | Δgrounded | acc McNemar b/c | c+p% | 痕迹计数 e/v/n/p | 账目 |
| S1 注入 | RFT→s1 | +0.0xx [..] | +xx | ... | ../.. | ..% | ../../../.. | 蒸馏 |
| S2 DPO  | s1→s2  | +0.0xx [..] | +xx | ... | ../.. | ..% | ../../../.. | RL |
| S3 GRPO | s2→s3  | +0.0xx [..] | +xx | ... | ../.. | ..% | ../../../.. | RL |
| 合计核账 | RFT→s3 | 端到端 vs Σ阶段 差 ≤0.01 ✔/✘ | | | | | | ΔRL=S2+S3 |
附录 A 痕迹演化全史表（V1→冷启动→RFT→s1→s2→s3）｜B 构对漏斗｜C 判分溯源（judge_version/run_id/调用量）｜D 非自产对占比与分桶 pref-acc 消融（记账披露）
GRPO 负结果模板段（预注册）：「GRPO 相对 s2 净转化 +x（门 +8，未过/过）；Kimi 非劣 ✔/✘；哨兵触发 x 次；按预注册门判定 GRPO 增量【成立/不成立】。该阶段以预注册协议完整执行并如实入账，链路含 GRPO 阶段的项目要求已满足。」
```

## 七、给组长的诚实话术

"旧链路跑通过且冷启动成功（痕迹 −90%、h 0.23→0.70、grounded 守住），但三轮 RL 无提升。我们没有推翻它，而是做了行级归因：①旧改写在'去检索腔+必须扣依据'双指令下自己引入了文号罗列（41→70），残留的两类结构痕迹正是它修不掉的；②两代 DPO 的 margin 一个押在含照抄泵的本地综合分上、一个押在 Kimi 噪声上（复判收缩 −68%）——而 14B 成功那轮用的恰是本地确定性 margin；③RL 跑在了没有正样本的分布上（best-of-8 复判≈base）。derag_v4 把这三个钉子逐个拔掉：定向修复注入（记蒸馏账）→注入后 on-policy DPO（确定性 margin，RL 账）→本地确定性 reward GRPO（RL 账），每阶段相对自己起点配对记账+望远镜核账。预期：注入 +0.03~0.05、DPO +0.015~0.035、GRPO +0.005~0.015（后者低于裁判分辨率，验收换确定性痕迹计数 McNemar）。每道门 raise 硬停、补救次数预注册，负结果如实入表——3.5 天名义、7 天硬帽，任何分支下都交付含 DPO+GRPO 的完整链路与三行记账表。"
