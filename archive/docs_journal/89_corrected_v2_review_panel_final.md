# 89 · corrected-v2 四角色评审结论与最终实施方案（交 Codex 落地）

> 评审输入：88 草案、90 reward 对齐审计、91 旧 pair Kimi 审计、code/（step10/12、reward.py、dpo_on_model.sh、kimi_client、run.py）。
> 四个角色 agent 独立审阅后由主审仲裁。所有产物 corrected-v2 命名，不覆盖 corrected-v1 与历史目录。

---

## 一、四个 agent 独立意见

### Reward/Preference Agent — 有条件可行

1. 旧 `60_dpo_pairs.jsonl` 作为训练集**必须整体废弃**：胜率 34.0%、margin +0.025、本地 margin vs Kimi margin Spearman=−0.002、DPO loss 钉死 ln2——零信号偏好，加 epoch/调 beta 救不回。
2. 方案 Y（翻转救援）严格劣于方案 X：可救对仅 ~25-40；更根本的是旧 pair 是被零相关代理选出的 max/min，每 query 中间 6 个候选从未被 Kimi 看过——真正 h=0.85 的 chosen 大概率在那里。
3. 单判 |Δh|≥0.25 的翻转不可直接采信（同文本重判 Δ 实测达 0.15，叠加 winner's curse）：翻转对必须二次复判，两次同向 ∧ 两次均值 ≥0.2 才保留。
4. 方案 X 与重 rollout 在对齐上等价、Kimi 成本相同（瓶颈是 API 不是 GPU）；旧池 within-query h 跨度 0.35→0.85，多样性够。mini 用 X，前提是工程核实通过。
5. chosen h≥0.7 太低（base 均值就是 0.697，Kimi 给资料归纳腔也打 0.7）→ **h≥0.75 且 ≥50% 的对 h_chosen≥0.8**；产出 <80 对再降回 0.7，但 traces 规则必须兜底。
6. acc 必须锁方向：`acc_tier(chosen) ≥ acc_tier(rejected)`（同档优先，放宽档差对 ≤20%），chosen 加本地 `fact_recall(answer,gold)≥0.7` 廉价闸——pair 874 证明放任会教模型丢事实换自然感。
7. chosen g≥0.8（非草案 0.7）且 `g_chosen ≥ g_rejected − 0.05`：de-RAG 丢 grounding 有前科，不锁方向 DPO 会把"更不 grounded"学成偏好。
8. **chosen 必须 traces ∩ {ref_enumeration, verbatim_copy, explicit_ref} = ∅**：step90 证明 local copy 0.14-0.19 时 Kimi 已判 verbatim_copy，数值阈值看不见结构性罗列，judge 的 traces 字段是零成本的结构信号，直接排除 pair 320 类"对但 RAG 腔"chosen。copy_ratio(think, refs∪gold)≤0.30 仅作预筛。
9. 防讨好 Kimi 三件套：连接词密度 p90 较 base 涨 >50% 报警；第二裁判 50 样本交叉判；结构上限制单轮离线 DPO、不对同一裁判迭代。
10. 同意 GRPO 推迟到 DPO 后。新增准入：用 8k 判分数据实测 `P(fact_recall≥0.6 | Kimi acc=incorrect) ≤ 20%`，否则错答案会进入 humanness 渐变区被正向优化。

### Training Stability Agent — 有条件可行

1. **mini 按草案剂量不可判读**：有效 batch 64 × 1 epoch = 1-2 个优化步，lr 调度都没启动。修正：ga=2（有效 batch 16）、3 epochs、warmup 0.1 + cosine → 120 对 ≈22 步；产出 <100 对则 ga=1。
2. **Go 门槛改 held-out**：留 25-30 对不训练，held-out rewards/accuracies ≥0.7（0.6 对随机的二项 p≈0.21 不显著，0.7 才 p≈0.04）；多 epoch 后训练内指标被记忆污染，不可做 Go 依据。
3. 坍缩监控：rewards/chosen 下坠 + margins 上行 = chosen 概率同跌的经典退化，立即停；每 epoch 末 60 条 mini Kimi 评测 + 口癖统计（开头 n-gram 频率、distinct-2、长度漂移）；epochs 封顶 3，eval loss 连升 2 点早停。
4. beta 0.1 / lr 5e-6 保持（控制变量）；不要 1e-5；**pair 少时 beta 应升不应降**（出现退化征兆 → 0.2 重跑）。
5. DPO-first 正确：直接 GRPO = 把预算压在未验证的在线 scorer 上（step90 已证代理 Spearman 0.077 时 GRPO 沿错误方向走）。蒸馏 scorer 探针不过门（Spearman≥0.6 ∧ AUC≥0.8）就整段不上，不降门槛硬上。
6. GRPO v2 超参：**优先升 lr 至 1e-6，beta 保留 0.06-0.08 不降 0.04**（方向未验证前别同时松两个旋钮）；KL 带 1e-3~1e-2 配行动规则：30 步后 KL<5e-4 → lr×2；KL>1e-2 持续 10 步 → 回退上一 ckpt + lr 减半。
7. 长度规则收紧两层：每对 |Δlen|/max ≤0.30；全集 |mean(len_c)−mean(len_r)|/mean(len_r) ≤10%（DPO 无长度归一，旧 pair 已有 ~7% 系统差）。
8. full 剂量：ga=4（batch 32）、2 epochs ≈84 步、10% 留出 eval、load_best_model_at_end；现脚本 save_strategy=epoch + save_total_limit=2 会冲掉最优 ckpt，改 steps 保存 + limit≥3。
9. NLL 正则要加（防 chosen 概率同跌）：mini A/B `rpo_alpha ∈ {0, 1.0}` 两枝各 ~45min，full 用胜者；swift 4.0.1 确切旗标名【需 smoke 验证】，不要按记忆写死。
10. dpo 脚本 ga/epochs 硬编码需提成环境变量；eval split / load_best 在 swift rlhf 下的透传旗标同样需 smoke。

### Evaluation/Risk Agent — 有条件可行

1. **Δh≥+0.02 在单判非配对下立不住**（z≈1.05，掷硬币水平）；配对检验（base 逐样本判分已存在，零成本）SE≈0.012，必须为主。
2. mini 顺序设计：新模型单判 224 次 + 配对；**Δh≥+0.03 直接 Go**；落入 [0.02,0.03) 决策带才双臂补判（+448+基线 224）。full 终评双臂双判 896 次，Δh≥+0.04 ≈ 4σ，这才是上会数字。
3. acc 护栏改口径：**c+p% ≥90.0% 且配对新增 incorrect ≤4**；平均分 Δacc≥−0.01 低于噪声底，是假精度，只报告不卡门。
4. 自我裁判（Kimi 构对+Kimi 验收）风险，报 Δh 必须并列四件旁证：trace 计数（verbatim_copy 60→应 ≤52）、copy_ratio 分布左移、10-20 条人工盲评（新模型偏好 ≥60%）、第二裁判 50-60 条方向一致 ≥70%；rubric 冻结 v1 版禁迭代。
5. 灰区（0<Δh<0.02）不许无条件放量：需 (a) held-out ≥0.70 (b) trace 计数方向正确 (c) 配对 h_up−h_down>0 三者齐备才升级，且只升级一次。**前置免费检查：best-of-K headroom**——K=8 池内 Kimi-h 最优候选均值领先 base <+0.05 则池内无信号，放量必败。
6. 失败叙事五件套（缺一结论塌）：pair 可学性（held-out≥0.7 + loss<0.60）；pair 复核 50 对方向保持率 ≥80%；策略确实动了（KL 进 1e-3 带 + 文本变化样本 ≥50%）；评测功效（双判配对 CI ±0.02）；headroom 归因——"优化缺口"与"数据上限"是两个对组长含义完全不同的结论。
7. 为什么不交付 GRPO acc 0.831 的口径：+1.3pt 在噪声内（净 +7/224）；主指标退步（h↓、verbatim_copy 60→63）；机制已查明是照抄泵伪收益；同配方在 DPO 起点 acc 反掉 0.809，不稳健。
8. chosen 允许 partial 须加分数下限 **acc_score≥0.6**（Kimi partial 低至 0.4，否则是 acc 牺牲的最大暗道）。
9. grounded 加尾部条款：**g≤0.5 样本数 ≤ base+3**（de-RAG 的 grounding 损失历史上是尾部形态，均值护栏照不见）。
10. 救援 pair 必须二次判分 + 来源标签单列，便于事后归因。

### Engineering Agent — 有条件可行

1. 方案 X 代码侧前提成立：step10_rollout 输出每 query 的 candidates K=8 全文，step12 直读；pool=SFT_TRAIN(2014) 天然排除 224 评测集。
2. **唯一硬风险**：60_dpo_rollout 可能由首版 RFT（acc 0.784 选样 bug 版）生成并被 run.py `exists()` 门固化。一条服务器命令裁决（见 §三 Phase 0）。
3. X 省 GPU 不省 Kimi；即便验证失败，重 rollout 仅 mini 15-25min / full 1-2.5h，不构成路线阻塞。
4. Kimi 吞吐实测锚点 ~19 次/min @3 workers（224 次 11.6min）；**mini ≤2000 次 ≈2-2.5h；full 8000 次一夜 8.5-11h；不要 4 workers**（429 退避吃掉收益）；不要全 2014q（16k 次跨两夜）。
5. 若 v2 评测双判，**基线必须补第二判 224 次**，否则 Δh 在不可比口径上裁决。
6. step93 断点续跑 **done-key 必须 (qid, cand_idx)**（现有 step01/06/07 模式的裸 query key 会跳过同 query 其余 7 候选）；失败返回 None 不落盘（step04 失败写 0 分的语义会污染 rejected 选择）。
7. rubric 同版本是硬约束：93/94/95 全部复用 step91 的 judge_text（JUDGE_SYSTEM/TEMPLATE/截断口径/temp=0.0），抽成共享函数，禁止第三份复制粘贴。
8. 同 query 候选按 text_sha1 去重判分（temp0.9 下重复常见，省 5-15%）；mini→full 用同一 scores 文件做续跑超集，白省 1/8 预算。
9. **mini 不需要 merge**：serve RFT_MERGED + v2 LoRA adapter 直接评测（serve_model_vllm.sh 原生支持），省 ~30min + 65GB；merge 推迟到 full Go 后。
10. 方案 Y 弃用（被 X 严格覆盖：X 判全 8 候选、能产硬负样本，Y 只看旧极值对）；残值 = X 失败当天救急 ~30-40 对。命名预检 7 条拒绝清单见 §三。

---

## 二、冲突点与最终折中

| # | 冲突 | 折中裁决 |
|---|---|---|
| C1 | chosen h 门槛：草案 0.7 vs Reward 0.75 | **0.75 + traces=∅ 为主档**；120q 产出 <80 对时降 0.7，traces 排除与 fact_recall 闸不放松 |
| C2 | mini Go 门槛三个版本（草案 Δh≥0.02+训练内 0.6 / Training held-out 0.7 / Eval 配对 0.03+决策带） | **三层合并**：数据层 held-out pref-acc ≥0.7；效果层配对 Δh≥+0.03 直接 Go，[0.02,0.03) 决策带→双臂双判+符号检验 p≤0.10；护栏层 c+p%≥90.0% ∧ 新增 incorrect≤4 ∧ grounded≥0.84 ∧ g≤0.5 尾部≤base+3 ∧ verbatim_copy 计数不升 |
| C3 | X vs 重 rollout vs Y | **X 优先，Phase 0 一条命令裁决**；失败→step92 重 rollout（成本可忽略）；Y 不作路线，仅回收已付费 300 判分（19 强对齐 + 双判翻转 ~30-40 对，带 source 标签并入 v2 池） |
| C4 | margin 0.25 单判 vs Eval 担心 Kimi 重复性 | 主档 **0.25 单判**；构对后抽 50 对复判，方向保持率 ≥80% 作过程验证（<70% 触发 Eval blocker：构对全双判或 margin 升 0.35）；放宽档/翻转一律双判 |
| C5 | rpo_alpha A/B 双倍训练成本 | 采纳但收口：两枝各训 ~45min，用 held-out pref-acc + 60 条 mini Kimi 评测选胜者，**仅胜者跑 224 全评**（评测预算不翻倍） |
| C6 | GRPO beta：草案 0.04 vs Training 0.06-0.08 | **0.06-0.08 + lr 升至 1e-6**；KL 行动规则照采 |
| C7 | 评测预算：Eval 顺序加判 vs Engineering ≤2000 上限 | 兼容：mini 默认仅 224 单判（配对用现成 base 判分）；进决策带才 +448+224（基线补判），仍在 ≤2000 内 |
| C8 | headroom 检查放哪 | 放 step94 构对报告里（93 判分完即免费可算）：**best-of-K Δh̄ <+0.05 → 直接 No-Go，不训**，失败叙事走"数据/模型上限" |

**对用户六个问题的直接回答**：
1. 旧 60_dpo_pairs：**废弃**作为训练集。保留条件 = 仅回收 step91 已判 150 对中（a）19 个强对齐对（b）翻转后双判同向且均值 ≥0.2、acc 不降的 ~7-15 对，带 `source=legacy` 标签作补充，≤40 对。
2. 新 pair 构造：见 §三 Phase 1（rollout 起点 = 60_dpo_rollout 旧池【X，待验证】或 RFT merged base 重采【fallback，K=8 temp0.9-1.0 top_p0.95】；Kimi 字段 h/g/acc+acc_score/traces/comment；阈值见 C1-C4 与下表）。
3. mini：120q、ga2×3epoch×lr5e-6×beta0.1、A/B rpo_alpha、**先 DPO**；Go 门槛 = C2 三层。
4. full：1000q→≥1.5k 对→ga4×2epoch+load_best→双臂双判终评 Δh≥+0.04。
5. 现在不做：不加 GRPO 步数、不调 beta 救旧 DPO、不用旧本地 reward 构对或在线训练、不重采 2014 全量判分、不在 mini 阶段 merge、不对 Kimi 迭代多轮 prompt。
6. 工程清单见 §三。

**v2 构对规则终表**（写入 config 的 `PAIR_*` 常数）：

| 项 | 值 |
|---|---|
| chosen | Kimi h≥0.75（≥50% 的对 h≥0.8）∧ traces∩{ref_enumeration,verbatim_copy,explicit_ref}=∅ ∧ g≥0.8 ∧ (acc=correct 或 partial∧acc_score≥0.6) ∧ fact_recall(answer,gold)≥0.7 ∧ copy_ratio(think,refs∪gold)≤0.30 |
| rejected | 同 query ∧ acc_tier ≤ chosen（同档优先，档差对全集 ≤20%）∧ g≥0.6 ∧ h ≤ h_chosen−0.25 |
| 长度 | 每对 \|Δlen\|/max≤0.30；全集均值差 ≤10%（报告必须输出分布） |
| 方向约束 | g_chosen ≥ g_rejected − 0.05 |
| 硬负样本 | 同 query "对且自然" vs "对但 traces 含 ref_enumeration/verbatim_copy"，目标占比 ≥30%，单独打 tag |
| 留出 | 25-30 对不训练 → heldout 文件，算 held-out pref-acc |
| 复核 | 随机 50 对二次判分，方向保持率 ≥80% 才放行训练 |

---

## 三、Codex 可执行实施清单

### Phase 0 · 服务器验证（5 分钟，无 GPU，今天先跑）

```bash
cd /home/nvme01/zhjg && \
stat -c '%y %n' output/60_dpo_rollout.jsonl ckpts/v1-32b-cs-rft-lora/v*/checkpoint-*/adapter_config.json && \
grep -hE 'rollout：model=|阶段5|build_dpo_rollout' logs/pipeline.log 2>/dev/null | tail -8 && \
/home/nvme02/conda/zhjg_rl/bin/python -X utf8 -c "import json;rows=[json.loads(l) for l in open('output/60_dpo_rollout.jsonl',encoding='utf-8')];print('queries=',len(rows),'K_set=',{len(r.get('candidates',[])) for r in rows},'fields=',sorted(rows[0]))"
```

判据：rollout mtime 晚于 `v0-20260609-192005/checkpoint-38` ∧ 日志显示 model=cs_rft ∧ queries≈2014 ∧ K_set=={8} → **走方案 X**；任一不满足 → mini 用 step92 重 rollout（120q×8，TP=8，~15-25min）。
同时跑 smoke：`swift rlhf --help | grep -iE 'rpo|split_dataset|load_best'`（决定 Q7/NLL 正则与 held-out 评测走原生还是手工）。

### Phase 1 · mini（X 成立时 GPU 净占用 ≈70min，全程 ≈4-5h）

| 步 | 新增/修改 | 产物 | 要点 |
|---|---|---|---|
| 1 | 新增 `pipeline/judge_common.py` | — | 从 step91 抽 judge_text（JUDGE_SYSTEM/TEMPLATE/_parse_json/截断口径 ref3000/think4000/ans2000/max_tokens512/temp0.0），93/94/95 共用，禁止复制粘贴 |
| 2 | 新增 `pipeline/step93_kimi_score_rollouts.py` | `output/93_corrected_v2_rollout_scores.jsonl` | 输入 60_dpo_rollout(或 92 产物)+`--queries 120 --seed 7`；done-key=(qid=sha1(query)[:12], cand_idx)；text_sha1 去重回填；失败 None 不落盘；增量 append+flush |
| 3 | 新增 `pipeline/step94_build_dpo_pairs_v2.py` | `output/94_corrected_v2_dpo_pairs.jsonl`、`94_corrected_v2_pairs_heldout.jsonl`、`94_corrected_v2_pair_report.md` | §二终表全部规则；报告必含：产出率、margin 分布、长度差分布、硬负占比、**best-of-K headroom**（<+0.05 直接 No-Go）、50 对复判方向保持率 |
| 4 | 新增 `swift/dpo_v2.sh`（复制 dpo_on_model.sh 改，不动 v1 脚本） | — | env 化 DPO_GA(默认2)/DPO_EPOCHS(3)/DPO_RPO_ALPHA/warmup_ratio 0.1；save_strategy steps + save_total_limit 3 |
| 5 | 训练 A/B 两枝 | `ckpts/v1-32b-corrected-v2-dpo-lora-rpo0`、`...-rpo1` | beta0.1 lr5e-6；监控 rewards/chosen 下坠即停 |
| 6 | 新增 `pipeline/step94b_heldout_pref_acc.py` | `output/94b_corrected_v2_heldout_acc.md` | 对 heldout 对算两枝隐式奖励准确率（swift 无原生 eval split 时的兜底）；选胜者 |
| 7 | 评测：serve RFT_MERGED + 胜者 adapter（不 merge） | `output/96_corrected_v2_dpo_infer.jsonl`、`96_corrected_v2_dpo_report.md` | 224 推理（TP=2 ~25min）+ Kimi 单判 224 + 与 80_judge 配对统计；决策带才补双判 448+基线 224 |
| 8 | 新增 `pipeline/step96b_style_monitor.py` | `output/96b_corrected_v2_style_monitor.md` | 连接词密度 p90、开头 n-gram 频率、distinct-2、长度漂移、trace 计数对比、copy_ratio 分布 |

Go/No-Go（全部满足才 full）：
- 数据层：held-out pref-acc ≥0.7；50 对复判方向保持 ≥80%；headroom ≥+0.05
- 效果层：配对 Δh≥+0.03（或决策带加判后 ≥+0.02 ∧ 符号检验 p≤0.10）；h_up−h_down ≥ +20
- 护栏层：c+p%≥90.0% ∧ 配对新增 incorrect≤4 ∧ grounded≥0.84 ∧ g≤0.5 样本 ≤base+3 ∧ verbatim_copy 计数 ≤60

### Phase 2 · full（mini Go 后，约 1.5-2 天含一夜 API）

1. step93 扩 `--queries 1000`（同一 scores 文件续跑，mini 960 条白省）→ 8000 次判分一夜（workers=3）。
2. step94 全量构对（目标 ≥1.5k 对，硬负 ≥30%，legacy 回收对带标签并入）。
3. `swift/dpo_v2.sh`：DPO_GA=4、DPO_EPOCHS=2、10% 留出、load_best（或 94b 手选）、胜者 rpo_alpha。
4. merge：沿用 `.partial`→验证→原子发布 → `models/v1-32b-corrected-v2-dpo-merged`；前置空盘 ≥80GiB 预检。
5. 终评：双臂双判 896 次 → `output/97_corrected_v2_full_report.md`；验收 **配对 Δh≥+0.04** + 护栏同 mini + 四件旁证（trace 计数、copy 分布、10-20 条人工盲评、第二裁判 50-60 条）。
6. 汇总 `output/98_corrected_v2_summary.md`。

### Phase 3 · GRPO v2（可选冲刺，full 达标后）

1. `pipeline/step95_distill_humanness_scorer.py`：用 ≥8k 判分蒸馏 7B scorer → `models/corrected-v2-humanness-scorer/` + `output/95_corrected_v2_scorer_probe.md`；**探针门 held-out Spearman≥0.6 ∧ AUC≥0.8，不过整段不上**。
2. `pipeline/reward_v2.py` + `swift/grpo_reward_plugin_v2.py`（注册 humanness_v2）；上线前用判分数据实测 fact_recall 闸泄漏率 ≤20%。
3. `swift/grpo_v2.sh`：lr 1e-6、beta 0.06-0.08、100-150 步、每 25 步 ckpt+60 条 mini 评测；KL 行动规则（30 步 <5e-4→lr×2；>1e-2 持续 10 步→回退+lr 减半）+ copy_ratio 上行报警。

### 防覆盖预检（新增 `scripts/preflight_corrected_v2.sh`，所有入口先跑）

拒绝：① 写路径不含 corrected_v2/corrected-v2；② 对 output/0-83 号段及 60_dpo_* 的任何写入；③ 94 pairs / 96-98 报告 / 新 ckpt 目录已存在且非显式 RESUME（93_scores.jsonl 是唯一 append 白名单）；④ 目标命中现有 ckpts/v1-32b-{coldstart,cs-rft,dpo,grpo*}-lora 或 models/*-merged；⑤ merge 前空盘 <80GiB；⑥ pgrep 命中活跃 rlhf/torchrun；⑦ step93 前 DASHSCOPE_API_KEY 未设。

### 现在明确不做

- 不加 GRPO 步数 / 不调 beta 救旧 DPO（零信号数据上剂量只会放大噪声）
- 不用旧本地 reward（R_human Spearman 0.077）构对或做在线 GRPO reward
- 不判全 2014q（16k 次跨两夜，边际收益递减，截 1000q）
- mini 阶段不 merge、不动 v1 任何脚本（复制改名）
- 不对 Kimi rubric 做任何迭代（冻结 v1 版，跨轮可比性优先）

---

## 四、仍缺的 3 项关键信息

1. **Phase 0 验证命令的输出**（60_dpo_rollout 的 mtime/来源/queries/K 完整性）——裁决 X vs 重 rollout，是当前唯一路线分叉点。
2. **DashScope 配额与近期 429 频率**——mini ≤2000 次、full ≈9000 次的排期与 workers 数依据。
3. **swift 4.0.1 rlhf 的旗标 smoke 输出**（`swift rlhf --help | grep -iE 'rpo|split_dataset|load_best'`）——决定 NLL 正则（rpo_alpha）与 held-out 评测走原生参数还是 94b 手工脚本。
