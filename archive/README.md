# archive/ — 历史归档（便于未来查验，非活跃代码）

重构时把所有**被取代的历史世代**整体移到这里。它们记录了项目"四轮 RL 净≈0 的诊断长征"——
每一轮怎么做、为什么没成、拆出什么根因。活跃链路只剩 `../code/`（V2/derag2），实验全貌见
`../32B强化学习_实验报告.md`。

> 说明：这些历史代码**只为查阅保留**。它们当年从各自的 `code/` 上下文运行，移到 archive 后
> 相对 import（`from pipeline.x import ...`）不再解析，**不保证能原地跑**——要复现请回看报告对应章节。
> 大产物（`*.jsonl`/`*.log`/模型权重/`runs/`）一律不入库。

---

## 一、`code_legacy/` — 历史世代源码（按世代分目录）

| 世代 | 是什么 | 结局 / 为什么被取代 |
|---|---|---|
| `original_run/` | 最初的全链：冷启动→PMI→RFT→DPO→GRPO（`run.py`） | 跑通但 RL 净≈0；PMI 尺子验证了 AUC 反相关、默认关；旧 humanness 口径后被永久退役 |
| `corrected_v1/` | 合并 RFT 底座修正 πref 后重跑 DPO/GRPO（`run_merged_dpo_grpo.py`） | 修对了 πref（swift `disable_adapter` 把参考退回 V1 的 bug），但 DPO/GRPO Δh 仍全负、RFT base 0.697 最优 |
| `corrected_v2/` | 发现"奖励量的像人 ≠ Kimi 量的"（Spearman 0.077）后重选 pair 做 mini DPO | 裁判噪声 σ=0.117 + 0.85 封顶 + 设计效应 < MDE，开训前就注定读不出；五项全没涨 |
| `v3_probes/` | 零训练诊断 E0–E3 / 去检索腔 rubric（`run_corrected_v3*.py`） | 证明"评分能读出去痕(实验A CI全正)、但模型采样里没有可学好样本(实验B跨0)"；旧 humanness 口径退役 |
| `derag_v4/` | 先造好样本再训：注入 SFT→on-policy DPO→确定性 reward GRPO | 跑完仍净≈0；致命=痕迹计数器里 60% 是 policy_source（合法政策引用）误判，等于优化一个六成假阳的指标 |
| `derag_v5_probe/` | headroom 探针：1 天零训练先判"有没有救"（`run_derag_v5_probe.py`） | 产出关键正结论 **X_kimi=0.79**——79% 病题里"think 干净∧答案不漂"的好样本真实存在，瓶颈在奖励函数 → 据此开第二条路径 |

> 注：`step151_rft_selfsample` / `step152_v1_support` / `v5_probe_common` / `reward_v3` / `v3_utils`
> 出身于探针世代，但被最终链路（`../code/`）复用，**没有归档在此**，仍在 `../code/pipeline/`。

## 二、文档归档

| 目录 / 文件 | 内容 |
|---|---|
| `docs_journal/` | 编号分析文档 85–109（fable5 诊断 / corrected-v2 审计 / reward v5 裁决 / judgecal 设计等，纯人读） |
| `docs_research/` | 奖励函数与裁判体系业内调研 |
| `docs_learning/` | vLLM 入门教学样例 |
| `legacy_masters/` | 重构前的 living-master 与提示词：理解与现状备忘 / 技术方案 V1·V2 / **实验全历程·从零讲起** / PROJECT_INDEX / codex 提示词 / CODE_HEALTH。它们是新报告的**素材来源** |
| `legacy_reports/` | 重构前的旧实验报告（md/html/docx），已被根目录整合版取代 |
| `results/` | 关键实测证据：`judgecal/`（裁判标定 σ 表与图）、`159_probe_summary.md`（headroom 探针结论）等 |

## 三、口径提醒（数字不可横向比）

三套验收口径互不相同，引用时务必看清来自哪一套：
- **V2 三件套**（`code/` 最终链）：500 冻结验收，Kimi 干净分 0–10 + 规则 + 答案在池率。**这是最终结论的口径。**
- **derag_v4 确定性口径**：224，纯 trace 正则。
- **14B / 早期 humanness 口径**：相似度泵，已弃。
