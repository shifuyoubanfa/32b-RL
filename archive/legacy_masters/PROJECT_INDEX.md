# 32B 强化学习项目 · 总索引

> 目标：让 V1-32B 的 `<think>` 不再暴露机械念手册/查 RAG 的痕迹（trace-free），同时
> `<answer>` 不漂移 V1 自己会给的答案、事实/数字/政策引用保持 grounded。
> 详细背景与现状看 [32B_理解与现状备忘.md](32B_理解与现状备忘.md)，技术方案看 [32B强化学习_技术方案.md](32B强化学习_技术方案.md)。

整理日期：2026-06-22。06-13 的目录归位未改训练逻辑；06-22 已进入 V2/derag2 DPO 验收接手阶段，详见 [codex提示词.md](codex提示词.md)。

## 目录结构

```
32b强化学习/
├─ 32B_理解与现状备忘.md       # living-master：现状/坑/决策（留根，code/README.md 有相对链接指它）
├─ 32B强化学习_技术方案.md     # living-master：技术方案（同上）
├─ 32B_实验全历程_从零讲起.md  # ★从零讲懂的实验全历程（按阶段，含每步数字/问题/教训）
├─ PROJECT_INDEX.md            # 本文件
├─ rl_code.zip                 # 【当前】完整 code/ 打包，上传到服务器解压用（打包规矩见下）
├─ last.zip                    # 【上一版】打包备份
├─ code/                       # 唯一活跃源码根（内部结构不动，即 rl_code.zip 的内容）
├─ docs/
│  ├─ journal/                 # 编号分析文档 85–107（纯人读，代码不引用）
│  ├─ research/                # 调研报告（reward_judge_deep_research.md）
│  └─ learning/                # vLLM 入门教学样例
├─ runs/                       # 所有 run 产物（保留原时间戳目录名，run 间靠编号互引、勿改名）
│  ├─ main_pipeline/           # 主流水线产物（原 output/，含 sft_train/rollout 大 jsonl + loss png）
│  ├─ corrected_v2/            # corrected_v2 散落数据（94/95 的 jsonl/json；.md 报告在 journal）
│  ├─ corrected_v3/            # corrected_v3 run（20260611_203429）
│  ├─ corrected_v31/           # corrected_v31 run（20260611_222235）
│  ├─ derag_v4/                # derag_v4 三个 run（20260612_022656/135121/151520）
│  └─ derag_v5_probe/          # ★ headroom 探针结果（20260613_164043，verdict=GO_RL，但见下方"现状"）
└─ archive/
   └─ local_logs/              # 本地旧日志（pipeline.log，含本地路径，无代码依赖）
```

> 残留：根目录可能仍有一个**空的 `derag_v4/`** 目录（整理时被文件句柄锁住没删掉，里面是空的，可随手 `rmdir derag_v4` 清掉）。

## code/ 里的活跃链路 vs 历史复现

- **活跃入口（当前要跑的）**：`scripts/run_v2_dpo_resume.sh`（CPU 合并 DPO adapter → 等两张真正空卡 → 本地 vLLM 冻结 500 题 → 原 Kimi k=3 三分），独立监控 `scripts/monitor_v2_dpo_resume.sh`。它强制复用 `run_v2.py` 原正式叶命名 `v2-2s-2s-2s`，并校验 adapter/base、历史对照、infer/eval provenance。完整命令见 `codex提示词.md` 第三章。
- **V2 数据/训练链**：`run_v2.py` + `pipeline/step_v2_{split,coldstart,rft_select,dpo_pairs,eval}.py` + `v2_{paths,common}.py`；确定性规则唯一来源 `pipeline/rules_v6.py`。本轮 `V2_TAG=derag2`，服务器产物在 `output/derag2/`。
- **AutoDL 重跑入口**：`scripts/run_dpo_autodl.sh`，现为 DPO → transformers 推理，不再启动已知 cu13 错配的 vLLM。
- **上一代探针/训练链（历史复现）**：`run_derag_v5_probe.py`（step150–159）与 `run_derag_v4.py`（SFT→DPO→GRPO）均保留，但不是当前验收入口。
- **历史复现（勿删，按记忆约束保留）**：`run.py`（冷启动→RFT→DPO→GRPO 原始链）、`run_merged_dpo_grpo.py`、
  `run_corrected_v2_mini.py`、`run_corrected_v3.py`、`run_corrected_v31.py` 及各自 `scripts/run_*`/`monitor_*`。
- **核心被复用模块（三链共用，勿删）**：`pipeline/{reward, reward_v3, judge_common, vllm_client, kimi_client, logger, step06_rewrite_seeds}.py` + `config.py`。
  - `reward.py` = swift GRPO 在线 reward；`reward_v3.py` = v4/v5 离线确定性特征。**两者不重复，勿合并。**

## 编号文档导航（docs/journal/）

| # | 文档 | 主题 |
|---|---|---|
| 85–88 | fable5_* | v1 诊断上下文 / delta 统计 / compact 上下文 / v2 诊断与计划 |
| 89–97 | corrected_v2_* | v2 评审、reward 对齐审计、DPO 对 Kimi 审计、pair 复判、mini DPO 复盘 |
| 98 | corrected_v3_postmortem | corrected_v3 复盘与计划 |
| 99 | rl_route_verdict_and_plan_a | 方向裁决（已被用户约束覆盖：RL 必须保留） |
| 100–103 | derag_v4 系列 | v4 强制 DPO/GRPO 计划 → 最终蓝图 → stage1 门禁重构 → 二值门 |
| 104–106 | 奖励函数 | 业内调研 → 落地执行方案 → 两套方案 fable5 裁决稿 |
| 107 | reward_v5_裁决与融合方案 | **当前奖励函数终裁**（headroom 探针的设计依据） |

## 关键约束（红线，勿动）

- **GRPO colocate 显存五连关**：`swift/grpo.sh`、`swift/grpo_on_model.sh` 的
  `vllm_gpu_memory_utilization=0.5` + `offload_model/optimizer=true` + `vllm_tensor_parallel_size=8(=NPROC)`
  + `move_model_batches=16` + `sleep_level=1`。**调高 util 会 OOM**，本次整理已两路核验这些值未被改动。
- 推理 serve（`scripts/serve_v1_vllm.sh`）用 `VLLM_SERVE_GPU_UTIL=0.90`，是独立变量、只服务推理、不与训练抢卡。

## 现状（一句话）

V2/derag2 已完成 SFT→RFT→DPO 训练；DPO adapter=`checkpoint-108` 与 RFT 合并底座均已在本地服务器审计确认。当前唯一主任务是运行 `run_v2_dpo_resume.sh`，产出原链 `output/derag2/v2-2s-2s-2s_{infer,scores,report,summary}.*`。8 张卡仍被同事占用时脚本会先 CPU 合并再安全等待，不会抢卡。
