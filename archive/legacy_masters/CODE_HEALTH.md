# 代码体检报告（2026-06-13）

三路并行审查（pipeline 代码审查 / 编排+swift+显存配置 / 全项目盘点）的汇总。
方法：实跑 `py_compile`、stub 掉 `requests` 后全链 import、pyflakes、`bash -n`、逐函数签名核对。

## 结论速览

- **活跃 v5 探针链健康**：14 个核心文件全部 `py_compile` 通过；所有跨模块函数签名对得上
  （`vllm_client.gen_one/gen_k/map_concurrent`、`judge_common.judge_text_derag`、
  `reward_v3.{trace_counts,nums,parse_think_answer,extract_references,_POLARITY}`、
  `step06.{REWRITE_SYSTEM,REWRITE_TEMPLATE}`、`kimi_client.chat`）；运行期逻辑实测正确，**无会炸的 bug**。
- **28 个 shell 全部 `bash -n` 通过。**
- **显存配置经两路独立核验：未被改高**（详见下）。

## 显存配置核验（用户重点关切）✅ 未改动

| 文件 | 配置 | 状态 |
|---|---|---|
| `swift/grpo.sh:49` | util `${VLLM_GPU_UTIL:-0.5}` + TP=`${VLLM_TP:-$NPROC}`(8) + offload_model/optimizer=true + move_model_batches 16 + sleep_level 1 | ✅ 默认 0.5，符合血泪约束 |
| `swift/grpo_on_model.sh:50-55` | 同上 + DEEPSPEED 默认 zero3（derag_v4 走这条） | ✅ 未改高 |
| `scripts/serve_v1_vllm.sh:18` | `VLLM_SERVE_GPU_UTIL:-0.90`，仅推理、独立变量名、TP=2 占 GPU0,1 | ✅ 不与训练抢卡 |

**本次整理 + 今天的代码改动都没有碰任何训练脚本/显存参数。** 改的只有 v5 探针的 3 个文件（见下"本次已修"）。

## 本次已修（活跃链，零行为风险）

1. **`pipeline/v5_probe_common.py` — verbatim_copy 恒为 0 的隐患**（真实逻辑问题）。
   `real_trace` 原先从 `reward_v3.trace_counts` 取 `verbatim_copy`，但 v4 版该函数把它**硬编码为 0**
   （只有 frozen 版才返回 `int(copy_ratio>=0.40)`），导致"纯照抄但无触发词"的 think 在 `--detect rule`
   模式下被判为非病题。已改成 `real_trace` 自己按 `copy_ratio>=VERBATIM_COPY_MIN(0.40)` 计 verbatim，
   并修正了误导性注释。**默认 kimi 模式不受影响**（靠裁判挑题），此修让 rule 备选模式也诚实。
2. **`run_derag_v5_probe.py` — 删 `stop_vllm` 里未用的 `model = _SERVED["dir"]`**（改 kill_gpu_procs 后的残留）。
3. **`pipeline/step159_probe_report.py` — 删未用的 `a_rw` 绑定**（表格直接用 `rw.get(...)`）。

三个文件改后 `py_compile` 通过。**因此 `rl_code.zip` 已重新打包**（旧版进 `last.zip`）。

## 剩余可清理项（低优先，零行为影响，留作记录）

pyflakes 报的未用 import/变量，都在**历史复现脚本**里，不影响运行，为减少对复现记录的扰动**未动**：

- `pipeline/step04_judge.py:17` — `JUDGE_SYSTEM/JUDGE_TEMPLATE/parse_judge_json` 三个未用导入
- `pipeline/step100_judge_noise_calibration.py:21` — `v3_utils.sd` 未用
- `pipeline/step101_paired_eval_stats.py:11-12` — `config.OUTPUT_DIR`、`v3_utils.acc_tier` 未用
- `pipeline/step126_v4_report.py:73` — 死变量 `prev_label`
- `run.py:32` — `config.GRPO_LORA_DIR` 未用导入

## 结构性观察（需谨慎，按记忆约束不建议改）

- **重复工具函数**：`qid` 生成有三处（`v3_utils.qid_for` 吃 dict / `step150.qid_of` 吃 str / `step93` 的 `sha1[:n]`）；
  数字/事实抽取有三套（`reward._facts` / `reward_v3.nums` / `step06._nums`，且 `reward` 的极性表缺"应当"）。
  属 legacy/v3/v4 历史分层沉积，活跃链只依赖 `reward_v3.{nums,facts}` + `reward.copy_signal`，**收敛需谨慎别改历史 run 行为**。
- **编号撞名**：`step124_build_dpo_pairs_v4.py` vs `step124_rewrite_residual.py`；`step126_dpo_seed_pools.py` vs `step126_v4_report.py`。
  均被 `run_derag_v4.py` 正常引用、非死代码；改名要同步改引用，属 derag_v4 复现记录，**勿删**。
- **孤立但保留**：`pipeline/pmi_scorer.py` 只被 `step11/step12` import（PMI 验证过但没部署、s_pmi 全 null），属 RFT/DPO 复现链，**保留**。
- **环境默认不一致（非 bug）**：`grpo.sh:25` 默认回退 `zhjg_rl`，`grpo_on_model.sh:12` 默认 `grpo_env`；
  活跃链路都显式 `export GRPO_ENV=grpo_env` 覆盖，实跑不受影响。建议裸跑 `grpo.sh` 前确认走的是装了匹配 vLLM 的 `grpo_env`。
- **serve 0.90 vs 评测 0.88（非 bug）**：`run_merged_dpo_grpo.sh`/`run_grpo_rft_*.sh` 的 `VLLM_SERVE_GPU_UTIL` 默认 0.88
  是评测期给 CUDA graph 留余量的有意降值，非被改动。
- **dpo.sh GPU2-7 vs dpo_v2/dpo_on_model GPU0-7（非 bug）**：对应不同链路的资源分工（原始链假设 vLLM 占 0,1 同时在跑 / merged 链先停 vLLM 再用全卡），各自自洽。

## 加固建议（可选，非必须）

- `run_derag_v5_probe.py:181-188` 的 `wait_gpu_free`（阈值 `GPU_FREE_MIB=4000`）假设 GPU0,1 专用：
  若那两张卡上有 >4GB 常驻作业会等满超时再强起、可能抢卡 OOM。建议注释里点明"需 GPU0,1 空闲"作为前置条件。
- 在 `code/` 下补一份 RUNBOOK 列"活跃入口 + 历史链路对照表（哪个 run_*.py 用哪条 swift 壳、GPU 分工、serve 0.90 vs 评测 0.88）"，减少后续误判。
