# Codex 接力提示词 · 32B 强化学习「think 去 RAG 化」项目

> 你是接手这个项目继续工作的 codex。请先完整读这份文档，再动手。
> 这份是**自包含**的：读完它你应该不需要追问就能接着干。落盘时间 2026-06-22（经 4-agent 对源码核验后修订）。
> 配套深读：同目录 `32B_理解与现状备忘.md`（含全部心路历程，最新章节在顶部）、`32B强化学习_技术方案V2.md`、`PROJECT_INDEX.md`。

> **2026-06-22 17:09 DPO正式评测完成（优先级最高）**：本地正式链已完成 DPO merged → 冻结500题推理 → 格式失败记账 → Kimi k=3 三分，产物均有 provenance。结果：V1 baseline `3.140 / 2.6% / 93.8%`，SFT `4.408 / 45.4% / 83.0%`，RFT `4.489 / 45.6% / 85.0%`，DPO `4.643 / 49.0% / 84.6% / 格式99.8%`。DPO 相对 RFT 的 clean 增量 `+0.155`，略高于 `3×SE≈0.149`，判 **DPO_TRUE_GAIN**；但答案在池率低于 0.85，最终三件套 **FAIL**。第437题真实 max-token 截断已诚实计入500分母：partial think照常Kimi，规则think失败，空answer不在池，不重采样、不补V1/gold。用户明确：不再围绕该边界补跑，下一步进入 **GRPO**，目标是保住 DPO 的 clean/rule gain，同时用答案在池/格式/规则think奖励把在池率拉回 ≥0.85。最新完整包 `rl_code.zip` 3,636,897 bytes，SHA256=`03CFB07B17F47484181E2D4EE0B9B38646D8BC7BC92DB3985B3C0BEA8A3191C4`。

> **2026-06-22 夜 GRPO代码已落地（当前最高优先级）**：新增 V2 online GRPO 续跑器，正式入口 `scripts/run_v2_grpo_online.sh`，监控 `scripts/monitor_v2_grpo_online.sh`，主控 `run_v2_grpo_online.py`。训练不新造框架，仍复用旧跑通 `swift/grpo_on_model.sh` + `/home/nvme02/conda/grpo_env` + colocate vLLM（TP=8、util=0.5、offload、move_model_batches=16）。reward 插件新增 `v2_rule_warmup` 与 `v2_online`：格式/空answer硬负分，`answer_in_v1_pool` 不可被 Kimi 补偿，`detect_rag_style` 做规则think档位，Kimi think 默认 k=2 只做组内排序。默认产物不覆盖 DPO：`v2-grpo-warmup-2s-2s-2s-*`、`v2-grpo-2sigma-2s-2s-2s-*`，评测 tag=`v2-2s-2s-2s-grpo`。22:39 服务器首跑暴露 `swift.grpo_reward_plugin` 与已安装 ms-swift 包名冲突，已 hotfix：reward audit 与测试改为按文件路径加载 `code/swift/grpo_reward_plugin.py`，不新增 `swift/__init__.py`，避免影响 ms-swift。22:43 第二次首跑暴露旧 GRPO 数据含不可训练 answer pool（无可比较事实），已 hotfix：builder 过滤该类题，runner 发现旧数据 schema 会自动改名重建。当前本地测试 `23/23` 通过，`py_compile` 覆盖 GRPO runner/data/audit/reward 插件。

> **2026-06-22 Codex 最终接回更新（覆盖本文更早的 AutoDL/HF 手工方案）**：服务器已审计确认 RFT 底座在 `/home/nvme01/zhjg/models/v2-rft-2sigma-2s-merged`（14 分片，index total_size=65527752704），DPO adapter 在 `/home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108`。正式旧链命名只能是 LoRA `v2-dpo-2sigma-2s-2s-lora`、合并模型 `v2-dpo-2sigma-2s-2s-merged`、tag `v2-2s-2s-2s`、served name `v2_2s_2s_2s`，评测四件套只能写入 `output/derag2/v2-2s-2s-2s_*`。新唯一入口是 `scripts/run_v2_dpo_resume.sh`，监控是 `scripts/monitor_v2_dpo_resume.sh`；它会 CPU 合并、等两张连续空闲且无计算 PID 的卡、用本地已验证 vLLM 环境推理冻结 500 题，再按原 Kimi k=3 三分口径评测。旧的 `derag2_dpo*` 命名和手工 `infer_hf` 方案均不再用于正式验收。该时点完整包 SHA256=`0936C54D110464BB6BBA4D6FFF51268CBDEAFBEB8A67AF48D612C56081D2F45F`，已由上方新包替代。

---

## 一、一句话项目本质

用强化学习把公司 **V1 税务大模型**（Qwen2.5-32B 微调）的 `<think>` 推理过程**「去 RAG 化」**——V1 对外宣称端到端推理，实际 think 里满是"参考问答对N / 资料显示 / 检索结果"的检索腔，像机器不像人。**本项目验证：能否用 RL 在不损答案准确率的前提下让 think 自然化，成功则把配方迁回 V1。** 公司 V2 已训崩，这是独立验证线。

### 成功标准（三件套，缺一不可）
1. **Kimi 干净分（0-10）** 从 V1 基线明显上涨（基线数以服务器 `output/derag2/v2-baseline-v1_summary.json` 的 `clean_mean` 为准——见第三章注）；
2. **规则去检索腔通过率**（`detect_rag_style`）从 V1 近 0% 明显上涨；
3. **答案在池率**（`answer_in_v1_pool`）守住 **≥ 0.85**。
   - ⚠️ 注意 **0.85 是最终验收线**；正文会看到 SFT 阶段实测只有 83%、靠 `V2_PRUNE_INPOOL_FLOOR=0.80` 这条**放行线**先进 RFT（指望 RFT 的答案门把 grounding 捞回 ≥0.85）。**DPO 评测要对的是 0.85，不是 0.80。**

### 核心分工铁律（全程不变）
| 目标 | 工具 | 备注 |
|---|---|---|
| think 检索腔**表面词** | 规则 `detect_rag_style` | 确定性、零噪声；定义在 `pipeline/rules_v6.py` |
| 答案有没有**漂离 V1** | 规则 `answer_in_v1_pool` | 确定性、零噪声；定义在 `pipeline/rules_v6.py` |
| think **照抄（换词复述）** | **只能靠 Kimi** | 语义判断；规则对换词复述全盲 |

---

## 二、当前确凿状态（你接手时的现实）

整条链已跑到 **DPO 正式评测完成，GRPO 代码已落地待服务器实跑**。链路：
`SFT 突破(k2-930+ep7, Kimi 干净分 7.11) → RFT(2σ 桶) → 合并底座 v2-rft-2sigma-2s-merged → DPO(885 条 2σ 大间距对子) → DPO merged 500题三分评测完成 → V2 online GRPO 待跑`

**DPO 这一轮在 AutoDL 2×A800 上跑完**（本地 8 卡当时被占）。产出：

| 东西 | 位置 | 状态 |
|---|---|---|
| DPO adapter（DPO 全部成果）| 本地 `/home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108/` | ✅ 已下载（`adapter_model.safetensors` 257M + `adapter_config.json`）|
| RFT 合并底座 `v2-rft-2sigma-2s-merged`（推理底座，约 62-65G / 14 shards）| `/home/nvme01/zhjg/models/v2-rft-2sigma-2s-merged` | ✅ 已审计 |
| DPO 评测推理（**500 验收集** infer）| `/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_infer.jsonl` | ✅ 500/500，含第437题格式失败诚实记账 |
| DPO 三分报告 | `/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_report.md`；汇总 `/home/nvme01/zhjg/logs/v2_dpo_resume/result.md` | ✅ 完成：`4.643 / 49.0% / 84.6%`，DPO_TRUE_GAIN 但最终 FAIL |
| GRPO 代码入口 | `scripts/run_v2_grpo_online.sh` + `scripts/monitor_v2_grpo_online.sh` | ✅ 已落地，待本地服务器实跑 |
| AutoDL 实例 | `ssh -p 20007 root@connect.nma1.seetacloud.com` | 可能已被关机；要用得先开 |

> AutoDL 的 vLLM 环境确实不可用，但本地服务器原链 vLLM 环境 `/home/nvme02/biyh/vllm_env` 已验证可用。正式接回必须走本地合并模型 + 原 vLLM 推理链；`infer_hf.py` 只保留为故障兜底，不作为本次正式结果来源。

### ⚠️ 两个最容易踩的口径（我已替你核对源码，照这个来）
- **评测集是 500 条**（`00_data_v2_eval.jsonl`），不是 224。224 是旧 V1 管线的 `00_data_sft_eval.jsonl`，与本链无关。依据：`code/pipeline/v2_paths.py:20 V2_N_EVAL=500`、`:25 # 500：全程冻结`。
- **derag2 这轮的数据/产物都在 `output/derag2/`**，不是 `output/v2/`。依据：`V2_OUTPUT_DIR = OUTPUT_DIR/$V2_TAG`，本轮 `V2_TAG=derag2`。所以下面命令统一 `export V2_TAG=derag2` + 走 `output/derag2/`。

---

## 三、你的立即任务：进入 GRPO（DPO 评测已完成）

目标：以 DPO merged 为当前 policy 起点，做 GRPO，把 DPO 已证明有效的 clean/rule 增益保住，同时把答案在池率从 84.6% 拉回 **≥0.85**。

当前 DPO 的正式 infer/scores/report/provenance 均已完成，不要再跑 `run_v2_dpo_resume.sh` 当主任务。下一步是直接跑 V2 online GRPO。

### ★ 正式执行入口（以下命令优先级最高）

先把新的完整 `rl_code.zip` 上传到 `/mnt/pfs/zhjg/` 并覆盖解压到原 `code/`。GRPO 训练会用 8 卡 colocate vLLM，若卡被占应先等空；脚本不会抢占或杀别人的进程。

主终端：

```bash
cd /mnt/pfs/zhjg/code
bash scripts/run_v2_grpo_online.sh
```

监控终端：

```bash
cd /mnt/pfs/zhjg/code
bash scripts/monitor_v2_grpo_online.sh
```

GRPO 脚本流程：构建 train-only GRPO 数据 → `v2_rule_warmup` 短 warmup → CPU 合并 warmup → `v2_online` Kimi-k2 GRPO → CPU 合并 final → 冻结500题 V2 三分评测。最终产物固定为：

```text
/home/nvme01/zhjg/ckpts/v2-grpo-warmup-2s-2s-2s-lora
/home/nvme01/zhjg/models/v2-grpo-warmup-2s-2s-2s-merged
/home/nvme01/zhjg/ckpts/v2-grpo-2sigma-2s-2s-2s-lora
/home/nvme01/zhjg/models/v2-grpo-2sigma-2s-2s-2s-merged
/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s-grpo_infer.jsonl
/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s-grpo_scores.jsonl
/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s-grpo_report.md
/home/nvme01/zhjg/output/derag2/v2-2s-2s-2s-grpo_summary.json
```

下方 Step 0–2 是早期手工/HF 故障兜底记录，**不要用它生成本次正式验收结果**；只有新续跑器明确报出无法恢复的环境故障时才用于诊断。

### Step 0 — 定环境 + 确认两样东西齐了
```bash
cd /mnt/pfs/zhjg                                  # 代码根（含 code/）；若代码在别处，cd 到含 code/ 的父目录
export PATH=/home/nvme02/conda/zhjg_rl/bin:$PATH  # 激活训练栈（torch/transformers/peft/accelerate 齐）。
                                                  # ⚠️ 别用 `source activate`：项目已踩过坑，set -u 脚本里 CONDA_PREFIX unbound 会崩。
export ZHJG_WORK_DIR=/home/nvme01/zhjg            # 否则 config 默认旧路径会乱建目录
export V2_TAG=derag2                              # ★关键：让 step_v2_eval 的默认 support 等路径落到 output/derag2/
export OMP_NUM_THREADS=8                          # 消 libgomp 警告（无害但烦）

# ① adapter（已在手，确认一下）
ls -lh /home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108/adapter_model.safetensors   # 期望 ~257M
ls -lh /home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108/adapter_config.json

# ② 底座 rft_merged 已确认
ls -lh /home/nvme01/zhjg/models/v2-rft-2sigma-2s-merged/model.safetensors.index.json
#   认准本轮底座名 = v2-rft-2sigma-2s-merged，里面含 config.json + 14 个 *.safetensors。
#   （别误用旧的 v1-32b-cs-rft-lora / corrected-v1-*rft* 之类历史目录——口径会错配。）

# ③ 评测集存在性（500 条冻结集）
wc -l /home/nvme01/zhjg/output/derag2/00_data_v2_eval.jsonl   # 期望 500
```

### Step 1 — 推理（二选一，看哪台有空闲显卡）

> 32B + LoRA 推理需要 ~2 张 A800（`device_map=auto` 自动切）。先 `nvidia-smi` 看空卡。

**分支 A：本地服务器（首选，环境已激活）**
```bash
mkdir -p /home/nvme01/zhjg/runs /home/nvme01/zhjg/logs   # 日志目录仍需建；infer_hf.py 已会自动建 --out 父目录
export CUDA_VISIBLE_DEVICES=<两张空闲卡，如 0,1>          # 从 nvidia-smi 选 memory.free 最大的两张

# 先 3 条小样冒烟（确认能加载+生成，再跑全量，省得白等）
head -3 /home/nvme01/zhjg/output/derag2/00_data_v2_eval.jsonl > /tmp/smoke3.jsonl
python -X utf8 code/pipeline/infer_hf.py \
  --base    <BASE> \
  --adapter /home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108 \
  --eval_file /tmp/smoke3.jsonl \
  --out     /tmp/smoke3_out.jsonl
# 看到 "[infer] done -> ..." 且无报错 → 冒烟过，跑全量：

nohup python -X utf8 code/pipeline/infer_hf.py \
  --base    <BASE> \
  --adapter /home/nvme01/zhjg/ckpts/v2-dpo-derag2-lora/checkpoint-108 \
  --eval_file /home/nvme01/zhjg/output/derag2/00_data_v2_eval.jsonl \
  --out     /home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_infer.jsonl \
  > /home/nvme01/zhjg/logs/infer_hf.log 2>&1 &
tail -f /home/nvme01/zhjg/logs/infer_hf.log    # 500 条贪心，约 1-2 小时
# infer_hf.py 默认按 query 断点续跑；中断后原命令重跑即可。只有确认要清空结果时才加 --overwrite。
```

**分支 B：AutoDL（本地无空闲显卡时）**——别碰那台 vLLM，照样用 `infer_hf.py`：
```bash
# AutoDL 上激活训练栈（注意是 /root/envs/zhjg_rl，不是本地的 /home/nvme02/conda）
export PATH=/root/envs/zhjg_rl/bin:$PATH      # 同样别 source activate
export ZHJG_WORK_DIR=/root/autodl-tmp/dpo
export V2_TAG=derag2
export OMP_NUM_THREADS=8
mkdir -p /root/autodl-tmp/dpo/output
# 底座在 /root/autodl-tmp/dpo/rft_merged；adapter 在 out_dpo/v0-20260620-183006/checkpoint-108；
# 评测集若不在则从本地传 → /root/autodl-tmp/dpo/pkg/00_data_v2_eval.jsonl
cd /root/autodl-tmp/dpo   # 含 code/ 的父目录
python -X utf8 code/pipeline/infer_hf.py \
  --base    /root/autodl-tmp/dpo/rft_merged \
  --adapter /root/autodl-tmp/dpo/out_dpo/v0-20260620-183006/checkpoint-108 \
  --eval_file /root/autodl-tmp/dpo/pkg/00_data_v2_eval.jsonl \
  --out     /root/autodl-tmp/dpo/output/v2-dpo-derag2_infer.jsonl
# 跑完若仅作故障恢复，必须先经过新续跑器的严格 infer/provenance 校验，不能直接冒充正式结果
```

### Step 2 — 三分评测（本地，Kimi judge）
```bash
cd /mnt/pfs/zhjg
export PATH=/home/nvme02/conda/zhjg_rl/bin:$PATH
export ZHJG_WORK_DIR=/home/nvme01/zhjg
export V2_TAG=derag2
export DASHSCOPE_API_KEY=<轮换后的新 key>
#   注：即使忘了 export，config.py:53-54 会落到明文兜底 key（已泄露，见第六章④），Kimi 不会因缺 key 报错——
#       但务必用轮换后的新 key，别继续用泄露的旧 key。

python code/pipeline/step_v2_eval.py \
  --infer   /home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_infer.jsonl \
  --scores  /home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_scores.jsonl \
  --report  /home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_report.md \
  --summary /home/nvme01/zhjg/output/derag2/v2-2s-2s-2s_summary.json \
  --support /home/nvme01/zhjg/output/derag2/152_v1_support.v2.jsonl \
  --tag     v2-2s-2s-2s
#   --support 显式指到 derag2（默认值也依赖 V2_TAG，已 export 但显式更稳）；它是答案在池率的 V1 池。
```

### Step 3 — 读报告，对三件套
报告 `output/derag2/v2-2s-2s-2s_report.md` 固定就三行带 marker 的指标（`step_v2_eval.py` 直接打印）：
- **Kimi干净分(k=3)均值**——报告自己按真实 N 打印 `SE≈x，两阶段差 >~3×SE 才算真涨`（N=500 时 3×SE 约 **0.15**；别用任何 224 推出的 0.22）；
- **规则去检索腔通过率**（`detect_rag_style` 无痕迹占比）；
- **答案在池率**（含漂移率）。
- **答案可比较审计**（附加披露，不替代三件套主指标）：无极性/数字/日期的非空回答规则无法判断，单列数量与可比较题在池率；空答案直接判失败。

**达标判定（三件套全满足才算 DPO 真过）**：
1. Kimi 干净分相对 baseline 报告 **涨过 3×SE（≈0.15）**；
2. 规则去检索腔通过率 **≥ SFT 阶段的 45%**（最好更高）；
3. 答案在池率 **≥ 0.85**。

**填这张基线对照表**（数值从各 `output/derag2/*_summary.json` 取，跑完 DPO 填最后一行）：

| 阶段 | Kimi干净分(k=3) | 规则去检索腔通过率 | 答案在池率 | 对应 summary |
|---|---|---|---|---|
| V1 baseline | ？ | ~0% | ~1.0（基线） | `v2-baseline-v1_summary.json` |
| SFT-2s | ~4.4* | ~45%* | ~0.83* | `v2-sft-2s_summary.json` |
| RFT-2s-2s | ？ | ？ | ？（应回到 ≥0.85）| `v2-rft-2s-2s_summary.json` |
| **DPO（你跑的）** | ？ | ？ | ？ | `v2-2s-2s-2s_summary.json` |

> *SFT 那几个数来自历史记述，**以服务器上实际 summary 为准**先 `cat` 核对再用。备忘里另有「kimi_think 3.1→4.4」是**不同口径**的 Kimi 分（不是本表的 k=3 整段干净分），别和本表的 clean_mean 混。

---

## 四、关键文件地图（你会用到的）

| 文件 | 作用 | 关键点 |
|---|---|---|
| `code/pipeline/infer_hf.py` | **transformers 兜底推理**（新增，vLLM 不可用时用）| `--base`+`--adapter`+`--eval_file`+`--out`；默认断点续跑并做完整性检查；`--model_name` 默认 `v2-dpo-derag2`（非 v1）|
| `code/pipeline/step03_eval_infer.py` | vLLM 路径的推理（仅当 vLLM 环境健康时用）| 内部走 `vllm_client`；当前各机 vLLM 都不可用，暂别用 |
| `code/pipeline/step_v2_eval.py` | **三分评测**（Kimi干净分/规则/在池率）| 吃 step03 或 infer_hf 的输出；参数 `--infer/--scores/--report/--summary/--support/--tag` |
| `code/pipeline/v2_paths.py` | V2 路径/常量 | `V2_N_EVAL=500`、`V2_EVAL=00_data_v2_eval.jsonl`、`V2_V1_SUPPORT=152_v1_support.v2.jsonl`、`dpo_pairs()` 等 |
| `code/pipeline/v2_common.py` | V2 三分核心 | `V2_OUTPUT_DIR=OUTPUT_DIR/$V2_TAG`、`score_think_eval/score_think_rule/answer_drift`、`confident_cleaner()`（第 49-58 行；**默认 n_sigma=3.0**，选样时传 `2.0` 得 2σ 桶）|
| `code/pipeline/rules_v6.py` | 确定性规则定义处 | `detect_rag_style`、`answer_in_v1_pool` 真正定义在这（v2_common/reward 只 import 使用）|
| `code/pipeline/reward.py` | 打分器 | `parse_think_answer(text)->(think,answer)`（`:62`，infer_hf 复用它）|
| `code/config.py` | 集中配置 | `system_for(name)`（`:85`）：`v1`→RAG腔，其他→去检索腔；`resolve_adapter(root)`（`:90`）找最优 ckpt；`GEN_MAX_NEW_TOKENS=1536`（`:158`）；明文 key（`:54`）；`WORK_DIR` 默认旧路径，**务必 export ZHJG_WORK_DIR** |
| `code/swift/dpo_v2.sh` | DPO 训练（swift rlhf dpo）| `<pairs> <base> <out>`；`per_device_train_batch_size 1` 硬写在 `:69` |
| `code/scripts/run_dpo_autodl.sh` | **AutoDL DPO + HF 推理全链** | 已移除 cu13 vLLM 路径；训练后直接调用 `infer_hf.py`，末尾评测路径已修正为 `output/derag2/` |
| `output/derag2/dpo_pairs.2s-2s_2sigma.v2.jsonl` | DPO 对子（885 条）| answer-lock、大间距；3σ 桶版 478 条 |
| `output/derag2/00_data_v2_eval.jsonl` | **500 验收集** | 全程冻结 |
| `output/derag2/152_v1_support.v2.jsonl` | V1 答案池 | 算"在池率"用 |

### `config.system_for` 的坑（最易错，重申）
```python
system_for("v1")  → SYSTEM_PROMPT（RAG 腔："基于参考问答对搜索…"）
system_for(其他)  → COLDSTART_SYSTEM_PROMPT（去检索腔："像资深税务老师，依据参考但别复述编号…"）
```
**推理 DPO 模型时 `--model_name` 必须是 `v2-dpo-derag2`（非 v1）**，否则套 RAG 腔 prompt，模型行为和评测口径全错。`infer_hf.py` 默认值已是 `v2-dpo-derag2`，别改成 v1。

---

## 五、已知坑全记录（每条都验真过，别重新踩）

### vLLM cu13 死结（AutoDL 上，已放弃 vLLM）
- 现象：起 vLLM 报 `ImportError: libcudart.so.13: cannot open shared object file`。
- 根因：`vllm 0.20.1` 的 `_C.abi3.so` 是 **CUDA 13 编译**（依赖 `libcudart.so.13`），但配套 `torch 2.11.0+cu128` 自带 `libcudart.so.12` —— **错配安装**。
- driver 其实支持（`nvidia-smi`=Driver 590 / CUDA 13.1），缺的只是 `libcudart.so.13` 文件本身。
- **软链 so.12→so.13 不行**：ELF SONAME 写死，报 `version not found`。
- **pip 装 nvidia-cuda-runtime-cu13 不行**：AutoDL pip 不认 cu13 manylinux wheel，退 build sdist 失败（默认源 + 官方 PyPI 源都试过）。
- **结论**：评测推理一律走 `infer_hf.py`（transformers），不要再修任何机器的 vLLM。若哪天必须修：换一个真正 cu12 build 的 vllm，或换能拿到 cu13 wheel 的环境。

### AutoDL DPO 训练的 env 坑（若要重跑 DPO，`run_dpo_autodl.sh` 已全部规避）
| 坑 | 解决 |
|---|---|
| python3.11/swift exit127 | `export PATH="$ZHJG_ENV/bin:$PATH"` **第一条**，不 source activate（set -u 下 CONDA_PREFIX unbound 崩）|
| 8 卡 OOM | `TRAIN_GPUS=0,1` |
| 有效 batch 漂移 / GA 自动降 1 | `DPO_AUTO_GA=0` + `DPO_GA=8`（2卡×1×8=16，等效原 8 卡×GA2）|
| vllm/log/work 旧路径 | `VLLM_ENV`/`ZHJG_LOG_DIR`/`ZHJG_WORK_DIR` 全指 `/root/autodl-tmp/dpo` 下 |
| rpo_alpha 假警报 | swift 4.0.1 支持，保留 `DPO_RPO_ALPHA=1.0` |
| eval 套 RAG 腔 | `--model v2-dpo-derag2`（非 v1）|

### 环境速查
| 环境 | 路径 | 关键 |
|---|---|---|
| 训练栈（本地）| `/home/nvme02/conda/zhjg_rl`（py3.11）| torch 2.10 / transformers 5.2.0 / peft 0.18.1 / accelerate 1.13.0 / ms-swift 4.0.1；**无 vLLM（设计如此）**。激活 `export PATH=.../bin:$PATH` |
| 训练栈（AutoDL）| `/root/envs/zhjg_rl`（py3.11）| 同上；激活同样 `export PATH=/root/envs/zhjg_rl/bin:$PATH` |
| 评测 vLLM（本地）| `/home/nvme02/biyh/vllm_env`（py3.12）| vllm 0.20.1+cu129（本地这份可用；AutoDL `/root/envs/vllm_env` 那份 cu13 错配，别用）|
- 路径分工：代码 `/mnt/pfs/zhjg/code`；产物 `/home/nvme01/zhjg`（`output/$V2_TAG/` JSONL+报告、`ckpts/` LoRA、`models/` 合并全量、`runs/` 下载/产出的 infer）。`/mnt/pfs` 空间紧，别写大模型。

---

## 六、待办与待决策（接手后优先处理）

1. **❓ 确认底座 `v2-rft-2sigma-2s-merged` 本地位置**：`find /home/nvme01 /mnt/pfs -maxdepth 4 -iname "*rft*merged*" -type d`。
   - **找到**（含 config.json + 14 个 *.safetensors）→ 记为 `<BASE>`，直接用。
   - **没找到** → 二选一：
     - (a) 从 AutoDL 下回：`/root/autodl-tmp/dpo/rft_merged`（约 62-65G，慢）；
     - (b) **本地用 V1 + RFT adapter 重新合并**（走 CPU，32B bf16 约需 ~65G 内存）：
       ```bash
       # 先确认本轮 RFT adapter（认准 v2-rft-2sigma-2s，不是旧 v1-32b-cs-rft-lora / corrected-v1-*）
       ls -d /home/nvme01/zhjg/ckpts/*rft*2sigma*2s* 2>/dev/null
       # 用 config.resolve_adapter 取其最优 ckpt，再 PEFT 合并：
       export PATH=/home/nvme02/conda/zhjg_rl/bin:$PATH
       export ZHJG_WORK_DIR=/home/nvme01/zhjg
       python -X utf8 - <<'PY'
       import sys, torch; sys.path.insert(0, 'code')
       from config import resolve_adapter, V1_DIR
       from transformers import AutoModelForCausalLM, AutoTokenizer
       from peft import PeftModel
       RFT_ROOT = "/home/nvme01/zhjg/ckpts/<填上面 ls 找到的 v2-rft-2sigma-2s 目录>"
       OUT = "/home/nvme01/zhjg/models/v2-rft-2sigma-2s-merged"
       adp = resolve_adapter(RFT_ROOT); print("adapter:", adp)
       base = AutoModelForCausalLM.from_pretrained(V1_DIR, torch_dtype=torch.bfloat16, device_map="cpu")
       m = PeftModel.from_pretrained(base, adp).merge_and_unload()
       m.save_pretrained(OUT, safe_serialization=True)
       AutoTokenizer.from_pretrained(V1_DIR).save_pretrained(OUT)
       print("merged ->", OUT)
       PY
       ```
       合并出的 `OUT` 就是 `<BASE>`。（参考备忘 2026-06-11 节的历史 `merge_v1_rft` 做法，同理。）
2. **⏳ 跑完 DPO 评测**（第三章），出 `v2-2s-2s-2s_report.md`，对三件套、填基线对照表。
3. **⏳ AutoDL 善后**：DPO 成果（adapter）已落本地，AutoDL 可关。若重开跑推理，走 `infer_hf.py` 路线（第三章分支 B），别碰 vLLM。
4. **⚠️ 安全**：DashScope key `sk-REDACTED-ROTATE-ME` 明文在 `config.py:54` + 多个 .sh，已上传公网机=泄露，**建议轮换**；轮换后改 config 默认值或统一用 `export DASHSCOPE_API_KEY`。
5. **下一步训练（DPO 之后）**：若三件套达标且答案守住，按技术方案 V2 走 **本地 reward GRPO**（三阶段分账的最后一段）；若 DPO 无增益，先回看 memory `corrected-v2-mini-failure-judge-noise`（裁判噪声 + rubric 天花板）——**先算 σ/MDE 再决定要不要训**，别被噪声淹没了又空跑一轮。

---

## 七、打包规矩（改了 code 之后）
- `rl_code.zip` = 全量 `code/` 树（`code/` 前缀、含 `data/`、排 `__pycache__`/`.pyc`）；本地只留上一版 `last.zip`；`rl_code_v1.zip` 冻结不动。
- 本次新增的 `code/pipeline/infer_hf.py` 和 `code/scripts/run_dpo_autodl.sh` 都已在本地 `code/` 树里，下次打包自动带上。

---

## 八、给简历/面试的沉淀（用户最终交付物之一）
- **裁判即仪器**：连续分裁判是 prompt 的函数不是测量；judgecal 先标定 Kimi 分辨率，才敢用它构 DPO 大间距对子（chosen≈7 vs rejected≈1-2，中间档分不开不用）。
- **测量先于优化**：σ/MDE/设计效应/headroom 开训前算清，避免空跑（corrected-v2 的 0.006<0.022 本可零成本拦下）。
- **诚实记账**：分阶段相对自己起点 paired Δ，蒸馏/RL 分账；DPO≈中性也照实写。
- **环境工程**：本次 AutoDL DPO——逐层定位 vLLM cu13 错配根因（ELF SONAME / wheel tag / driver-runtime 版本），拿不到正确 wheel 时果断切 transformers 兜底，且**先保住 adapter 再折腾推理**。

> 接手后如有任何"文档说的和现实对不上"，以**现实为准**并回填这两份文档（备忘 + 本提示词）。祝顺利。
