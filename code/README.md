# code/ — 32B 税务模型 think 去检索腔 · 最终链路（V2/derag2）

把公司 **V1**（Qwen2.5-32B 微调的税务模型）的 `<think>` 推理过程，从"念手册/查 RAG 的检索腔"
改造成"像人一步步推"，同时锁死 `<answer>` 不漂离 V1 自己认可的答案。手段是**纯自我改进**的四阶段：

```
V1(只读) ──冷启动 SFT──► RFT 自采样 ──► DPO 偏好对 ──► 在线 GRPO
          (学去检索腔)   (修grounding)   (主升力)      (强化/守地板)
```

> 这是重构后保留的**唯一活跃链路**（V2/derag2 世代）。历史世代（原始 run、corrected-v1/v2/v3、
> derag_v4、derag_v5 探针、judgecal 标定原稿等）已整体移到仓库根的 `../archive/code_legacy/`，本目录不再含。
> 完整实验讲解见仓库根的 `../32B强化学习_实验报告.md`。

---

## 一、目录速览

```
code/
├─ config.py                 # 集中配置：路径 / 两套系统提示 / 采样 / 训练超参 / Kimi key（走环境变量）
├─ requirements.txt          # 轻依赖（swift/torch/vLLM 装在服务器 conda 环境，不在此列）
├─ run_v2.py                 # ★主编排：V1→SFT→RFT→DPO 二叉树（2σ/3σ 双桶，断点续跑）
├─ run_v2_dpo_resume.py      # 把 AutoDL 训完的 DPO adapter 本地接回（合并+评测+因果对照）
├─ run_v2_grpo_online.py     # DPO→在线 GRPO 续训（warmup + Kimi-k2 online 两段）
├─ run_judgecal.py           # 裁判标定（量 Kimi 的分辨率，产 σ 表，见 §5）
├─ rebuild_coldstart_sft.py  # 用缓存改写重挑 SFT 集（k2-pass-930，零额外 Kimi）
├─ probe_rewrite_v2.py       # 改写提示词 A/B 探针（OLD vs FINAL vs FINAL_POOL）
├─ pipeline/                 # 各步骤 + 共用模块（见 §3）
├─ swift/                    # ms-swift 训练脚本（SFT/DPO/GRPO）+ GRPO reward 插件
├─ scripts/                  # 启动 / 监控 / 起 vLLM / 合并 LoRA
├─ tests/                    # 纯 CPU 契约测试（规则 / reward 词典序 / 数据合同）
└─ data/                     # 公司样本与中间数据（.gitignore，不入库，自备）
```

---

## 二、最终链路全景（输入 → 脚本/函数 → 关键门控 → 输出）

所有 V2 产物落 `OUTPUT_DIR/<V2_TAG>/`（本轮 `V2_TAG=derag2`），靠 `.<name>.done` 标记断点续跑。

### 共享上游（`run_v2.build_upstream`，只跑一次）
| 步 | 做什么 | 脚本 | 输出 |
|---|---|---|---|
| U0 | V1 重产 think/answer（RAG 腔系统提示、贪心 t=0） | `step01_build_v1_data.py` | `00_v1_outputs.jsonl` |
| U1 | 切 1739 训练 + **500 冻结验收** + 建池题集 | `step_v2_split.py` | `00_data_v2_{train,eval}.jsonl` |
| U2 | V1 答案池（每题贪心1+采样8=9条，收极性/数字/日期事实） | `step152_v1_support.py` | `152_v1_support.v2.jsonl` |

### 阶段 A — 冷启动 SFT（学去检索腔；纯 Kimi、无 GPU）
- `step_v2_coldstart.py`：Kimi **不喂 V1 think**、只给【问题+依据+池锚点】从头改写 → 三道门
  （① 规则门 `detect_rag_style` 无检索腔 ② facts 极性闸 不越界 ③ **σ 门** k=2 粗筛→k=16 双评，比 V1 干净 2σ/3σ）
  → 攒够 `V2_COLDSTART_TARGET=700` 个 2σ 即停 → 930 条（837 训+93 早停）。
- 训练 `swift/sft_on_model.sh`（LoRA r16/α32, lr 5e-5, ep `COLDSTART_EPOCHS=7`）→ `merge_lora_model.sh` 合并。

### 阶段 B — RFT 自采样（修 grounding、把在池率捞回地板）
- `step151_rft_selfsample.py`：SFT-merged 对训练池每题采 `V2_RFT_SELFSAMPLE_K=32` 条。
- `step_v2_rft_select.py`：三门（答案门 `answer_in_v1_pool` → 规则门 → σ 门 k16 2σ/3σ）→ 攒 `V2_RFT_TARGET=200`。
- 训练同 `sft_on_model.sh`（base=SFT-merged, lr 3e-5, ep 3）→ 合并成 **DPO 底座**。

### 阶段 C — DPO 偏好对（主升力）
- `step10_rollout.py`（RFT-merged 每题采 16） → `step_v2_dpo_pairs.py`：
  **chosen=(洗净 think Kimi≈7, V1 原 answer)**，**rejected=(V1 原脏 think Kimi≈1–2, 同一段 V1 原 answer)**
  —— **answer-lock**：一对里 answer 是同一段、只有 think 不同，梯度只压 think。攒 `V2_DPO_TARGET=900`。
- 训练 `swift/dpo_v2.sh`（`swift rlhf dpo`, β 0.1, lr 5e-6, ep 2, rpo_alpha 1.0；πref=base+禁用 LoRA）。
  服务器是 8 卡，本轮 DPO 实际在 AutoDL 2×A800 上跑（`scripts/run_dpo_autodl.sh`，路线 B 只产 adapter）。

### 阶段 D — 在线 GRPO（守住 DPO 增益 + 把在池率拉回 ≥0.85）
- `step_v2_build_grpo_data.py`：train-only，过滤无可比较事实题；每行带 `v1_answers_json` 供在线硬门。
- `swift/grpo_on_model.sh` + `swift/grpo_reward_plugin.py`（colocate vLLM, K=8）。两段课程式：
  **warmup**（30 步、只用规则+答案硬门、不调 Kimi）→ 合并 → **online**（90 步、接 Kimi k=2 在安全区内排序）。
- 在线 reward=**词典序硬门**（格式→答案在池→think 干净，越靠前越不可补偿；见报告附录与 `grpo_reward_plugin.py`）。

### 评测（三件套，所有阶段复用）
`step03_eval_infer.py`（500 题贪心）→ `step_v2_eval.py`：① Kimi 干净分 k=3；② 规则去检索腔通过率；
③ 答案在池率。真涨判据：两阶段均值差 > 约 3×SE≈0.15。产 `<tag>_{infer,scores,report,summary}.*`。

---

## 三、文件索引（一句话职责）

**入口编排器**
- `run_v2.py` — 8 叶/14-LoRA 二叉树主编排（建数据→swift 训→合并→评测→剪枝，断点续跑）。
- `run_v2_dpo_resume.py` — 接回 AutoDL DPO adapter（SHA256/provenance 强校验，与冻结历史 baseline 做因果对照）。
- `run_v2_grpo_online.py` — DPO→在线 GRPO 续接（建数据→reward 审计→smoke→warmup+online→合并→评测）。
- `run_judgecal.py` — 裁判标定主控（step160/161/162），产 σ 表与分辨率结论。
- `rebuild_coldstart_sft.py` / `probe_rewrite_v2.py` — SFT 数据重挑 / 改写提示词 A/B 探针。

**pipeline/ 步骤**
- `step_v2_split / step_v2_coldstart / step_v2_rft_select / step_v2_dpo_pairs / step_v2_eval` — V2 五步主链。
- `step_v2_build_grpo_data / step_v2_grpo_reward_audit` — 在线 GRPO 数据与 reward 预检。
- `step01_build_v1_data / step03_eval_infer / step10_rollout / step151_rft_selfsample / step152_v1_support`
  — 上游与采样步（出身较早世代但被最终链复用）。
- `step160/161/162_*` — judgecal 标定三步（造标定集 / Kimi 逐句打分 / 读分辨率曲线+σ 表）。

**pipeline/ 共用模块**
- `config.py`（在 code/ 根）— 集中配置。`logger.py` — 统一日志。
- `rules_v6.py` — 确定性规则层：`detect_rag_style`（表面检索腔）+ `answer_in_v1_pool`（答案漂移）。
- `rewrite_v2.py` — 去检索腔改写提示词（四铁律 + 池锚点 `build_pool_anchor`）。
- `judgecal_common.py` — Kimi 换词复述裁判 `judge_clean_score`（0–10，唯一活跃裁判）。
- `v2_common.py` — σ-可分选择（`confident_cleaner`/`cleaner_scores`，查 σ 标定表）+ 三件套口径 + v2 命名。
- `v2_paths.py` — V2 路径常量 + 数据助手（`sft_row`/`dpo_row` answer-lock 拼装、`gather_until` 早停续跑）。
- `reward.py` — 本地打分器（自 14B 沙盒移植；`parse_think_answer*`/`extract_references` 全链复用）。
- `kimi_client.py` / `kimi_budget.py` — DashScope Kimi 客户端（429 退避）/ 预算围栏与去重缓存。
- `vllm_client.py` / `infer_hf.py` — 本地 vLLM 客户端 / transformers 兜底推理（vLLM cu13 不可用时）。
- `reward_v3.py` `v3_utils.py` `v5_probe_common.py` `step06_rewrite_seeds.py` `plot_loss.py`
  — 被上述步骤传递依赖的工具（探针世代留下、最终链仍引用，勿删）。

**swift/**（训练脚本）
- `sft_on_model.sh`（SFT/RFT 共用）、`dpo_v2.sh`、`grpo_on_model.sh`、`grpo_reward_plugin.py`。

**scripts/**（启动/支撑）
- `run_v2.sh`+`monitor_v2.sh`、`run_dpo_autodl.sh`、`run_v2_dpo_resume.sh`+`monitor`、
  `run_v2_grpo_online.sh`+`monitor`、`run_judgecal.sh`+`monitor`、
  `serve_v1_vllm.sh` / `serve_model_vllm.sh`（起 vLLM）、`merge_lora_model.sh`（CPU 合并 LoRA）。

**tests/**（纯 CPU，不需 GPU）：`test_rules_v6` / `test_v2_grpo_reward`（reward 词典序）/
`test_v2_grpo_contract` / `test_v2_dpo_resume_contract` / `test_eval_format_accounting` / `test_infer_hf_preflight`。
`python -m pytest tests/ -q` 可直接跑。

---

## 四、怎么跑

主链（建议 tmux，前台启动防终端挂断 SIGHUP）：
```bash
export DASHSCOPE_API_KEY=sk-...          # Kimi key，只走环境变量，绝不写进代码
cd code/ && bash scripts/run_v2.sh       # 内部 exec python run_v2.py
bash scripts/monitor_v2.sh               # 另开一窗，只读监控（阶段/二叉树/漏斗/GPU/三分曲线）
```
AutoDL 上单独训 DPO：`bash scripts/run_dpo_autodl.sh <dpo_pairs.jsonl>`（2×A800，路线 B 只产 adapter→transformers 推理）。
本地接回：`bash scripts/run_v2_dpo_resume.sh`。在线 GRPO：`bash scripts/run_v2_grpo_online.sh`（env 见脚本头）。

---

## 五、裁判标定（judgecal）——为什么先量尺子再优化

整条链唯一靠大模型判的只有"换词复述照抄"（规则看不见）。开训前先用 78 条标定 think（13 参考 × 6 档脏度）
让 Kimi 各打 16 遍，量它的分辨率：得出"只有'基本干净 vs 重度照抄'大间距才稳定分得开"、
"整体评测 N=500/k=3 时 SE≈0.05、真涨门≈0.15"两条铁律 → 直接钉死 DPO 对子的造法和真涨判据。
σ 标定表（6 档）固化在 `v2_common.JUDGECAL_CALIB`，是 σ-可分选择门的依据。标定证据在 `../archive/results/judgecal/`。

---

## 六、环境 / 资源分工（8×A800）

- 训练 conda `zhjg_rl`（ms-swift 4.0.1）；推理 conda `vllm_env`（vLLM）；在线 GRPO 独立 `grpo_env`。**勿把 vllm 装进训练环境。**
- V1 权重、conda 路径、AutoDL 路径等均为服务器绝对路径，集中在 `config.py` + 各 `scripts/*.sh` 顶部，
  本机跑需按注释用环境变量覆盖（`ZHJG_WORK_DIR` / `V1_DIR` / `TRAIN_GPUS` 等）。
- 推理 serve `util=0.90` 只服务推理；训练 GRPO 的显存五连关见 §七。

## 七、红线（勿动）

1. **GRPO colocate 显存五连关**（`swift/grpo_on_model.sh`）：`vllm_gpu_memory_utilization=0.5` +
   `offload_model/optimizer=true` + `vllm_tensor_parallel_size=8(=NPROC)` + `move_model_batches=16` + `sleep_level=1`。
   调高 util 会 OOM。
2. **answer-lock**：训练样本 answer 永远拼 V1 原版，只训 think（`v2_paths.sft_row`/`dpo_row`）。
3. **`served_name='v1'` 是语义开关**：数据构建步把目标模型 serve 成名字 `v1`，`config.system_for` 据此套系统提示；
   serve SFT/RFT-merged 时要显式传 COLDSTART（去检索腔）腔绕过，别误套 RAG 腔（否则评分全错）。

## 八、⚠️ 安全与数据

- **DashScope key 一律走环境变量**（`config.py` 缺 key 只告警、不兜底）。历史明文 key 已从仓库清除并标 `REDACTED-ROTATE-ME`；
  该 key 早期曾随代码上传公网，**视为泄露，请到 DashScope 控制台轮换作废**。
- `code/data/`（公司样本、教师输出、judgecal 句集）与所有 `*.jsonl`/模型权重**不入库**（`.gitignore`），需自备。
  本仓库是**可读参考实现 + 实验报告**，不是从 clone 即可复跑的工件（缺公司数据/V1 权重/8 卡）。
