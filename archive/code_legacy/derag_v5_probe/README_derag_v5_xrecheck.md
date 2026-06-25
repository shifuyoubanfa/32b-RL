# derag_v5 · X 重测补丁（规则 X vs Kimi 真 X）

**目的**：探针报的 X=0.851 是用确定性规则判"think 干净"算出来的，而规则对结构性念手册全瞎
（489 病题 98% det_trace=0、clean 闸几乎不卡人、5 裁判复核真干净仅 ~0.44）→ X 系统性虚高。
本补丁**只换"判 think 干净"那一把尺子**（规则→Kimi，与挑题同一把），重算真 **X_kimi**，和规则 X 并排给你看。

**特点**：纯离线、**零 GPU**、不重新生成任何采样、**不碰任何训练显存设置**。只读探针已经在盘的 151/152。

## 装（并入现有 code/，和探针一套）
```bash
cd /mnt/pfs/zhjg            # code/ 的上一级
unzip -o rl_code.zip        # 覆盖更新 code/（含本补丁）
chmod +x code/scripts/run_derag_v5_xrecheck.sh code/scripts/monitor_derag_v5_xrecheck.sh
```

## 前提
探针（run_id=main）已经跑过，`output/derag_v5_probe/main/` 下有
`151_rft_samples.jsonl` + `152_v1_support.jsonl`（+ `153_rft_headroom.json` 用于对照）。
—— 你已经跑完探针，这些都在，直接重测即可。

## 跑（两页）
```bash
cd /mnt/pfs/zhjg/code
# 页面1：跑（纯 Kimi API，约几十分钟）
bash scripts/run_derag_v5_xrecheck.sh
# 页面2：另开终端看
bash scripts/monitor_derag_v5_xrecheck.sh
```

## 它做什么（一步）
对每道病题的 16 遍采样：
1. **先用确定性 `answer_in_support` 过滤**——只留"答案没漂 V1"的样本（这把尺子可信，不换；漂移的不可能自救，跳过省调用）。
2. **对这些 in-support 样本，用 Kimi 判 think 干没干净**（trace_free≥`V5X_TF_CLEAN` 且无结构痕迹），k 次取均值降噪。判到**首个干净即早停**（对"自救率"这个二值无偏）。
3. 这道题只要有 1 条 (Kimi 判干净 ∧ 答案不漂) = 自救成功。占比 = **X_kimi**。

## 配置（都有默认值，按需 export）
| 变量 | 默认 | 含义 |
|---|---|---|
| `V5_RUN_ID` | `main` | 读哪个探针 run 的 151/152，结果写回同目录 |
| `V5X_K` | `2` | 每条样本 Kimi 判几次取均值（降噪） |
| `V5X_TF_CLEAN` | `0.70` | trace_free≥此 且 无结构痕迹 = 干净（与挑题阈值对称） |
| `V5X_CAP` | `16` | 每题最多判几条 in-support 样本（早停于首个干净，cap 仅全脏时生效） |

成本想再降：`V5X_K=1`（省一半调用）或 `V5X_CAP=8`（每题少判几条，自救率会略偏低）。

## 产物（`output/derag_v5_probe/main/`，下载给我）
- `159b_xrecheck.md` —— **规则 X vs Kimi 真 X 对照表 + 判据**（先看这个）
- `153b_kimi_headroom.json` —— 汇总（X_kimi / X_rule，分 all/eval/train）
- `153b_kimi_headroom.jsonl` —— 逐题：每条 in-support 样本的 kimi_clean/tf/traces，`pass_idx`=救活它的样本号

## 判据（X 换成 X_kimi，沿用探针口径）
- **X_kimi ≥ 0.45 → GO_RL**：去 RAG 真有正样本可学，整链值得跑。
- **X_kimi < 0.45**：自采样信号弱，再看 Y（154 的 Kimi 改写率 0.53）决定先 SFT 搬中心还是 NO_GO。

> ⚠️ Kimi 有噪声：看 X_kimi 的**量级/方向**（是 ~0.45 还是 ~0.85），别抠小数点——一批题取比例，噪声会抵消。
> 真到训练那步，痕迹 reward **不能**直接用任何单把裁判在线打分（会被 GRPO 钻噪声 Goodhart），那是另一步的事。

## 断点续跑
同 run_id 重跑会跳过已完成（`153b_kimi_headroom.json` 非空即跳）。想重算：删它再跑。
