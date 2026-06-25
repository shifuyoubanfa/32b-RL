# derag_v5 headroom 探针包

**目的**：不训练、只采样+判定，回答一个问题——"那批有念手册毛病的题，到底有没有救、值不值得烧卡训练"。
跑完给两个数字 + 一个生死判断（GO_RL / GO_SFT_FIRST / NO_GO）。

## 怎么装（zip 内已带 `code/` 前缀，在 code/ 的【上一级】解压即自动合并进 code/）
```bash
cd /mnt/pfs/zhjg            # ← 注意是 code/ 的上一级（和现有 code/ 同级），不是进 code/ 里
unzip -o derag_v5_probe.zip # 文件落入 code/pipeline/step15x*.py、code/run_derag_v5_probe.py、
                            #          code/scripts/{run,monitor}_derag_v5_probe.sh
chmod +x code/scripts/run_derag_v5_probe.sh code/scripts/monitor_derag_v5_probe.sh
```
它复用现有 `code/pipeline/{vllm_client,reward_v3,kimi_client,step06_rewrite_seeds,logger}.py` 和 `code/config.py`，
**只新增文件、不改动任何现有文件**。入口脚本带自动定位 config 的兜底，万一解压位置不对也能找到代码根。

## 怎么跑（两页）
> Kimi(DashScope) key 已明文写死在 `scripts/run_derag_v5_probe.sh` 里作默认值，不必每次 export；
> 换 key 时改那一行，或 `export DASHSCOPE_API_KEY=...` 覆盖即可。
```bash
cd /mnt/pfs/zhjg/code                  # 进 code/

# 页面1：跑
bash scripts/run_derag_v5_probe.sh

# 页面2：另开终端看运行记录
cd /mnt/pfs/zhjg/code
V5_RUN_ID=<页面1打印的run_id> bash scripts/monitor_derag_v5_probe.sh
```

## 它做的五步（全自动）
1. serve RFT-merged → 在 224 + 训练集上让 **RFT 贪心生成 think** → **默认用 Kimi DERAG 判挑病题**（能看见
   "从资料向答案归纳"这种结构性念手册；确定性规则只抓字面词"参考问答对"会漏 10-20 倍，故只作 `V5_DETECT=rule` 备选）。
   全部题的打分明细落 `150_problems.all.jsonl`，便于看分布/复核漏检。
2. 同窗口让 **RFT 对每道病题自由答 16 遍**（存原始文本）。
3. 停 RFT，serve **原始 V1** → 每道病题让 **V1 答 8 遍**，建"V1 认可答案范围"（判答案跑没跑偏的尺子）+ 1 条 V1 贪心做改写底稿。
4. （CPU）评 RFT 16 遍里有几遍"think 干净 ∧ 答案在 V1 范围" → **X = 自救率**。
5. （Kimi）把 V1 的 think 改写干净、答案不动，看改得动几道 → **Y = 改写成功率**。最后汇总判断。

## 配置（按需 export，都有默认值）
| 变量 | 默认 | 含义 |
|---|---|---|
| `V5_RFT_MERGED_DIR` | `/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged` | 被测的当前模型 |
| `V5_V1_DIR` | `/home/nvme01/zhjg/V1-32B/checkpoint-1500` | 答案金标准 V1 |
| `V5_VLLM_GPUS` | `0,1` | vLLM 占的卡（TP=2） |
| `V5_TRAIN_CAP` | `1000` | 扫多少训练题找病题（0=全 2015，卡不要钱可设 0） |
| `V5_DETECT` | `kimi` | 挑病题方式：`kimi`(Kimi DERAG 判，看结构性念手册) / `rule`(确定性规则，只抓字面词) |
| `V5_DETECT_K` | `2` | Kimi 挑题每题判几次取均值降噪 |
| `V5_PROBLEM_TF` | `0.70` | trace_free 低于此 或 Kimi 标出结构性痕迹 = 病题 |
| `V5_RFT_K` | `16` | RFT 每题自采样遍数 |
| `V5_V1_N` | `8` | V1 每题采样遍数（建答案范围） |

## 产物（`$ZHJG_WORK_DIR/output/derag_v5_probe/<run_id>/`，下载这些给我）
- `159_probe_summary.md` / `.json` —— **结论 + 两个数字 + 生死判断**（先看这个）
- `153_rft_headroom.json` —— X 的分 train/eval 明细
- `154_rewrite_headroom.json` —— Y 的分 train/eval 明细
- `150_problems.jsonl` —— 病题清单 + 每题脏在哪
- `153_rft_headroom.jsonl` —— 逐题 16 遍各自的 think_clean/answer_in_support/pass（复核用）
- `154_rewrite_headroom.jsonl` —— 逐题改写前后痕迹数 + 新 think（复核用）

## 判据（写在 step159）
- **X ≥ 0.45 → GO_RL**：RFT 自己已能蒙出干净且不漂移的样本，SFT→DPO→GRPO 整链值得跑。
- **X < 0.45 且 Y ≥ 0.60 → GO_SFT_FIRST**：自采样信号弱，先用 Kimi 改写 SFT 搬中心，再续 RL。
- **X < 0.45 且 Y < 0.60 → NO_GO**：天花板太低，别烧卡，交报告。

> 注：以 `all`(train+eval 合并) 为准做判断；train 见过数据会偏乐观，故同时单列 eval 看水分。
> 全程不训练、不写任何 checkpoint，不碰现有产物；vLLM 用完即停。

## 断点续跑 & 显存
- **续跑**：用同一个 `V5_RUN_ID` 重跑，会自动跳过已完成步骤（产物非空即跳）。例如 s150/s151 跑完后中断，
  `V5_RUN_ID=<旧id> bash scripts/run_derag_v5_probe.sh` 会跳过 RFT 窗口、直接起 V1 接着 s152。
- **显存切换**：停一个模型后会**轮询 nvidia-smi 等显存真正释放**（30s 不掉就 `kill -9` + 按模型路径精准 pkill，
  只杀本探针进程），起下一个模型前再确认一次。解决了"上一个模型 TP worker 没退干净、下一个模型抢不到显存卡死"的问题。
- 万一手动中断留下残留 vLLM：`pkill -9 -f vllm; sleep 5; nvidia-smi` 确认 GPU 释放后再续跑。
