# 税务 RL 蒸馏项目（一期：SFT 蒸馏 + 评测）

## 背景
公司已经有一套 **微调好的内部大模型 + RAG 检索**。这一期不动它，先把它当 teacher，对一个 **7B 学生模型** 做蒸馏 SFT，并用公司模型广场上的 **Kimi K2.6（Moonshot，262K 上下文）** 做对比评测，为后续的 RL 训练打基础。跨家选 judge 是为了规避同源 bias——teacher / student 都是 deepseek 系，judge 不能再用 deepseek。

## 目录结构

```
RL/
├── run.py                       # ★ 一键运行入口（编排 + 跳过已完成 + 统一日志）
├── config.py                    # 路径 / 接口 / 模型 / 训练超参集中点
├── requirements.txt
├── README.md
├── data/                        # 原始 xlsx
├── output/                      # 各阶段中间数据 + 最终报告
├── ckpts/                       # 学生模型权重 / LoRA / 合并产物
├── logs/                        # ★ pipeline.log（INFO+ 全量日志，滚动 20MB×5）
└── pipeline/
    ├── logger.py                       # 统一日志（文件 + stdout）
    ├── rag_client.py                   # 检索 + 流式调 teacher
    ├── step01_extract_usable.py        # A模型纯样本 -> '可用' query (498)
    ├── step01b_extract_summary.py      # 一阶段模型输出 -> '模型总结问题' (1741)
    ├── step01c_merge_queries.py        # 合并去重 -> 2239
    ├── step02_call_company_model.py    # 调公司 teacher 拿 think/answer（并发 + 断点续跑）
    ├── step03_build_sft_dataset.py     # 拼 system+user+assistant，切 train/eval
    ├── step04_download_student.py      # 魔搭 ModelScope 下载 7B base
    ├── step04_train_sft.py             # LoRA SFT（bf16 + grad ckpt，单卡 24G+）
    ├── step04c_merge_lora.py           # 合并 LoRA 回 base
    ├── step05_infer_student.py         # 学生模型对 eval 集生成 think/answer
    ├── step06_judge_with_claude.py     # Kimi K2.6 评测准确率 + RAG 痕迹
    └── step07_report.py                # 汇总报告
```

## 关键接口（公司内部）

| 用途 | URL | 鉴权 |
| --- | --- | --- |
| 检索 | `http://mlp.paas.dc.servyou-it.com/agentic_system_service/rag/getRetrieve` | payload 内置 appId |
| teacher（公司微调，流式） | `http://mlp.paas.dc.servyou-it.com/llm_finetune/v1/chat/completions` | 无需 |
| **judge（Kimi K2.6）** | `http://mlp.paas.dc.servyou-it.com/mudgate/api/llm/moonshot/v1` | mudgate：`Authorization` / `app_id` / `appId` 三个 header 同 key |
| 学生模型下载 | ModelScope：`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 公开模型 |

mudgate key 默认在 [config.py](config.py)；如需覆盖：`set MUDGATE_API_KEY=...`。

## 学生模型选型说明
原用 `deepseek-llm-7b-chat`（2023 纯指令模型），SFT 后 224 条里只有 3 条输出 `<think>` 标签——它没有原生推理链，轻量 LoRA 压不出 think 格式。改用 **`DeepSeek-R1-Distill-Qwen-14B`**（R1 蒸馏推理模型），原生先 `<think>...</think>` 再作答，契合"端到端 CoT"目标。换模型只改 [config.py](config.py) 的 `STUDENT_MODEL_ID`。

显存预算（14B, bf16, LoRA）：训练 ~35–45G、推理 ~30G、合并峰值 ~30G；96G 卡（RTX PRO 6000）富余。换小卡可降 `MAX_LEN` 或把 `BATCH_SIZE` 调回 1。

## 一键运行

```powershell
# 首次：装依赖
python -m pip install -r requirements.txt

# 一条命令跑全流程（数据 -> 训练 -> 推理 -> 评测 -> 报告）
python run.py
```

- 每个 stage 都自带产出判断，**已完成自动跳过**，可以放心多次重跑。
- 中途挂了：直接再 `python run.py` 接着跑；step02 / step06 内部还有 query 级断点续跑。
- 所有 INFO+ 日志写到 [logs/pipeline.log](logs/pipeline.log)（同时打印到屏幕），按 20MB 滚动、保留 5 份历史。

### 其它常用命令

```powershell
python run.py --list                            # 列出所有 stage 及完成状态
python run.py --only 02_call_company            # 只跑某一步
python run.py --from 04_train_sft               # 从某一步开始跑
python run.py --skip 04_train_sft 04c_merge_lora  # 跳过指定步骤
python run.py --force                           # 不看产出，强制重跑
```

## Stage 一览

| ID | 产出 | 描述 |
| --- | --- | --- |
| `01_extract_usable`   | `output/01_usable_queries.jsonl`   | A模型纯样本 -> '可用' query |
| `01b_extract_summary` | `output/01b_summary_queries.jsonl` | 一阶段模型输出 -> '模型总结问题' |
| `01c_merge_queries`   | `output/01c_all_queries.jsonl`     | 合并去重 |
| `02_call_company`     | `output/02_company_outputs.jsonl`  | 调 teacher 拿 think+answer |
| `03_build_sft`        | `output/03_sft_train.jsonl` / `_eval.jsonl` | 切分 SFT 数据 |
| `04_download_student` | `ckpts/deepseek-7b-chat/`          | 魔搭下载 7B |
| `04_train_sft`        | `ckpts/deepseek-7b-chat-lora/`     | LoRA SFT |
| `04c_merge_lora`      | `ckpts/deepseek-7b-chat-merged/`   | 合并 LoRA |
| `05_infer_student`    | `output/05_student_outputs.jsonl`  | 学生模型推理 |
| `06_judge`            | `output/06_judge_results.jsonl`    | Kimi K2.6 评测 |
| `07_report`           | `output/07_report.md`              | 最终对比报告 |

## Judge 输出字段（裁判 = Kimi K2.6）

| 字段 | 含义 |
| --- | --- |
| `student_accuracy_score` | 0~1，学生答案 vs teacher 答案的相符度 |
| `student_accuracy_label` | correct / partial / incorrect |
| `teacher_reasoning_humanness` | 0~1，teacher 的 think 像不像端到端 CoT（高=像人，低=有 RAG 气息） |
| `student_reasoning_humanness` | 0~1，student 的 think 像不像端到端 CoT |
| `teacher_rag_trace_types` | 多标签：explicit_ref / verbatim_copy / ref_enumeration / policy_source |
| `student_rag_trace_types` | 同上 |

`step07` 汇总：
- 学生模型准确率（均值 + 标签分布）
- humanness 均值 + 分布直方图（teacher vs student）
- RAG 痕迹类型频次对比表
- accuracy × humanness 交叉
- humanness 最低 Top-10 样本（RL 阶段重点）

为什么用 humanness 而不是 bool 标签：背景是公司之前对外说自己是端到端模型、实际用了 RAG，下一步 RL 想把推理里的 RAG 气息抹掉。humanness 是连续分数，既能做基线对比，也能直接作为 RL reward 的尺。

## 一期里没做的事
- RL（PPO/GRPO/DPO）。先 SFT 蒸馏；评测稳定后再上 RL。
- 离线缓存检索结果。当前每次 step02 都会重新打检索接口，可按需把 `retrieve_items` 一并存盘。
- 多 GPU 训练。需要时换 `accelerate launch --num_processes N` 启动 `step04_train_sft.py`。
