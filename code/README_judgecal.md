# judgecal · 判官标定（离线、零 GPU、只调 Kimi）

新版实验方案【模块1】：标定 Kimi「逐句换词复述识别」能力。设计与读法见
`docs/journal/109_judgecal_module_design.md`。

## 它干嘛
**本质是分辨率问题**：把"完全照抄/部分照抄/合法用事实/没照抄"当成带语义的精度档位，排成 6 档、用 0-10 数字锚定（没抄=10 / 抄1=8 / 抄2=6 / 抄3=4 / 抄4=2 / 完全照抄=0），让 Kimi 给整段 think 打 0-10 干净分，看曲线在哪压平——那拐点=分辨率上限=RL 天花板。
用一批人工四类标注、6 档铺匀的样本（78 条 think = 13 参考 × 6 档，由 `data/build_judgecal_dataset.py` 受控生成；换词复述力度经 `data/check_judgecal_vs_rule.py` 用真规则标定为"规则抓不到"）量两件事：
- **实验一**：分数跳不跳 → 该打几遍 K（噪声 σ/√k 压到 < 半个档距所需）。
- **实验二·分辨率曲线**：Kimi 平均干净分 vs 真实档位锚点 → 单调性(Spearman) + 相邻档位分不分得开 + **在哪段压平=RL 天花板**；单独盯【没抄↔抄1句】拐点（分不开=把合法用事实误判成照抄，阶段7 的病）。
- 每条 think 的客观真实档位(`true_level`/`anchor`)由 step160 从人工标签算出、留存。

**不 serve vLLM、不碰 GPU、不重新生成采样**——和 X 重测补丁一样安全，不动任何训练显存设置。

## 怎么跑（公司服务器）
```bash
# 页面1：跑（需 DASHSCOPE_API_KEY，脚本里有默认 key，离开内网请轮换）
bash scripts/run_judgecal.sh
# 页面2：另开终端看进度/结论
bash scripts/monitor_judgecal.sh
```
冒烟（只判前 3 条）：`JUDGECAL_LIMIT=3 bash scripts/run_judgecal.sh`

## 链路与产物
`step160`(校验装配) → `step161`(每条 think 判 16 遍, 唯一花 Kimi 钱) → `step162`(读两遍出报告)

产物 `output/judgecal/main/`：
- `162_judgecal_report.md` —— 人看的总报告（两实验 + 误伤逐句清单）
- `162_judgecal_decision.json` —— 机读结论（chosen_K / reworded_recall / legit_fp / go）
- `161_sentence_judges.jsonl` —— 逐遍原始判定（断点续跑会跳过，重判先删它）

## 数据
`data/judgecal_sentences.jsonl`（随代码带上，人工四类标注：verbatim/reworded/legit_use/original）。
扩样本格式与要求见 109 设计文档第 7 节。

## 可调环境变量
`JUDGECAL_KMAX`(16) `JUDGECAL_WORKERS`(3) `JUDGECAL_LIMIT`(0=全量)
`JUDGECAL_STABLE_THRESH`(0.05) `JUDGECAL_RECALL_MIN`(0.80) `JUDGECAL_FP_MAX`(0.15)
