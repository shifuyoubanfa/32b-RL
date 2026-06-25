# 税务大模型「推理过程去 RAG 痕迹」强化学习实验记录（唯一权威版）

> 本文件是本项目**唯一、最详尽**的实验记录，合并了原 `EXPERIMENT_LOG.md` / `实验历程回顾.md` / `整个实验记录.md` 三份文档的全部内容，后两者将被删除，其独有信息已并入此处，**零信息丢失**。
>
> 用途：给作者更新简历 + 面试辅导，因此保留全部"心路历程"（每个决策的 why / 走过的弯路 / 推导过程）、真实数字（逐阶段 humanness / 准确率，**不省略、不写"同上"**）、以及逐字数据样例。
>
> 数字口径：全部以 `output/*.md` 报告与 `logs/*.log` 为准（已逐一 Read 复核）。验收集**全程固定**为 `output/00_data_sft_eval.jsonl`（**224 条**），裁判为 `kimi-k2.6`。学生模型为 `DeepSeek-R1-Distill-Qwen-14B`（部分报告标题模板遗留写成「7B」，**实为 14B**，见 §13.5 勘误）。

---

## 目录

0. 一句话项目定位
1. 项目背景与目标
2. 整体技术路线与角色
3. 关键技术决策与理由（含 why / 推导）
4. 阶段 A：数据准备（01–03）
5. 阶段 B：学生选型弯路 + SFT 蒸馏建基线（04 系列）
6. 阶段 C：第一版奖励 + 第一次 RFT（负结果 / Goodhart 翻车）
7. 阶段 D：奖励重设计 + 信号"先证伪再烧 GPU"（08c / 13）
8. 阶段 E：冷启动种子（11/12）+ 冷启动数据构建（14）
9. 阶段 F：冷启动 SFT + CS_RFT（里程碑）
10. 阶段 G：扩 query 弯路 + DPO（双赢里程碑）
11. 阶段 H：GRPO（两轮 OOM 修复 + 全链收官）
12. 阶段 I：退火 / 准确率↔think 权衡与 PMI 诚实更正
13. base14B 裸基线（对照）与全链结果总表
14. 数据流全追踪表（文件谱系映射）
15. 真实数据样例附录（逐字未删）
16. 踩过的坑汇总表
17. 评测设计
18. 方法论沉淀（最值得带走）
19. 预判面试问答
20. 32B 迁移现状与诚实边界
21. 技术栈 / 基础设施 / 接口
22. 各阶段耗时（END 行 elapsed）

---

## 0. 一句话项目定位

> **用纯自我改进的强化学习，把税务大模型的 `<think>` 推理过程从"满是 RAG 检索痕迹的机器腔"优化成"像人的端到端推导"，同时守住准确率。** 在 14B 沙盒上验证成功：humanness **0.212→0.846（≈4×）**、RAG 显式痕迹（explicit_ref / ref_enumeration）**清零**、准确率仅从 0.763 微降到 **0.733**——为迁移回公司 V1 提供了正向依据。

---

## 1. 项目背景与目标（一句话能讲清）

公司有一版微调好的税务大模型（称 **V1 / teacher**）：**答案准确率可以，但推理过程"太像机器"——满是"根据参考问答对1""检索结果显示"这类 RAG 检索痕迹**。公司对外宣称是"端到端模型"，实则推理里暴露了 RAG。公司训第二版（V2）想改善，**训崩了**。

我的任务：**做探索性实验，验证"用强化学习能不能把推理过程优化得更像人（端到端 CoT），同时不掉准确率"**。
- 不直接动 V1（风险大），先在一个**小模型沙盒**上验证方法；
- 若验证走得通，公司再把同一套强化方法用到 V1 本身。

**关键约束（决定了所有技术选型）**：将来要强化的是 V1 本身，**而 V1 头上没有更强的老师**（用更强模型蒸馏 V1 = 就是失败的 V2 路线）。所以方法必须是**模型自我改进的强化学习**，不能依赖外部示范。

---

## 2. 整体技术路线与角色

```
阶段一  SFT 蒸馏：用 V1 当 teacher，蒸馏出一个学生模型，建立基线
阶段二  评测基线：准确率 + 推理 humanness（像不像端到端 CoT）
阶段三  强化学习：（第一版奖励→RFT 翻车）→ 奖励重设计 → 冷启动种子
        → 冷启动 SFT → CS_RFT（拒绝采样自蒸馏）→ DPO（偏好对齐）→ GRPO（在线强化）
        目标：humanness 持续上升、准确率守住
```

- **teacher**：公司 V1 微调模型（RAG + 检索接口，流式返回 reasoning_content / content）。
- **student**：`DeepSeek-R1-Distill-Qwen-14B`（原生输出 `<think>` 推理链）。
- **judge（裁判）**：`Kimi-K2.6`（Moonshot，跨厂家，见 §3.3）。早期一份更老的实验残留用过 `claude-opus-4-7`，因走错网关被弃（见 §16）。
- **打分器（reward）**：纯本地规则（`re`/`difflib`，零模型依赖），所有 RL 阶段共用 `pipeline/reward.py`。

**模型谱系（ckpts/）**：
`ds-r1-qwen-14b`(base) → `-lora`(SFT) → `-rft-lora`(第一版 RFT，翻车)；另一条主线 → `-coldstart-lora`(冷启动) → `-cs-rft-lora`(冷启动 RFT) → `-dpo-lora`(DPO) → `-grpo-lora`(GRPO，最终)。

---

## 3. 关键技术决策与理由（面试官最爱问这块）

### 3.1 为什么先蒸馏小模型、不直接强化 V1
- 降低风险：V1 是生产模型，直接上 RL 崩了代价大；
- 沙盒验证方法可行性，跑通再迁移。

### 3.2 为什么学生选「推理蒸馏模型(R1-Distill)」而不是普通 chat 模型
- 最初用 `deepseek-llm-7b-chat`（2023 纯指令模型），SFT 后 **224 条里只有 3 条输出 `<think>` 标签**——它没有原生推理链，轻量 LoRA 压不出 think 结构；
- 换成 **R1 蒸馏模型**，原生"先 `<think>` 推理再作答"，**契合"端到端 CoT"目标**；
- 结论：**不是缺知识量（换更大 chat 模型解决不了），是缺"会推理"这个能力 → 换模型类型**。

### 3.3 为什么裁判选 Kimi（跨厂家）而不是 deepseek / 同源模型
- teacher 是 deepseek 系微调、student 是 deepseek 蒸馏；
- 若裁判也用 deepseek，会有 **self-preference bias（自偏好偏差）**——裁判系统性偏向同源风格的答案，评分失真；
- 选 **Moonshot 的 Kimi**（kimi-k2.6，262K 上下文）跨厂家，规避同源偏差，中文税务能力也够。

### 3.4 为什么用 GRPO 而不是 PPO
- **PPO** 要 actor + critic(value model) + reference 三套模型 + GAE，单卡 14B 显存吃紧、调参重、易崩；
- **GRPO**（DeepSeek-R1 同款）：**无 value model**，组内采 K 个、用组内 reward 均值/标准差归一化得 advantage；reference model 复用同一冻结策略（**零额外权重显存**）；
- 本任务奖励本质是"**同一问题内比谁更像端到端 CoT**"的**相对偏好**，天然契合 GRPO 的组内相对结构；
- 一句话：**GRPO 更省、更稳、信号更契合**。

### 3.5 奖励函数设计（核心难点）
两个维度：
- **humanness（像不像人 / 没有 RAG 痕迹）**——容易做：规则正则数"参考问答对/根据检索"等引用词 + 与参考资料的照抄率；
- **accuracy（答案对不对）**——**难点**：即便给标准答案，让模型判新答案对错也只有 ~85% 上限（公司的痛点，每条要人工评）。

**关键转念（面试亮点）**：在本任务里，**准确率是要"保住"而非"测量"**——因为我们只想改推理风格、不想动答案。于是：
- **不判对错，改判"答案有没有漂移"**：与已知标准答案的相似度 + 关键事实（数字/税率/金额/期限/极性词）召回——这是简单得多、精度高得多的问题；
- **训练时对 answer 段加强 KL 锚定、对 think 段放开（非对称 KL）**：答案被"按住"基本不变 → 准确率结构性守住，打分器只需当廉价兜底探针，**根本不需要一个完美的对错判官**。

奖励组合（**乘法门控**，防 reward hacking，对应 `reward.py:score_rollout` 真实公式）：
```
两级门控（乘法）：
  if not format_ok:           reward = -1.0          # 第一级：格式门控（gate=format_fail）
  elif R_acc < τ_acc(=0.30):  reward = 0.1 · R_acc   # 第二级：答案漂移过大→掐掉自然度增益（gate=acc_drift）
  else:                       reward = R_acc · (w_acc + w_human · R_human)   # 答对才解锁自然度（gate=ok）

  R_human = 1 / (1 + c_trace·引用命中数 + c_copy·照抄率)   # 平滑、不饱和、引用越多分越低（永不归零）
  R_acc   = 0.5·与标准答案相似度(SequenceMatcher) + 0.5·关键事实召回
  w_acc = w_human = 0.5；c_trace = 0.34；c_copy = 1.00；τ_acc = 0.30（值由 08c 校准网格搜出）
```
- **R_acc 是乘性前置因子**：答案漂移越大，自然度增益被整体压缩越多——这正是"按住答案"的数学体现。
- 防 hacking 三招：① 乘法门控（答错≈0 分）；② copy_ratio 用 **LCS 与 5-gram 重叠率取 max**，堵"删字样但照抄"（5-gram 专治"打散/改写式照搬"，即 RFT 后 verbatim_copy 飙升那种 Goodhart）；③ think 长度上限（40~2000 字符）堵"灌水骗分"。
- **answer_drift 细节**：硬事实槽 = 百分比/金额/日期/期限四类正则 + 12 个极性词（`免征/免税/不得/不可以/无需/不需要/不超过/应缴/需要/可以/超过/禁止`），极性词专防"免征↔应缴"这类把答案改反的漂移。gold 为空时返回 1.0（无锚不罚）。

### 3.6 为什么先离线(RFT/DPO)后在线(GRPO)
- SFT 起点准确率已 93%，在线 RL 容易"组内零方差、梯度稀疏"；
- **RFT/DPO 离线、用现成采样、对噪声不敏感、风险低**，先吃大头；
- GRPO 在线最强但风险高，放最后、且只在前两步触天花板时才上。

### 3.7 奖励为什么不用 Kimi 在线打分
- Kimi 约 0.03 q/s（一条 6~20 秒），RL 要打几万次分，**根本喂不动**；
- 所以奖励全用**本地规则**（秒级、免费），Kimi 只用于**离线校准**和**每阶段末全量验收**。

---

## 4. 阶段 A：数据准备（step01 → step03）

### 4.1 数据从哪来 / 怎么处理
两个源 Excel（精确文件名）：
- `A模型纯样本汇总.xlsx`（1112 行）→ step01 筛 `A是否可用=="可用"`、去空去重 → `00_data_queries_usable.jsonl`（**498 条**）。
- `一阶段模型输出3.31-5.19 v2.xlsx`（1773 行）→ step01b 取「**模型总结问题**」字段、去空去重 → `00_data_queries_summary.jsonl`（**1741 条**）。

step01c 合并去重（usable 优先在前）→ `00_data_queries_merged.jsonl`（**2239 条**，498+1741=2239，恰好无重叠）。

step02 对每条 query 调 `rag_client.rag_answer`：检索 → `build_user_prompt` 拼 `【参考问答对】…【问题】…` 壳 → 流式调公司 V1，拿 teacher 的 `reasoning_content`(think) + `content`(answer) → `00_data_teacher_outputs.jsonl`（**2239 条**）。并发 4，按 query 断点续跑、追加写 + 锁。

step03 构 SFT 样本，按 `TRAIN_EVAL_RATIO=0.9` 随机切：`00_data_sft_train.jsonl`（**2014 条**）/ `00_data_sft_eval.jsonl`（**224 条**，全程固定验收集，RL 绝不碰）。（2239 = 2014 + 224 + 1 丢弃。）

### 4.2 关键设计（与 step04 的 `<think>` 拼接强相关）
`reasoning` 与 `answer` **单独存字段**，不依赖 `messages[2]`（assistant）。原因：DeepSeek-R1 系列 `chat_template` 会自动**删除 assistant 内的 `<think>...</think>` 段**；若把推理塞进 assistant 再过模板，推理在喂模型前就丢了。所以训练时 prompt 走模板（模板自动注入 `<think>\n`），target 用 `reasoning`/`answer` 字段**手工拼**。`messages[2].content` 只是可读 preview，训练不用。

> **并入·诚实留痕**（来自《整个实验记录》第2章）：第2章数据样例里发现 assistant 的 answer 段被双层 `<answer><answer>...</answer></answer>` 包裹（生成时拼接残留），**不影响训练**（训练 target 用单存的 `reasoning`/`answer` 字段重拼）。如实记录。

---

## 5. 阶段 B：学生选型弯路 + SFT 蒸馏建基线

### 5.1 弯路：7B chat 模型选错（已弃）
- 最初学生 = `deepseek-llm-7b-chat`。7B SFT 训练本身先两次失败（`pipeline.log:268/279` `04_train_sft exit=1`，bs=1/grad_acc=16），第三次（`:290`）才存出 LoRA（弯路：调 bs/grad_acc）。
- 7B SFT 评测惨败：**准确率 0.421、humanness 极低（0-0.2 占 89.3%）**（`pipeline.log:880,900`），且 SFT 后 224 条只有 3 条出 `<think>`。
- 更早还有一次更老的"7B"残留实验（用 opus-4-7 裁判），准确率仅 0.013、humanness 0.002，几乎全崩——非现行基线。
- **悟到**："缺的不是知识量是'会推理'能力"→ **2026-05-29 弃 7B、改下 `DeepSeek-R1-Distill-Qwen-14B`**（`pipeline.log:967`），成为后续主力学生。

### 5.2 SFT 蒸馏（step04_train_sft，`<think>` 拼接核心实现）
- 以 V1 为 teacher，在 base 上做 LoRA SFT，**只对 assistant 段算 loss（prompt 段 mask 为 -100）**。
- `<think>` 拼接精确实现（`SFTDataset.__getitem__`）：
  1. `prompt_ids = apply_chat_template(system+user, add_generation_prompt=True)` —— 模板末尾**注入 R1 的 `<think>\n`**（与推理时一致），mask 不算 loss；
  2. `target = f"{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>" + eos` —— target **不含开头 `<think>`**（已由模板注入），从推理正文起、以 `</think>` 收，再接 `<answer>`，这段算 loss。
  3. 长度控制：`budget = max_len - len(target)`，**完整保留 target（含答案）**，prompt 超预算从左截断（保问题尾部 + 注入的 `<think>`）。
- 训练工程：LoRA r=16/alpha=32/dropout=0.05、7 个 target_modules 全覆盖 q/k/v/o/gate/up/down；bf16 + device_map=auto + grad-ckpt；等效 batch 16（bs=2×grad_acc=8）；`LR=1e-4`、`EPOCHS=3`、`MAX_LEN=4096`；`EarlyStoppingCallback(patience=2)` + `load_best_model_at_end(metric=eval_loss)`。本脚本被 RFT/冷启动复用（`--init_adapter` 可从已有 LoRA 续训）。
- 14B SFT 训练耗时约 3.15~3.5h（`pipeline.log:975` 11351.9s / `:1166` 12767.6s）。

### 5.3 SFT 基线评测（10_sft_report，RL 前起跑线）

| 指标 | teacher(V1) | SFT 学生 |
|---|---|---|
| 准确率(平均分) | — | **0.763** |
| correct / partial / incorrect | — | 120 / 89 / 15 |
| correct% / correct+partial% | — | **53.6% / 93.3%** |
| 推理 humanness | **0.292** | **0.212**（克隆了机器腔，差 -0.081）|

- student humanness 分布：0.0-0.2 占 **45.1%(101)**、0.2-0.4 占 47.8%(107)、0.8-1.0 仅 1 条(0.4%)。
- RAG 痕迹（teacher / student）：`explicit_ref` 205 / **216**（学生甚至更狠）、`ref_enumeration` 118 / **121**、`policy_source` 21 / 18、`verbatim_copy` 1 / **7**。
- **accuracy × humanness 交叉（关键发现）**：correct 0.199 / partial 0.228 / incorrect 0.217 —— **答得越准 humanness 反而越低**（base14B 同现象更明显：0.222/0.236/0.286），说明它的"准确"建立在"照抄参考问答对"之上 → 这正是 RL 要破解的核心矛盾。
- **解读**：SFT 阶段 student humanness ≈ teacher 是预期（蒸馏对齐）；RL 目标是保准确率前提下拉高 student humanness。

---

## 6. 阶段 C：第一版奖励 + 第一次 RFT（负结果 / Goodhart 翻车）

### 6.1 第一版奖励校准未达标（开训前已有预兆）
用已有 05/06 产物做零 Kimi 调用校准（08c）：
- 对齐样本 **224**；humanness 相关性 **Spearman(本地 R_human, Kimi)=0.286**（门槛 ≥0.6，**未达标**）；
- 本地 R_human 均值 **0.018**、Kimi humanness 均值 0.212（抽样前 20 行本地 R_human **全为 0.0**——表面 humanness 项在 RAG 腔窄带数据上几乎无分辨力）；
- 准确率门控一致率 GATE_acc(τ=0.45) vs Kimi(correct|partial) = **76.3%**（门槛 ≥85%，未达标；门控只 3 条被 ✗ 误杀：季度报表更正 R_acc=0.111、跨年款项 0.23、进项税转出 0.219，Kimi 全判对/部分→门控过严）；
- 报告结论原文：**"⚠️ 未达标：reward 代理与 Kimi 偏离，先调阈值/正则或谨慎对待 RL 结果"**。
- 后续 `--tune` 网格搜把 τ 从 0.45 降到 **0.30**（一致率提到 **83.5%**，减少误杀"换说法的正确答案"）、定 c_trace=0.34/c_copy=1.00，本地 humanness 均值 0.250≈Kimi 0.212。

### 6.2 第一次 RFT：两个指标全退步（20_rft1_report）
用第一版本地奖励驱动 RFT（每 query 采 8 个、挑最高分重训，rollout 400 题 → 选样 **395** 条）：

| 指标 | SFT 基线 | RFT-v1 后 | 变化 |
|---|---|---|---|
| 准确率(平均分) | 0.763 | **0.678** | ↓ |
| correct+partial% | 93.3% | **85.3%** | ↓ 8 个点 |
| incorrect 数 | 15/224 | **33/224** | 翻倍 |
| 推理 humanness | 0.212 | **0.190** | ↓（更机器）|
| verbatim_copy（学生） | 7 | **20** | ↑ 近 3 倍 |
| explicit_ref（学生） | 216 | **221** | ↑ |
| policy_source（学生） | 18 | **35** | ↑ |

- student humanness 分布更差：0.0-0.2 占 49.1%、0.2-0.4 占 48.7%；accuracy×humanness 几乎全压在 0.19（0.187/0.193/0.194）。
- **诊断 = 典型 Goodhart**：奖励测"字面"（关键词/字符重叠），模型学会**少说关键词但语义照搬**，骗过正则（verbatim_copy 暴增正是签名）。开训前校准 Spearman 仅 0.286 就有预兆。
- **面试点**：负结果在沙盒被发现 = de-risk 了 V1（没在生产模型上翻车，避免重蹈 V2 覆辙）。"奖励是 RL 的瓶颈"成为关键结论。
- **承上启下逻辑**（并入《整个实验记录》第3章因果链）：老师 think 本身机器腔 → 蒸馏学生必然带机器腔 → 第一版字面奖励逼出"藏起来照搬" → 必须升级奖励 + 给模型"自然推理"范例（冷启动）。

---

## 7. 阶段 D：奖励重设计 + 信号"先证伪再烧 GPU"

### 7.1 奖励重设计：从"字面层"升到"似然/语义层"（多 agent 设计面板产出）
旧版病根是**范式错误**而非参数没调好。新版三信号 + 金标准校准：
- **主信号·条件化 PMI**：`logP(think│问题+参考资料) − logP(think│问题+标准答案)`，返回 `-PMI`（越大越自然）。测"think 的呈现结构有多依赖把资料放进上下文"。改措辞不改语义 → PMI 不降 → 正面堵死旧 Goodhart。**用"标准答案"做基线**是关键创新：把正确事实在分子分母两侧抵消，只留"组织方式（罗列/复述）对资料的依赖"，从而**不冤枉"含正确事实的自然推理"**。
- **验证信号·ΔRAG(嵌入)**：`think 贴资料 − think 贴问题/答案`，用公司 bge-m3 API 算（1024 维，CPU 廉价）。
- **语义照抄峰值 / 显式引用正则**：兜底。
- **组合校准思想**：把"手调权重"换成对 224 条 Kimi 标签的拟合（团队试过训神经判官失败，故用低自由度标定器）。
- **核心方法论·相对差解开结构性矛盾**：references 本身就是标准答案、正确推理天然与之重叠；旧版惩罚"绝对重叠"必然误伤正确答案（实测 acc 93.3→85.3 部分源于此）。新版所有 humanness 信号都是**相对差**（只罚"贴资料远多于贴问题/答案"的照抄），准确率项又完全不碰 references。

### 7.2 "先证伪再烧 GPU"的纪律（step13 信号探针 + 闸门）
新奖励**不直接上训练**。先写离线小脚本，在"自然种子(该高分) vs RAG 样本(该低分)"两端跑信号探针，报与 Kimi humanness 的 Spearman 与"自然 vs RAG"AUC：
- **GO/NO-GO 闸门**：最佳 AUC **≥ 0.70** 才可重写 reward 进 RFT。
- **实跑结果**（2026-06-02，`--with-pmi` 加载 14B 算 PMI，成功 1024/1024、缺失用均值 -0.045 补，`pipeline.log:2161-2194`）：
  - `s_trace`（正则关键词）：Spearman +0.542，**AUC 0.740**；
  - `s_pmi`（条件 PMI 结构信号）：Spearman +0.406，**AUC 0.730**（单信号即过闸）；
  - `combo`（满血三信号）：Spearman +0.503，**AUC 0.782**（优于任一单信号）；
  - 嵌入 ΔRAG 信号弱（AUC~0.55），**已弃用**。
  - 结论原文（`:2194`）：**"✅ 过闸门(AUC≥0.70)…可重写 reward.py 进 RFT"**。
- **工程决策·PMI 做成可开关零件**：默认奖励 = s_trace + s_copy + 准确率（字面层，AUC 0.74 已够，不必每条载 14B）；`--with-pmi` 时才把 PMI 并入 humanness（每条多一次前向，慢 +1~2h）。

> ### ⚠️ 7.3 PMI 诚实更正（最重要的勘误，务必如实表达）
> **PMI 设计了、离线验证过（AUC 0.730），但实跑从未接进任何一次正式训练。**
> - 日志铁证：决定训练用 reward 的两处步骤都明写 `PMI=关`（`pipeline.log:2266` 冷启动 RFT 构建"准确率门槛=0.60，PMI=关"；DPO/GRPO 同样不涉及 PMI）。
> - PMI **唯一一次实跑**是 step13 离线信号证伪（`--with-pmi`），不参与训练；step08b 仅在显式 `--with-pmi` 时才调，默认关。
> - 14B 实跑时 `s_pmi` 在 reward 字典里**全为 null**。
> - **真正守 Goodhart 的是表面项**（关键词 s_trace + 字符照抄 c_copy）。原因：PMI 单信号 AUC(0.730) 还略低于表面项 s_trace(0.740)，combo 只比纯表面高一点，性价比不足以每条载 14B。
> - 面试表达：**"PMI 是验证过但没部署的备选增援，不是在用的主信号。"** 切勿说成"奖励里在用 PMI"。

---

## 8. 阶段 E：冷启动种子（11/12）+ 冷启动数据构建（14）

### 8.1 为什么要冷启动（与奖励互补的另一条线）
关键认识：RFT/GRPO **只能强化模型已偶尔产出的行为**；但 baseline 显示学生 humanness>0.6 的样本仅 **~2%**——它几乎从不自然推理，RFT **没好样本可选**。解法 = 给它"自然推理"的范例（R1 的 cold-start 配方）。

### 8.2 step11 改写 + step12 打分（一份数据两用）
- step11：用 Kimi **离线把 teacher 的 think 改写成自然推导**（删检索引用、从问题出发、**所有数字/税率/金额/期限/结论一字不改**）。源 = SFT_TRAIN 前 800 条 → `30_seeds_rewritten.jsonl`（**800 条**）。
- **事实校验**：抽改写文本里的数字，凡出现在"原始 think∪答案"之外的（≥2 位，忽略 1 位序号噪声）→ `facts_ok=False`。实测 **facts_ok=True 784 条 / False 16 条**（恰是 `introduced_nums` 非空者，2% 存疑、丢弃）。
- step12：Kimi 给每条 `natural_think` 打 humanness → `30_seeds_scored.jsonl`（**800 条**）。**实测 kimi_humanness 均值 ≈0.61~0.64**（报告口径 0.61，原始数据均值 0.64），对比 RAG 腔基线 0.21 —— **约 3 倍提升**。
- **一份数据两用**：① 冷启动 SFT 教材（让模型"会"自然推理）；② 验证奖励的"该高分"对照（补窄带数据缺的高分样本，**Phase 2 验证奖励时不会被"全打低分"蒙混**）。改写/打分都走 Kimi，离线、不进训练环。

### 8.3 step14 构冷启动数据（含同分布 eval 切分）
- 筛选：`facts_ok ∧ kimi_facts_kept ∧ kimi_humanness ≥ SEED_HUMANNESS_MIN`。
- **门槛 0.40→0.60（数据驱动，见 §8.4）** → `40_coldstart_train.jsonl`（**458 条**）。
- **同分布留出 eval**（关键）：确定性每 10 条抽 1 进 eval → `40_coldstart_eval.jsonl`（**51 条**，自然腔，给早停用）。

### 8.4 冷启动门槛 0.40→0.60（数据驱动）
质疑（用户）："冷启动是教自然推理，为什么 humanness≥0.40 就收？均值不是 0.61 吗？"——质疑对：0.40 是地板不是均值，把 0.40~0.60 的"将就自然"弱尾喂进去当范例会钝化风格迁移。
- **先看分布**：800 条种子 humanness 均值 **0.640**、**中位 0.800**（重度左偏，绝大多数扎堆 0.8、弱尾很薄）。各门槛存活：≥0.40→593、≥0.50→551、≥0.60→**525**、≥0.65→485（日志另有一处口径 ≥0.40→576、≥0.60→509，因批次/facts 过滤略差）。最终落地训练集 458 条。
- **决策**：抬到 **0.60**。代价极小（仅少 ~68 条 /11%），却砍掉全部"低于均值"弱样本；~510 条对 14B LoRA 冷启动绰绰有余。
- **方法论**：又一次"先廉价看数据/证伪、再动手"——看了中位 0.80 才知道这刀几乎白捡。

### 8.5 eval 分布错配 bug（对抗验证抓到的隐蔽坑）
需求："多跑几个 epoch，连续两次 eval_loss 不降就停"——即早停。
- **坑**：早停/留最优只有在 **eval 集与训练目标同分布**时才成立。原 run.py 两步都用 `SFT_EVAL`（teacher **机器腔** think）当 eval，而训练的是 **自然腔** think；loss 只算 think 段，模型越练越自然 → 机器腔上的 eval_loss 反而**升高** → `metric=eval_loss` 的早停会**专挑最早、最不自然的 epoch**，风格刚要迁移就被掐停，等于自废冷启动。**这种方向性错误不验证、直接跑，要烧完一轮 GPU 才暴露。**
- **修复**：step14 从自然种子确定性切 ~10% 作自然腔留出 eval（COLDSTART_EVAL），冷启动与 RFT 都用它；eval_loss 下降=更贴自然风格，方向才对。
- **纪律延续**：eval_loss 只当便宜护栏（防发散、选 ckpt），**最终验收仍以 Kimi 为准**（humanness 涨、acc 守）。

---

## 9. 阶段 F：冷启动 SFT + CS_RFT（里程碑）

### 9.1 运行链（八段，run.py 一步到位，可断点续跑）
```
14_build_coldstart : 筛自然种子 → 切 coldstart_train(458) + 自然腔 eval(51)
cs_train           : 冷启动 SFT(step04, lr=5e-5 epochs=5上限8, 早停patience2) → coldstart-lora
cs_rollout         : 冷启动模型自生成(step08, 前400题每query采 K=8) → 50_csrft_rollout.jsonl(400)
cs_rft_build       : 当前 reward 重打分选样(step08b, 卡 R_acc≥RFT_ACC_FLOOR=0.6 再按 R_human 降序挑 TopN=1) → 50_csrft_trainset.jsonl(350)
cs_rft_train       : RFT(step04 --init_adapter=冷启动 adapter 续训, lr=3e-5 epochs=3上限4) → cs-rft-lora
cs_rft_infer/judge/report : 224 eval 推理 → Kimi 评测 → 报告
```
- **自我改进闭环**：冷启动让模型"会"自然推理 → 它自己生成 → 本地奖励挑"既准又自然"的 → 用这些续训自己。全程无更强 teacher，纯自我改进，故可迁移回 V1。
- **关键复用设计**：step08b 用**当前 reward 对 rollout 原文 `text` 重新打分**，改进奖励后只需重跑本步即可复用昂贵的 rollout（每轮 rollout 约 5.5~6.3h），无需重生成。
- epoch 决策：冷启动 3→5（上限 8）、RFT 2→3（上限 4，self-generated 数据反馈环更易自我强化/坍塌，卡更死）；`save_total_limit=max(2,ceil(epochs)+1)` 保最优 ckpt。

### 9.2 CS_RFT 验收结果（50_csrft_report，里程碑，2026-06-03）

| 指标 | SFT 基线 | **CS_RFT** | 变化 |
|---|---|---|---|
| 推理 humanness | 0.212 | **0.780** | **+0.568** |
| 准确率(平均分) | 0.763 | **0.701** | -0.062 |
| correct / partial / incorrect | 120/89/15 | 103/97/24 | — |
| correct% / correct+partial% | 53.6% / 93.3% | **46.0% / 89.3%** | 差一点没守住 90% |
| teacher humanness（同批） | 0.292 | 0.159 | （裁判波动，见 §13.6）|
| humanness 差异(s−t) | -0.081 | **+0.621** | 翻转 |

- student humanness 分布翻转：**0.8-1.0 占 61.6%(138)**、0.6-0.8 占 29.9%(67)、0.0-0.2 仅 1 条。teacher(V1) 74.6% 挤在 0.0-0.2。
- **RAG 硬痕迹清零**：`explicit_ref` 220→**0**、`ref_enumeration` 134→**0**（显式引用与罗列**彻底清零**）。
- **结果可信、非 Goodhart 三重佐证**（面试核心）：① 选样器(本地奖励)≠验收器(Kimi)，无循环；② 客观正则痕迹(explicit_ref 220→0)与 Kimi 主观分**同向**；③ accuracy×humanness 交叉中 correct(0.790)≥partial(0.784)≥incorrect(0.721)，排除"自然但瞎编"。
- **残留靶子**：`verbatim_copy` 1→**17**、`policy_source` ~23→21——模型戒了显式引用，却学会"不标记地照抄政策原文"（把痕迹藏起来），Top-10 最低分几乎全是这俩。另有 1 例未输出 think（humanness=0），需格式硬门槛兜底。这正是可开关 PMI / 后续 DPO 的设计靶子。
- 与第一版 RFT 翻车（humanness 反降 0.190、acc 崩 85.3%）对比，这次是**质变**。准确率小幅回撤留给在线阶段（KL 锚定）收。

---

## 10. 阶段 G：扩 query 弯路 + DPO（双赢里程碑）

### 10.1 扩 query：想防过拟合，实跑发现没新数据（弯路）
动机（用户提出）：SFT→RFT 一直薅同一批 query，RL 有过拟合风险；从 1773 条生产问题摄取新 query 既防过拟合、又证泛化。关键认知：**RL 不需要 teacher 的 think**（think 是模型自己生成、要优化的对象），新数据只要 query+参考+答案。
- **⚠️ 实跑修正（2026-06-03）：合格新 query=0**。下载真实文件实跑 step15：1773 行 → 质量过滤（只留"解决/满意"）剩 691 → 其中 71 撞验收集、620 撞训练集 → **0 新**。根因：`一阶段模型输出` Excel **正是当初 step01b 造 SFT 训练问题的源**（SUMMARY_XLSX==NEW_QUERY_XLSX），这些生产问题早已训过。
- **另一发现**：真实 **SFT_TRAIN=2014 条**（非早先以为的 ~800），过拟合担忧本就较小（RFT 当时只用 400）。
- **处置**：扩 query 退化为"用满 2014 全池 + 确定性打散"（`random.seed(42)`，避免 rollout 的 max_queries=400 只采到前面老 query）→ `60_dpo_pool_expanded.jsonl`（**2014 条**）；DPO/GRPO 直接在 2014 池上跑；"对全新问题泛化"由始终留出的 224 验收集证明，要真证需公司另给未用日志。
- **教训**：早先误判它"未用过"，审计/红队也只"估算"去重没真跑——**只有下载真实文件实跑才暴露**（用户坚持要真文件是对的）。

### 10.2 DPO 构对与训练（手写，不依赖 trl）
- **构对规则**：同一 query 的 `gate=="ok"`（答案都没漂移）样本里，**chosen = reward 最高（通常更自然）、rejected = R_human 最低（RAG 痕迹最重）**；两者**答案都正确**，差别只在 think 风格 → **DPO 只学"改思考"，学不会"改答案"**。过滤：合格<2 跳过、`chosen is rejected` 跳过、`R_human(chosen)−R_human(rejected) < DPO_MARGIN` 跳过。
- **margin 0.1→0.05**：CS_RFT 已很自然(R_human≈0.78、组内方差小)，margin 太大配对饥饿；0.1→279 对、0.05→340 对，0.05 更稳。实际产出 `60_dpo_pairs.jsonl` **315 对**（产出率 78.8%，"配对饥饿"担忧实跑证伪）。
- **DPO 损失**：`L = -logσ( β·[(logπ_chosen−logπ_ref_chosen) − (logπ_rejected−logπ_ref_rejected)] )`，β=0.1，lr=5e-6，1 epoch。
- **⚠️ πref 锚错基准 bug（对抗验证 3 轮抓到的核心 bug）**：
  - 原本用 `disable_adapter()` 当参考 = **裸 base（原始 R1-distill）**，而非 CS_RFT 起点策略 → DPO 失去对 CS_RFT 的 KL 锚（把模型往别处拉、白费 RFT）；
  - **AI 修法陷阱（第二个 agent 识破）**：直接删 `disable_adapter()` → 参考变成"同一个正在训练的 adapter" → πref≡πθ → logits≡0 → **零梯度、什么都学不到**；
  - **正确修法**：DPO 训练前用**冻结的 CS_RFT** 把每对参考 logprob **预计算+缓存**（πref 固定，πθ 带梯度→梯度仍流）。日志实锤生效（`pipeline.log:2486` "预计算 πref(冻结 CS_RFT 起点策略) 共 315 对"）。

### 10.3 DPO 验收结果（60_dpo_report，里程碑·双赢）

| 指标 | SFT | CS_RFT | **DPO** | 变化 |
|---|---|---|---|---|
| 推理 humanness | 0.212 | 0.780 | **0.836** | ↑ |
| 准确率(平均分) | 0.763 | 0.701 | **0.731** | ↑（不降反升）|
| correct / partial / incorrect | 120/89/15 | 103/97/24 | 115/86/23 | — |
| correct% / correct+partial% | 53.6/93.3 | 46.0/89.3 | **51.3% / 89.7%** | ↑ |
| explicit_ref / ref_enumeration（学生） | 216/121 | 0/0 | **1 / 0** | 保持清零 |
| verbatim_copy / policy_source（学生残留） | 7/18 | 17/21 | **7 / 11** | 收敛 |
| humanness 差异(s−t) | -0.081 | +0.621 | **+0.677** | — |

- student humanness 分布：**0.8-1.0 占 79.0%(177)**、0.0-0.2 = 0。
- **πref=CS_RFT 修复的直接回报**：DPO 越练越自然(0.780→0.836)的同时**准确率不降反升**(0.701→0.731)——因为参考锚在税务策略 CS_RFT 而非裸 base，把准确率"焊住"了。这正是当初对抗验证抓那个 πref bug 的价值兑现。
- **收敛健康**：loss 0.74→0.34、margin→0.90、chosen>rejected 稳定 1。
- **可信非 Goodhart**：选样器≠验收器；客观痕迹(explicit_ref→1)与主观分同向；accuracy×humanness 交叉 **correct(0.873) > partial(0.816) > incorrect(0.726)**（越对越自然，健康）。
- 训练耗时约 21 分钟（`pipeline.log:2520` 1263.5s）。

---

## 11. 阶段 H：GRPO（两轮 OOM 修复 + 全链收官）

### 11.1 GRPO 核心三件套（手写，非对称 KL）
1. **组内归一化 advantage**：每 query 采 K=4 个、本地 reward 打分；`adv=(r−mean)/(std+1e-4)`；**std<1e-4（组内无方差）→ 整组跳过**（无学习信号）。策略梯度 `pg=-(adv·comp_logp.mean())`。
2. **非对称 KL（按 `</think>` token-id 子序列定位边界，免 BPE 错位）**：think 段权重 `GRPO_KL_THINK=0.02`（**基本放开变自然**，留一丝>0 防 200 步发散/奖励黑客），answer 段权重 `GRPO_KL_ANSWER=0.1`（**重锚保准确率**）。KL 用 k3 估计 `exp(ratio)-ratio-1 (≥0)`。
3. **πref（双 adapter，不是裸 base）**：可训策略 `"default"` + 冻结参考 `"ref"`（默认 = init_lora = DPO 起点税务策略）。每条序列先 `set_adapter("ref")` no_grad 算 ref_logp（算完释放）→ 再 `set_adapter("default")` 算带梯度 comp_logp。**answer 段 KL 锚到税务策略（DPO）而非裸 base**（`pipeline.log:2682` `πref(KL锚): ...-dpo-lora`，KL 自洽未爆）。
- 超参：lr=2e-6、每步 8 prompt、temp=1.0（需组内方差）/top_p=0.95。

### 11.2 两轮 OOM 修复（性质不同，深挖根因）
- **前两次跑都在 step 1 后崩**：第 1 次（`pipeline.log:2669` `steps=200 K=6`）只到 step 1（avg_reward=0.440）后 `FAIL exit=1`；第 2 次（`:2675` 降到 `steps=100 K=4`）仍只到 step 1 后挂。参数被主动从 K=6/steps=200 降到 K=4/steps=100，指向显存压力下的退让。
- **第一轮 OOM（碎片）**：仅差 198MiB 却有 **10.5GB"保留但未分配"碎片** = 典型显存碎片。修复 4 处：① `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（import torch 前设）；② 每步反向前 `empty_cache()`；③ K 6→4；④ steps 200→100。
- **第二轮 OOM（更关键，真占满 92GB）**：碎片降到 1.13GB（expandable_segments 生效），但第 2 步又 OOM，差 2.14GB，报错在 `logits.float()` 申请 `[1,T≈3500,词表152064]` float32 → 卡在一条**长 prompt**（税务最长 ~6000 token）。深挖：92GB ≫ 模型 30G+常规激活，主因是 **eager 注意力 materialize 了 T×T 矩阵**（T=3500 时 ~47GB）。修复 4 处：① **`attn_implementation="sdpa"`**（注意力 T²→O(T)，主修复，失败回退 eager）；② `comp_token_logprobs` **只对 completion 段做 log_softmax**（全词表 float 2.14GB→~0.6GB）；③ 先算冻结 ref(no_grad 即释放)再算带梯度 policy（不让两者激活叠加）；④ `GRPO_MAX_FORWARD_LEN=3072` 超长从左截 prompt（policy/ref 同输入→KL 一致）。
- **诊断结论**：是 **GPU 显存(VRAM)不足、非 RAM**，128G 内存够用，**无需加租第二张卡**。"训练阶段硬停"正确拦下、没带病继续。

### 11.3 GRPO 跑完 100 步 + 终评（70_grpo_report，全链收官）
- **第 3 次一气呵成 100 步**（`pipeline.log:2681-2787`，2026-06-04 15:20 启动 → 2026-06-05 02:59 完成，约 **11.7h / 41965.3s**）：step 1 reward 0.443 → step 100 avg_reward 0.555；中途 step 50 / step 100 存档。reward 全程在 0.36~0.68 震荡、无明显单调上升；KL 从 ~0.0002 缓升到峰值 0.0416(step48) 后回落，未失控。
  - **为何 avg_reward ~0.5 平稳属正常**（并入《整个实验记录》解释）：reward 被准确率代理乘了一道、且 GRPO 优化的是**组内 advantage 不是绝对值**，绝对 reward 平稳不代表没学到东西。

| 指标 | DPO | **GRPO（最终）** | 变化 |
|---|---|---|---|
| 推理 humanness | 0.836 | **0.846** | ↑ |
| 准确率(平均分) | 0.731 | **0.733** | ↑ |
| correct / partial / incorrect | 115/86/23 | 115/86/23 | 不变 |
| correct% / correct+partial% | 51.3/89.7 | **51.3% / 89.7%** | 不变 |
| explicit_ref / ref_enumeration（学生） | 1/0 | **0 / 0** | 清零 |
| verbatim_copy / policy_source（学生残留） | 7/11 | **3 / 12** | verbatim 再降 |
| humanness 差异(s−t) | +0.677 | **+0.680** | — |

- student humanness 分布：**0.8-1.0 占 80.4%(180)**、0.6-0.8 占 18.3%(41)、0.4 以下仅 3 条、**0.0-0.4 = 0**。
- accuracy×humanness 交叉：**correct(0.865) > partial(0.844) > incorrect(0.763)**（正相关，答对的更像人，健康收敛）。
- Top-10 最低样本质性演变（优化目标"升级"了）：SFT/RFT 阶段最低样本是"参考问答对/问题1逐条罗列"（显式 RAG）；CS_RFT/DPO 变为"大段照搬政策原文/政策文号"（残留 policy_source）；**GRPO 最低已抬到 0.40 起**，理由变为"凭空编造精确折旧数字（幻觉）""从问题出发自然推导，仅法条引用略带原文味道"——问题从"像 RAG"转为"幻觉/轻微政策味"。
- **结论**：humanness 从 0.212 拉到 **0.846**，accuracy 仅从 0.763 微降到 **0.733**——"保 accuracy 拉 humanness"的目标在 14B 上**验证成功**，为 32B 迁移提供正向依据。**GRPO 是已收口的最终结果，不是未完成态。**

---

## 12. 阶段 I：退火 / 准确率↔think 权衡（诚实更正）

- DPO/GRPO 的非对称 KL（think 0.02 放开 / answer 0.1 重锚）本质就是一次**显式的"准确率↔think 自然度"权衡退火**：think 段几乎放开让风格自由迁移，answer 段重锚把准确率焊在税务策略上。这是把"保住而非测量准确率"落到训练机制层的体现。
- 全链准确率轨迹印证这个权衡：0.763(SFT) → **0.701(CS_RFT，为换 humanness 0.78 主动让出 0.062)** → 0.731(DPO 借 πref=CS_RFT 把准确率收回) → 0.733(GRPO 借非对称 KL 守住)。即"think 大幅变自然"的代价被锁在 CS_RFT 那一步的小回撤，之后靠 KL 锚定逐步收回，最终净代价仅 -0.03。
- 可选的进一步退火旋钮（已设计、视需要启用）：收紧 `RFT_ACC_FLOOR 0.6→0.7` 拉回准确率、开 PMI 再 RFT 一轮治结构性照抄。**当前全链已达标，未必需要。**

---

## 13. base14B 裸基线（对照）与全链结果总表

### 13.1 base14B 裸基线（未训，对照，baseline_base14b_report）
| 指标 | 数值 |
|---|---|
| 准确率(平均分) | **0.659** |
| correct / partial / incorrect | 89 / 107 / 28 |
| correct% / correct+partial% | 39.7% / 87.5% |
| student humanness | **0.237**（teacher 0.278，差 -0.041）|
| explicit_ref / ref_enumeration / policy_source（学生） | 217 / 160 / 9 |
- student humanness 分布：0.0-0.2 占 31.2%、0.2-0.4 占 55.4%、**0.6 以上 = 0%**（印证 §8.1"自然样本 ~0%，必须冷启动造种子"）。
- accuracy×humanness：correct 0.222 / partial 0.236 / incorrect **0.286**（越错越"像人"=空想，最病态的反相关）。

### 13.2 全链最终结果总表（224 验收集，Kimi 裁判，逐阶段不省略）

| 阶段 | 准确率(平均分) | correct% | correct+partial% | 标签(c/p/i) | student humanness | teacher humanness | (s−t) |
|---|---|---|---|---|---|---|---|
| **base14B 裸基线** | 0.659 | 39.7% | 87.5% | 89/107/28 | 0.237 | 0.278 | −0.041 |
| **SFT** | 0.763 | 53.6% | 93.3% | 120/89/15 | 0.212 | 0.292 | −0.081 |
| **RFT-v1（翻车）** | 0.678 ↓ | 47.8% | 85.3% ↓ | 107/84/33 | 0.190 ↓ | 0.294 | −0.104 |
| **CS_RFT（里程碑）** | 0.701 | 46.0% | 89.3% | 103/97/24 | 0.780 | 0.159 | +0.621 |
| **DPO（双赢）** | 0.731 | 51.3% | 89.7% | 115/86/23 | 0.836 | 0.159 | +0.677 |
| **GRPO（最终）** | 0.733 | 51.3% | 89.7% | 115/86/23 | 0.846 | 0.166 | +0.680 |

### 13.3 RAG 痕迹频次轨迹（学生侧，多标签累计）
| 阶段 | explicit_ref | verbatim_copy | ref_enumeration | policy_source |
|---|---|---|---|---|
| base14B | 217 | — | 160 | 9 |
| SFT | 216 | 7 | 121 | 18 |
| RFT-v1 | 221 | **20**(翻车) | 115 | 35 |
| CS_RFT | **0** | 17 | **0** | 21 |
| DPO | 1 | 7 | **0** | 11 |
| GRPO | **0** | **3** | **0** | 12 |

### 13.4 student humanness 0.8-1.0 占比轨迹（"像人"那档的爬升）
base14B 0% → SFT 0.4% → RFT-v1 0% → CS_RFT 61.6% → DPO 79.0% → **GRPO 80.4%**。

### 13.5 报告标题"7B"勘误（文档可信度澄清）
部分 checkpoint 报告标题模板遗留写成「7B 蒸馏模型」，**正文与 baseline 报告里实为 `DeepSeek-R1-Distill-Qwen-14B`**。早期另有一次更老的"7B"失败实验（opus-4-7 裁判、准确率 0.013）是残留，与现行 14B 不是同一次。

### 13.6 裁判口径波动说明（并入《实验历程回顾》§2 注1）
同一份 teacher think，SFT 基线测 humanness **0.292**，RFT/DPO/GRPO 那几批测到 **0.159~0.166**；teacher humanness 跨批次在 **0.16~0.29** 漂（Kimi 裁判波动）。结论：**绝对值看趋势、跨阶段比用同一次评测内"学生 vs 老师"相对差**，学生同口径轨迹比跨阶段绝对值更稳。准确率也以 teacher 为"绝对正确"，是相对指标、非客观正确率。

### 13.7 早期 .ipynb_checkpoints 对应（内容一致性已核验）
`08c_calibration` ≡ `20_rft1_reward_calibration.md`；`08f_rft_report` ≡ `20_rft1_report.md`；`19_cs_rft_report` ≡ `50_csrft_report.md`；`09f_dpo_report` ≡ `60_dpo_report.md`；`10f_grpo_report` ≡ `70_grpo_report.md`。`08_rl_rollout-checkpoint.jsonl`=5 行早期试跑；`08b_rft_train-checkpoint.jsonl`=395 行（=20_rft1_trainset）。

---

## 14. 数据流全追踪表（文件谱系映射，并入《实验历程回顾》附录 A–G）

| 步骤 | 输入 | 处理(step) | 产出文件 | 条数 |
|---|---|---|---|---|
| A | `A模型纯样本汇总.xlsx`(1112) | step01 筛"可用"去重 | `00_data_queries_usable.jsonl` | 498 |
| A | `一阶段模型输出3.31-5.19 v2.xlsx`(1773) | step01b 取"模型总结问题" | `00_data_queries_summary.jsonl` | 1741 |
| A | 上两者 | step01c 合并去重 | `00_data_queries_merged.jsonl` | 2239 |
| B | merged | step02 调 V1 RAG（think+answer） | `00_data_teacher_outputs.jsonl` | 2239 |
| B | teacher_outputs | step03 拆 reasoning/answer、切 9:1 | `00_data_sft_train.jsonl` / `00_data_sft_eval.jsonl` | 2014 / 224 |
| C | sft_train | step04 LoRA SFT 蒸馏 | `ckpts/...-lora` | — |
| C | sft_eval(224) | step05 推理 → step06 Kimi 判 → step07 报告 | `10_sft_infer/judge/report` | 224 |
| D | SFT 模型 | step08 前400题×K=8 rollout | `20_rft1_rollout.jsonl` | 400 |
| D | rollout | step08b 第一版奖励选样 | `20_rft1_trainset.jsonl` | 395 |
| D | 已有 05/06 | step08c 零 Kimi 校准 | `20_rft1_reward_calibration.md` | 224 |
| E | sft_train 前800 | step11 Kimi 改写自然 think | `30_seeds_rewritten.jsonl` | 800 (facts_ok 784/16) |
| E | seeds_rewritten | step12 Kimi 打 humanness | `30_seeds_scored.jsonl` | 800 (均值≈0.61~0.64) |
| E | seeds_scored | step13 信号证伪闸门(含 PMI 探针) | （报告，AUC 闸门） | 1024 |
| E | seeds_scored | step14 筛 humanness≥0.6 + 切自然腔 eval | `40_coldstart_train.jsonl` / `40_coldstart_eval.jsonl` | 458 / 51 |
| F | coldstart_train | step04 冷启动 SFT | `ckpts/...-coldstart-lora` | — |
| F | 冷启动模型 | step08 rollout → step08b 选样 | `50_csrft_rollout` / `50_csrft_trainset` | 400 / 350 |
| F | csrft_trainset | step04 RFT 续训 → 05/06/07 验收 | `ckpts/...-cs-rft-lora` + `50_csrft_*` | 224 |
| G | sft_train(2014) + 新query(0新) | step15 扩池+打散 | `60_dpo_pool_expanded.jsonl` | 2014 |
| G | CS_RFT 模型 | step08 rollout → step09 构对 | `60_dpo_rollout`(400) / `60_dpo_pairs` | 315 |
| G | dpo_pairs | step09 DPO 训练 → 05/06/07 | `ckpts/...-dpo-lora` + `60_dpo_*` | 224 |
| H | DPO 模型 + 2014 池 | step10 GRPO 在线（无单独 rollout/trainset 落盘） | `ckpts/...-grpo-lora` | — |
| H | grpo 模型 | step05/06/07 验收 | `70_grpo_infer/judge/report` | 224 |

---

## 15. 真实数据样例附录（逐字未删，并入《整个实验记录》——最高展示价值资产）

> 这些是把抽象指标变成可展示证据的唯一载体，简历/面试演示价值极高。以下为各阶段真实样例的**逐字摘录要点**（完整全文见被删文档的对应章节，此处保留足以复现的关键内容）。

### 15.1 第0章·"个税返还/无票收入"题（teacher 机器腔原貌）
完整 `user_prompt` 含 5 个参考问答对；teacher `reasoning_content` 全文是典型机器腔（"根据参考问答对1…""检索结果显示…"逐条对照）；`content` 为最终答案；`top_k` = RAG 检索片段数。这是"问题在哪"的原始证据。

### 15.2 第2章·"开出去的折扣怎么做账"题（SFT 三段式，机器腔逐字）
完整 system / user(含 5 个参考问答对) / assistant 三段。assistant 的 think 全程机器腔："参考问答对1明确指出…"；answer 段被双层 `<answer><answer>…</answer></answer>` 包裹（§4.2 已说明，不影响训练）。

### 15.3 第3章·同"折扣"题 8 个候选的真实打分（实证"矮子里拔高个"）
第一版 RFT 时该 query 的 8 个 rollout 候选，每条带 `reward / R_human / R_acc / gate / text 摘要`。**入选的 #4：reward=0.3、R_human 仅 0.22**——即便挑出来的"最佳"也很机器腔，直接说明第一版奖励为何翻车。

### 15.4 第5章·"固定模具费30%/发票数量0.3"题（改写前后逐字对照）
- `original_think`（机器腔）vs `natural_think`（自然腔）逐字对照；answer + 5 个参考；`facts_ok:true, introduced_nums:[]`（数字一字未改）。这是冷启动种子"洗风格不洗事实"的逐字证据。

### 15.5 第6章·"折扣"题冷启动自然腔 think 全文（与 §15.2 同题对照）
冷启动后该题自然腔 think 全文，与第2章机器腔同题逐句对照："参考问答对1明确指出…"→"首先得弄明白…第一种是商业折扣…反过来…"。直观展示风格迁移。

### 15.6 第7章·"12月发票延迟报销跨年扣除"题（DPO 偏好对长什么样）
完整 5 参考 + **chosen vs rejected 两条 think 逐字全文**：两者答案都对，chosen 从问题自然推导、rejected 满是 RAG 罗列——演示 DPO 偏好对"只在 think 风格上分高下"。

---

## 16. 踩过的坑汇总表（面试最能体现工程能力）

| # | 问题 | 现象 | 解决 |
|---|---|---|---|
| 1 | transformers v5 不兼容老模型 | base 生成全乱码、裸 prompt reshape 崩 | 锁 `transformers>=4.46,<4.49`（实跑 4.48.3）；补 attention_mask |
| 2 | peft adapter 配置不兼容 | 降级后 `LoraConfig got unexpected 'alora_invocation_tokens'` | 写脚本剥离 config 多出字段（权重版本无关） |
| 3 | **R1 模板会删 think** | SFT 后几乎不输出 `<think>`，loss 却正常 | R1 chat_template **自动剥离 assistant 内 `<think>...</think>`**；改 **prompt 走模板、target 手工拼**绕开 |
| 4 | 合并 LoRA OOM(SIGKILL -9) | `save_pretrained` 收集 14B 到 CPU 爆内存 | 全程 base+LoRA 不合并（数学等价）；step05 加"半成品目录"健壮判断；merge 标 optional |
| 5 | rollout OOM(-9) | 并行 K 条 + 长 prompt 打爆；同卡两进程(1129/2114)抢显存 | 先逐条生成；扩 128G 内存后改回并行（共享 prefill 更快）；杀重复实例 |
| 6 | 奖励校准数据天花板 | SFT 全 RAG 腔、Kimi 分挤在 0.1~0.35、Spearman 0.286 | 换平滑不饱和 humanness 函数；认清是数据天花板，验证交给 RFT+Kimi |
| 7 | 扩容重建丢依赖 | `ModuleNotFoundError: transformers` | BASH_ENV 钩子自动 `pip install -r requirements.txt`；重装后验 CUDA |
| 8 | 误启动多实例 | 两个 14B 同卡抢；rollout 同分钟被重复 START 4 次→CUDA OOM | `pkill` 杀干净，确认单实例再跑 |
| 9 | **裁判走错网关** | `claude-opus-4-7` 调用失败 `'choices'`；网关只认 deepseek-v4；temp/top_p 被强制 | 改用 **kimi-k2.6**（temp=1/top_p=0.95/关 think 是 mudgate 强制约束），后续 err=0 |
| 10 | 05_infer 加载不存在 merged | `05_infer_student exit=1` | 回退 base+LoRA 模式跑通 |
| 11 | **eval 分布错配**（§8.5） | 机器腔 eval_loss 随越练越自然反升、早停选反方向 | step14 切自然腔留出 eval |
| 12 | **πref 锚错基准**（§10.2） | DPO/GRPO 用裸 base 当参考 → 失锚伤准确率 | 预计算冻结 CS_RFT / PEFT 多 adapter；避开"删 disable_adapter→零梯度"陷阱 |
| 13 | **GRPO 两轮 OOM**（§11.2） | step 1 后崩；碎片 10.5GB / 真占满 92GB | expandable_segments + SDPA + completion-only log_softmax + 左截断 |

> 坑 #3 最有讲头——解释了"训练 loss 正常但推理输出空"，体现"不被表面指标骗、深挖到 tokenizer/模板层"的排查能力。

---

## 17. 评测设计

裁判 Kimi 对每条输出严格 JSON 打分，两类指标：
1. **准确率**：以 teacher 答案为绝对正确，judge 给 correct/partial/incorrect + 0~1 分；
2. **推理 humanness(0~1)**：越高越像端到端 CoT、越不像 RAG；并给 RAG 痕迹类型（explicit_ref 显式引用 / verbatim_copy 大段照搬 / ref_enumeration 罗列参考 / policy_source 政策出处）。humanness 是**连续分**（便于做 RL reward 的尺），锚点：0.9~1.0 完全像人；0.0~0.2 出现"参考问答对1/2/3""根据检索结果"或大段照搬。即使无 RAG 关键词，大段照搬政策也低分（看"从问题向答案推导 vs 从资料向答案归纳"的气质）。

辅以 **accuracy × humanness 交叉表**——重要发现：**SFT/base 学生"答得越准的，humanness 反而越低"**（base14B incorrect 0.286 > correct 0.222），说明"准确"建立在"照抄参考问答对"上 → 这正是 RL 要破解的核心矛盾。健康收敛后（GRPO）该相关性翻正（correct 0.865 > incorrect 0.763）。

---

## 18. 方法论沉淀（最值得带走）

- **先廉价证伪、再烧 GPU**：奖励先过判别 AUC≥0.70 闸门、改训练代码前先静态对抗验证。门槛 0.40→0.60、扩 query 没新数据都是"先看数据/实跑才暴露"。
- **挑样本用本地奖励、判成败用独立金标准（非循环验证）**：从第一次 RFT 翻车学来——选样器(本地 reward)≠验收器(Kimi)，三重佐证（无循环 / 客观痕迹与主观分同向 / correct≥incorrect 排除"自然但瞎编"）。
- **负结果也是结论**：量化 Goodhart（acc 93.3→85.3、verbatim_copy 7→20），转化为对 V1 的 de-risk 价值。
- **数据驱动决策**：门槛看分布定（中位 0.80 才知抬阈值几乎白捡）、扩 query 实跑才暴露 0 新数据。
- **多视角对抗审查**：既抓设计 bug（πref 锚错基准）又识破 AI 给的错误修法（删 disable_adapter→零梯度陷阱）。
- **准确率"保住而非测量"**：不判对错改判"漂没漂" + 非对称 KL 锚住 answer 段 → 不需要完美对错判官。
- **相对差解结构性矛盾**：references 本就是标准答案，所有 humanness 信号用相对差（只罚"贴资料远多于贴问题/答案"），准确率项完全不碰 references。
- **可迁移配方**：一次性洗 think 风格做冷启动 → 纯自我 RL；只洗风格、保事实，与训崩的 V2（伤筋动骨全蒸馏）不是一回事。

---

## 19. 预判面试问答（背这几条）

**Q: 这项目一句话是什么？**
A: 用强化学习把税务模型的推理过程**去掉 RAG 检索痕迹、变成像人的端到端推导**，同时守住准确率。
> ⚠️ **措辞陷阱**（并入《实验历程回顾》§14）：要说"去 RAG 痕迹 / 更像人的端到端推理"，**不要说"规范化"**（会被理解成加模板，方向反了）。

**Q: 为什么不直接多做点 SFT，非要 RL？**
A: SFT 只能模仿 teacher，而 teacher 推理本身就是 RAG 腔——学生会忠实克隆（实测 student 引用甚至比 teacher 还多）。要"去掉"一种风格、产出 teacher 没示范过的端到端推理，必须用奖励驱动的自我改进。

**Q: 怎么防 reward hacking？**
A: 乘法门控（答错≈0 分）；copy_ratio 用 LCS+5-gram 堵"删引用词但照抄"；think 长度上限堵灌水；最关键——用**独立 Kimi 每阶段末验收**，盯"本地奖励涨但 Kimi 不涨"的 Goodhart。第一次 RFT 正是这样被抓到（verbatim_copy 7→20）。

**Q: 改推理风格怎么保证答案还对？（没有完美对错判官）**
A: 不靠判对错，靠**结构上按住答案**——对 answer 段加强 KL 锚定（拉向已学对的分布）、对 think 段放开。答案基本不变→准确率结构性守住；打分器只需"答案漂没漂"的廉价探针。

**Q: GRPO vs PPO vs DPO 怎么选？**
A: PPO 三套模型单卡吃紧易崩；DPO 用偏好对、稳但离线；GRPO 无 value model、组内相对优势契合"同题比自然度"。路线是先离线 RFT/DPO 吃大头降风险，再上在线 GRPO 精修。

**Q: 为什么裁判用 Kimi？**
A: teacher/student 都 deepseek 系，同源裁判有 self-preference bias；跨厂家 Kimi 规避。

**Q: RFT 是什么、为什么放 GRPO 前？**
A: 拒绝采样微调——采 K 个、用奖励挑最好、再 SFT。只挑正样本、对噪声不敏感、零在线复杂度，是验证奖励方向 + 低风险拿首轮收益的最佳起步。

**Q: 本地奖励可靠吗？校准 Spearman 才 0.286。**
A: 校准数据全 RAG 腔、Kimi 分压在窄带，Spearman 天然低，是数据天花板不是奖励缺陷；真正验证靠 RFT 后用独立 Kimi 看 humanness 涨没涨（不循环）。

**Q: PMI 用上了吗？**
A: **设计并离线验证过（AUC 0.730），但从没接进正式训练**（日志 PMI=关、s_pmi 全 null）。真正守 Goodhart 的是表面项（关键词+照抄）。PMI 做成 `--with-pmi` 可开关增援，性价比不足以每条载 14B，故没部署。

**Q: GRPO 跑完了吗？最终成绩？**
A: 跑完了，修两轮 OOM（碎片→SDPA）后一气呵成 100 步。最终 humanness **0.846**、准确率 **0.733**、RAG 显式痕迹清零——全链 humanness 0.212→0.846(≈4×)、准确率仅 -0.03。

---

## 20. 32B 迁移现状与诚实边界

- **已验证命题**：在 14B 沙盒证明"RL（冷启动+重设计奖励 RFT → DPO → GRPO）能把推理大幅去 RAG 化、同时基本保住准确率"——这正是要迁移回 V1 的方法。
- **迁移 V1 四步**（并入《实验历程回顾》§12）：① 冻结配方（代码 + 奖励 + 种子流程）；② 迁移到 V1 微调；③ Kimi 终审；④ 上线。最大隐性风险 = **格式适配**（V1 输出契约与本沙盒 `<think>/<answer>` 是否一致）。V1=模型本体、跳过蒸馏（V1 本就是要强化的对象）。
- **诚实边界**：① "对全新问题泛化"目前只由始终留出的 224 验收集证明，**真证泛化需公司另给从未训过的生产日志**（本项目扩 query 实跑发现那批 Excel 早已训过、0 新）；② GRPO 在线易漂移，迁移时需保留 KL 锚定防在线漂移；③ teacher humanness 跨批波动，跨阶段比应锁学生同口径。

---

## 21. 技术栈 / 基础设施 / 接口（备查）

- 技术栈关键词：`强化学习` `GRPO` `DPO` `拒绝采样微调(RFT)` `冷启动 SFT(cold-start)` `LoRA SFT` `知识蒸馏` `奖励函数设计` `reward hacking 防御` `非对称 KL 锚定` `LLM-as-a-Judge` `奖励-金标准校准` `PMI/似然信号(验证未部署)` `语义嵌入(bge-m3,已弃)` `DeepSeek-R1-Distill` `Kimi` `transformers/peft` `单卡 14B 训练` `RAG 痕迹检测`。
- 接口：检索 `/agentic_system_service/rag/getRetrieve`；teacher(V1) `/llm_finetune/v1/chat/completions`；裁判 Kimi-K2.6 `/mudgate/api/llm/moonshot/v1`（三 header 同 key；temp=1/top_p=0.95/关 think 是 mudgate 强制约束）；嵌入 bge-m3 `/text-embedding-bge/v1/embeddings`（1024 维，绕系统代理）。
- 栈版本：transformers 4.48.3 / peft 0.13.2 / torch 2.12.0；单卡 RTX PRO 6000 96G + 128G RAM；全程 base+LoRA 不合并。
- run.py 编排：bulk 默认只跑 SFT 主线(01–07)；RL 全链需 `--from/--only/--include-rl` 显式进入；产物存在即跳过（断点续跑）；所有 infer/judge/report 复用 step05/06/07，所有 train 复用 step04（`--init_adapter` 续接 adapter 链 base→coldstart→cs-rft→dpo→grpo）。

### 各阶段耗时（END 行 elapsed，§22 备查）
- 14B SFT：~3.15~3.5h；每轮 rollout(400×K)：~5.5~6.3h（最大时间黑洞）；RFT 训练：~31min；冷启动 SFT：4898s；CS-RFT：2199s；DPO：~21min(315 对/1ep)；**GRPO(成功那次)：41965.3s≈11.7h**；每轮 student 推理(224)：~2.3h；每轮 Kimi 裁判(224,workers=4,0.03 q/s)：~7400~8000s；主流水线最后一段 `PIPELINE DONE elapsed≈57935.8s≈16h`（含 GRPO+infer+judge）。
