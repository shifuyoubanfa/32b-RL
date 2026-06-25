# 106 两套奖励函数方案：交给 Fable5 裁决稿

> 日期：2026-06-13  
> 目的：把 Codex 方案与用户新方案并列整理，交给 Fable5 做技术裁决与最终设计。  
> 约束：强化学习路线必须继续走，最好保留 DPO + GRPO；目标是让 32B 模型的 `think` 不那么机械、不暴露 RAG 痕迹，同时答案正确率不能被破坏。

## 0. 当前共识

项目原始数据：

```text
总数据：2239
训练池：2015
固定验证/验收集：224
```

当前 derag_v4 已证明：

- 工程链路可以跑通：SFT/RFT → DPO → GRPO → 报告；
- 旧奖励/门禁会出现指标迁移：显式 RAG 痕迹减少，但复制、答案分、事实分可能退化；
- Kimi 不能作为唯一硬门禁或唯一在线 GRPO reward；
- 现在需要重新设计奖励函数与裁判体系。

最重要的边界：

```text
224 验证集只用于基线定标、验收与裁判校准，不能进入训练数据。
训练数据仍从 2015 训练池产生。
```

## 1. Codex 方案：正确锚点 A + 不退化约束 + 去 RAG 优化

### 1.1 核心思想

给每道题一个可信锚点 A，A 可以是：

- 标准 gold answer；
- 原始 V1 正确输出；
- RFT merged base 的人工/规则确认正确输出；
- 其他高置信正确答案。

然后所有候选 B 都先回答一个问题：

```text
已知 A 是正确锚点，B 是否相对 A 发生答案/关键事实退化？
```

只在 B 没有退化时，才继续优化 `think` 的去 RAG 风格。

### 1.2 判断 B 是否退化

候选 B 分三态：

```text
NO_DEGRADE：明确没有退化
DEGRADE：明确退化
UNKNOWN：无法可靠判断
```

判断维度：

- 最终结论是否变化；
- 税种、纳税主体是否变化；
- 税率、金额、时间、门槛是否变化；
- 适用条件、例外条件是否遗漏；
- 是否新增与 A 冲突的事实；
- 是否把“通常/一般/如无特殊情况”改成绝对表述；
- 是否删除了会影响结论的关键限定。

处理方式：

| 状态 | 训练处理 |
|---|---|
| `NO_DEGRADE` | 可以进入去 RAG 风格比较 |
| `DEGRADE` | 负样本或丢弃 |
| `UNKNOWN` | 不进入主训练信号，只进审计 |

### 1.3 DPO 构造

只使用这种 pair：

```text
同一道题；
chosen/rejected 都 NO_DEGRADE；
两者最终答案与关键事实等价；
chosen 的 think 更少 RAG 痕迹；
rejected 的 think 更机械或更像 RAG 复述。
```

禁止使用这种 pair：

```text
chosen 看起来更自然但答案可能变了；
rejected 答案正确但有 RAG 痕迹。
```

核心排序：

```text
正确但机械 > 自然但错误
正确且干净 > 正确但机械
```

### 1.4 GRPO 奖励

每题采样 K 个候选。

先过退化门：

```text
DEGRADE -> -1
UNKNOWN -> mask/skip，不产生风格梯度
NO_DEGRADE -> 继续计算去 RAG 分
```

只有同组内至少两个 `NO_DEGRADE` 候选，且去 RAG 分有差异时，才产生 GRPO 更新。

去 RAG 分拆成多个子项：

- 显式“参考资料/资料显示/根据文档”等；
- 资料一、资料二等枚举；
- 长段复制；
- 按资料顺序机械复述；
- 机械模板；
- 正常法规引用误伤。

最终不能只看 `trace_total`，而要防止：

```text
explicit_ref 降低，但 verbatim_copy 上升。
```

### 1.5 优点

- 目标清晰：先不变错，再去 RAG；
- 与 104/105 调研一致；
- Kimi 做相对退化判断，比做绝对正确性判断更可行；
- UNKNOWN 退出训练，降低错误信号污染。

### 1.6 风险

- 需要可信锚点 A；
- 如果 A 本身有噪声，退化判断会继承 A 的噪声；
- Fact Contract 或锚点抽取成本较高；
- 覆盖率可能较低，需要大 K 或更多训练池数据补足。

## 2. 用户方案：V1 自一致基线 + V1 概率/评分守答案 + 规则/Kimi 管 think

### 2.1 核心思想

用户认为：当前项目中，答案正确性的最权威来源应该是原始 V1 模型本身。

V1 本身对同一输入存在自然采样波动。因此不能把“与某一次 V1 输出不同”都视为错误。更合理的是：

```text
先测量 V1 自身在固定验证集上的答案分布与 think 分布；
把 V1 的均值和方差作为基线噪声；
后续 SFT/RFT/DPO/GRPO 的答案只要落在 V1 基线统计范围内，就认为没有显著退化；
think 则用规则函数 + Kimi 复核来优化去 RAG。
```

注意：用户后来修正了流程边界：

```text
V1 的答案/think 基线评分应在 224 验证集上做；
训练样本仍应从 2015 训练池产生，不能把 224 改写后拿去训练。
```

### 2.2 Stage 0：在 224 验证集上建立 V1 答案基线

对 224 验证集中的每个 query + RAG 输入，调用 V1 采样 6 次：

```text
每个 query -> 6 个 V1 输出
每个输出包含 think + answer
```

然后有两种答案基线方案。

#### 方案 0A：V1 自评打分

让 V1 对这 6 个答案进行评分，得到：

```text
score_1 ... score_6
均值 μ_answer
方差 σ_answer
```

后续模型生成 answer 时，让 V1 再评分。如果分数落在可接受区间内，则认为答案没有显著退化。

可能的区间：

```text
score_new >= μ_answer - k * σ_answer
```

其中 `k` 由 Fable5 决定。

#### 方案 0B：V1 条件生成概率

不让 V1 显式打分，而是计算 V1 对答案的条件 logprob。

类似 DPO 中计算 reference logprob：

```text
logP_V1(answer | query, RAG, optional think/context)
```

为了避免长度偏差，应使用长度归一化：

```text
mean_token_logprob
```

对 6 个 V1 答案得到：

```text
μ_logprob
σ_logprob
```

后续模型生成 answer 时，如果：

```text
logprob_new >= μ_logprob - k * σ_logprob
```

则认为答案处于 V1 自身可接受分布内。

### 2.3 Stage 0：在 224 验证集上建立 think 规则基线

对 V1 采样得到的 6 个 think，用规则函数打分。

规则函数只做一件事：

```text
判断 think 是否有机械 RAG 痕迹。
```

规则子项可包括：

- 显式“根据资料/参考资料/资料显示”；
- 资料枚举；
- 长段复制；
- 按资料顺序复述；
- 无必要的政策罗列；
- 模板化重复；
- 正常法规引用豁免。

每个 query 得到 6 个 think 分数：

```text
rule_score_1 ... rule_score_6
μ_think_rule
σ_think_rule
```

### 2.4 Kimi 校正规则函数

对规则判低分的 think，把以下信息交给 Kimi：

```text
原 query
RAG 内容
原 answer
原 think
规则打分结果
规则认为低分的原因与触发子项
```

Kimi 只做两件事：

1. 判断规则批评是否成立；
2. 如果成立，改写 think；如果不成立，拒绝改写并说明原因。

也就是：

```text
规则：这个 think 有 RAG 痕迹，原因是 X/Y/Z。
Kimi：我是否认同这个判断？
    - 认同：给出改写版 think
    - 不认同：拒绝改写，标记该规则子项可能误伤
```

这样可以得到：

- 哪些规则子项与 Kimi 一致；
- 哪些规则子项误伤正常推理；
- 哪些规则子项稳定可靠；
- 哪些规则子项应该降权或删除。

这里的 224 流程只用于**定标规则与裁判**，不用于训练。

### 2.5 Stage 1：训练池生成 Kimi 改写 SFT 数据

在 2015 训练池上复用 Stage 0 冻结后的规则/Kimi 流程。

对 V1 或当前基线模型输出：

```text
原 think + 原 answer
```

只让 Kimi 改写 think，不改写 answer。

训练样本形态：

```text
input: query + RAG
target: Kimi 改写后的 think + 原 V1 answer
```

这样 SFT 学到的是：

```text
答案仍保持 V1 风格/结论；
think 去掉机械 RAG 痕迹。
```

### 2.6 Stage 1 验收：SFT 在 224 上评测

让 SFT 模型在 224 验证集上生成 think + answer。

验收两件事：

1. **think 是否变好**

用冻结规则函数打分：

```text
think_score_sft > V1 基线 think_score
```

2. **answer 是否没下降**

用 V1 评分或 V1 logprob 判断 SFT answer 是否仍在 V1 基线统计范围内：

```text
score/logprob_sft >= μ_answer - k * σ_answer
```

如果 think 分明显提升，且 answer 分/概率没有越出下界，则 Stage 1 通过。

### 2.7 Stage 2：RFT 冷启动 / 自采样再筛选

让 SFT 模型在训练池上自采样，每个 query 采样多个候选：

```text
query -> K 个候选
每个候选包含 think + answer
```

先用 V1 对 answer 做严格过滤：

```text
如果 answer 的 V1 score/logprob 超出可接受区间：
    当前候选不要
否则：
    进入 think 评分
```

然后对 think：

- 规则函数打分；
- Kimi 对高风险或边界样本复核；
- 只保留答案安全、think 去 RAG 分高的候选。

用这批数据继续微调，得到 RFT 模型。

验收同 Stage 1：

```text
224 上 think 分是否提升；
answer 的 V1 score/logprob 是否仍在基线范围；
```

如果通过，则说明：

```text
SFT/RFT 模型可以产生可学习的强化信号；
V1 + 规则 + Kimi 的奖励体系可以识别强化信号。
```

### 2.8 Stage 3：DPO

让 RFT 模型自采样一批候选。

用奖励体系构造正负样本对：

1. 先用 V1 score/logprob 判断 answer 是否在安全区间；
2. 安全区间内，再比较 think 规则分；
3. 对规则/Kimi 一致的样本构造 DPO pair。

用户提出一个重要思想：

```text
V1 作为 answer 的 reference；
RFT 作为 think 的 reference。
```

工程上可以理解为：

- **答案侧**：V1 负责评估 answer 是否处在原始正确分布内；
- **think 侧**：RFT 作为当前可信生成基线，DPO 不应让 think 偏离到过短、过空、或不可解释；
- **DPO π_ref**：技术实现上可能仍需用 RFT 作为 DPO reference model，但 reward/筛选里显式加入 V1 answer reference。

这部分需要 Fable5 决定最合理的数学与工程实现。

### 2.9 Stage 4：GRPO 精细化雕刻

GRPO 不是大开大合地继续追分，而是做精细化雕刻。

在 DPO 结束后，先在 224 上评测 DPO 输出：

```text
rule_score(DPO think)
Kimi 对 rule_score 的认可/不认可
V1 对 answer 的 score/logprob
```

重点分析：

```text
哪些规则子项与 Kimi 高度一致；
哪些规则子项 Kimi 经常不认同；
哪些规则子项导致误伤；
哪些规则子项只是大开大合的粗规则。
```

用户提出的方向：

> 只要某个规则子项不是非常稳定，就不要把它作为 GRPO 强奖励。  
> 如果某个 7 分是由很多子项组成，而 Kimi 对其中若干子项经常不认同，就分析不认同样本中哪些子项占比最高，找到共性后删除或降权这些规则。

可落地成：

```text
规则子项稳定性审计：
    对每个子项统计：
        - 触发次数
        - Kimi 认可率
        - Kimi 不认可率
        - 不认可样本中的答案风险
        - 与最终人工/验证指标的一致性

只保留：
    Kimi 高认可
    低误伤
    与最终去 RAG 改善相关
    不伤害答案质量
的规则子项进入 GRPO reward。
```

GRPO reward 应是：

```text
先 V1 answer 安全门；
再稳定规则子项分；
Kimi 只做离线校准或边界复核，不做在线唯一 reward。
```

这一步用户还未完全想通，需要 Fable5 进一步设计：

- 是否直接用稳定规则子项做 GRPO；
- 是否蒸馏一个规则/Kimi 一致性 scorer；
- 如何防止规则被模型再次投机；
- 如何在保留 DPO/GRPO 的前提下证明增益。

## 3. 两套方案的共同点

两套方案都反对：

```text
直接优化抽象 humanness；
直接让 Kimi 作为唯一在线 reward；
只看 trace_total 下降；
答案退化还能因为 think 更干净而通过。
```

两套方案都支持：

- 答案正确性优先；
- think 去 RAG 其次；
- Kimi 做裁判/复核，不直接做唯一奖励；
- DPO pair 必须更干净；
- GRPO reward 必须保守；
- 224 验证集只做评测/校准，不进训练。

## 4. 两套方案的关键差异

| 问题 | Codex 方案 | 用户方案 |
|---|---|---|
| 答案正确性锚点 | 显式正确答案 A / Fact Contract | V1 自一致分布 |
| 答案质量判断 | B 是否相对 A 退化 | B 是否落在 V1 score/logprob 基线范围 |
| 事实保护方式 | NO_DEGRADE / DEGRADE / UNKNOWN | V1 自评或 logprob 均值方差 |
| think 优化 | NO_DEGRADE 后再优化去 RAG | V1 答案安全后，规则/Kimi 优化 think |
| Kimi 角色 | 相对退化判断 + 去 RAG 复核 | 判断规则是否误伤 + 改写 think + 边界复核 |
| 最大优点 | 语义退化定义清晰 | 利用 V1 作为项目内最权威模型，避免外部裁判绝对判断 |
| 最大风险 | A/Fact Contract 构建成本高 | V1 logprob/自评分可能偏向 V1 风格，不等于真实正确性 |

## 5. Fable5 需要裁决的问题

请 Fable5 不要只给宏观建议，而要给可以落地的完整方案。

### 5.1 V1 能否作为答案正确性的主裁判？

需要判断：

1. V1 自评答案是否可信；
2. V1 answer logprob 是否能代表答案质量；
3. logprob 是否会偏向短答案、常见答案、V1 自己的口癖；
4. 是否需要结合 A/B 退化判断，而不是只看概率；
5. V1 基线方差应该按每题算、全局算，还是分题型算。

### 5.2 224 验证集如何使用才不泄漏？

用户希望在 224 上做 V1 基线定标和规则/Kimi 校准。

需要 Fable5 明确：

- 哪些统计可以在 224 上做；
- 哪些决策会污染最终验收；
- 是否需要再切 development/eval/sealed final；
- 若没有更多数据，如何避免反复看 224 过拟合。

### 5.3 Kimi 改写 think 的训练数据如何生成？

需要明确：

- 训练池 2015 上如何筛需要改写的 think；
- Kimi 拒绝改写时样本怎么处理；
- Kimi 改写后是否必须再过答案一致性门；
- 是否保留原 V1 answer；
- SFT 训练比例、replay、去模板化如何做。

### 5.4 V1 answer reference + RFT think reference 如何实现？

用户提出：

```text
V1 作为答案 ref；
RFT 作为 think ref。
```

需要 Fable5 设计数学与工程实现：

- DPO 的 π_ref 用谁；
- V1 logprob 是筛选门、reward 项，还是额外 KL；
- think 如何对 RFT 保持；
- 是否需要拆 answer token 与 think token 的 loss 权重；
- 是否需要 answer lock。

### 5.5 GRPO 的稳定规则选择如何做？

用户希望：

```text
只保留 Kimi 高度认可、非常稳定的规则子项进入 GRPO；
不稳定规则删除或降权。
```

需要 Fable5 明确：

- 稳定性的阈值；
- Kimi 认可率怎么计算；
- 是否需要人工抽查；
- 规则子项怎么防投机；
- GRPO reward 的最终公式；
- GRPO 失败后是否回退 DPO。

## 6. 一个可能的融合方向

Fable5 可以考虑把两套方案融合：

```text
第一层：V1 自一致答案分布，作为项目内答案安全先验；
第二层：Codex 的 A/B 相对退化判断，抓 V1 概率看不出的语义偏移；
第三层：规则函数判断显式 RAG 痕迹；
第四层：Kimi 判断规则是否误伤，并改写/复核边界样本；
第五层：只把高置信 no-degrade + high-derag-margin 样本送入 DPO/GRPO。
```

训练链路可能是：

```text
V1 224 自一致定标
-> 规则/Kimi 在 224 上校准
-> 2015 训练池生成 Kimi think-only 改写 SFT 数据
-> SFT 训练与 224 验收
-> SFT/RFT 自采样，V1 answer 安全门 + 规则/Kimi think 门
-> RFT/SFT 再微调
-> RFT 自采样构造高置信 DPO pair
-> DPO
-> DPO 输出上审计规则/Kimi 稳定性
-> GRPO 使用稳定规则子项精细化雕刻
-> 最终 224 或 sealed set 验收
```

## 7. 交给 Fable5 的最终任务

请 Fable5 基于本文件、`104_奖励函数与裁判体系_业内调研.md`、`105_奖励函数落地执行方案.md`、`deep-research-report (1).md`，做一次完整技术裁决：

1. 判断 Codex 方案与用户方案各自是否成立；
2. 明确 V1 自评/logprob 能否作为答案质量基线；
3. 设计最终融合方案；
4. 给出每阶段输入、输出、门禁、失败回退；
5. 给出 DPO 和 GRPO 的 reward/pair 构造细节；
6. 给出如何避免 224 验证集泄漏和过拟合；
7. 给出可以直接让 Codex 落地的工程图纸。

