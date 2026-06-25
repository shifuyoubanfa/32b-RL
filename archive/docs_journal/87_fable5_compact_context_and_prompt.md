# fable5 高密度上下文包：32B RL v2 诊断与方案设计

> 用法：把本文件整体发给 fable5。不要同时塞全量聊天记录、全量日志、全量 JSONL。fable5 的任务是做研究判断和二版方案，不是复述环境。

## 你扮演的角色

你是一个资深大模型强化学习研究员、偏好数据设计专家和工程负责人。请基于下面事实，分析为什么 32B 税务模型的人性化推理 RL 阶段没有带来提升，并设计 **v2 技术方案**。方案必须能落地到现有服务器和代码，不要只说“调参”。

## 项目目标

把公司 V1-32B 税务模型的 `<think>` 从 RAG/检索腔改成自然的人类推理腔，同时不能牺牲答案准确率。

核心指标：
- `humanness` 越高越好。
- `grounded` 要保持，不能为了自然而脱离参考资料。
- `accuracy/漂移` 不能低于当前最优阶段太多。

## 当前工程与数据

- 代码：`/mnt/pfs/zhjg/code`
- 工作目录：`/home/nvme01/zhjg`
- V1 全量模型：`/home/nvme01/zhjg/V1-32B/checkpoint-1500`
- final RFT LoRA：`/home/nvme01/zhjg/ckpts/v1-32b-cs-rft-lora/v0-20260609-192005/checkpoint-38`
- RFT merged full model：`/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-merged`
- DPO merged full model：`/home/nvme01/zhjg/models/v1-32b-corrected-v1-rft-dpo-merged`
- 本轮 DPO LoRA：`/home/nvme01/zhjg/ckpts/v1-32b-corrected-v1-rftmerged-dpo-lora`
- 本轮 GRPO from RFT LoRA：`/home/nvme01/zhjg/ckpts/v1-32b-corrected-v1-grpo-from-rftmerged-lora`
- 本轮 GRPO from DPO LoRA：`/home/nvme01/zhjg/ckpts/v1-32b-corrected-v1-grpo-from-dpomerged-lora`
- DPO pairs：`/home/nvme01/zhjg/output/60_dpo_pairs.jsonl`
- GRPO data：`/home/nvme01/zhjg/output/70_grpo_data.jsonl`
- 训练环境：`/home/nvme02/conda/zhjg_rl`，Python 3.11.15，ms-swift 4.0.1，torch 2.10.0，transformers 5.2.0，trl 0.28.0。
- GRPO/vLLM 环境：`/home/nvme02/conda/grpo_env` 与 `/home/nvme02/biyh/vllm_env`，Python 3.12.13，vLLM 0.20.1+cu129。
- 硬件：8 x A800 80GB。

## corrected-v1 实验语义

历史 DPO/GRPO 曾有 `πref` 语义风险：旧 LoRA 脚本只传 `--adapters`，未传 `--ref_adapters`，reference 可能在 PEFT `disable_adapter()` 后退回原始 V1。

corrected-v1 已修正：
1. 合并 `V1 + final RFT LoRA` 得到完整 `RFT merged base`。
2. 在 `RFT merged base` 上新建 DPO LoRA，禁用新 LoRA 即回到冻结 RFT merged base。
3. 合并 DPO LoRA 得到完整 `DPO merged base`。
4. 分别从 `RFT merged base` 与 `DPO merged base` 新建 GRPO LoRA。

因此本轮所有 `πref` 都是明确的完整冻结基座，不依赖隐式双 adapter。

## corrected-v1 最终结果

| 模型 | humanness | grounded | acc | correct% | correct+partial% | Δh vs RFT基座 | Δacc vs RFT基座 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RFT merged base | 0.697 | 0.858 | 0.818 | 65.6% | 91.1% | +0.000 | +0.000 |
| DPO on merged base | 0.690 | 0.859 | 0.811 | 64.3% | 90.2% | -0.007 | -0.007 |
| GRPO from RFT merged | 0.690 | 0.856 | 0.831 | 67.9% | 92.0% | -0.007 | +0.013 |
| GRPO from DPO merged | 0.687 | 0.852 | 0.809 | 62.1% | 91.1% | -0.010 | -0.009 |

结论：链路跑通，但 DPO/GRPO 没有提升 humanness。`GRPO from RFT merged` 提升 acc，但牺牲 humanness。当前最佳仍是 `RFT merged base`。

## 关键诊断统计

### DPO vs RFT

- n=224
- mean_delta_h = -0.0074
- median_delta_h = +0.0000
- h_up=58，h_down=63
- acc_up=18，acc_down=21
- acc_up_but_h_down=0
- acc_same_but_h_down=46

解释倾向：DPO 基本没有把模型往更自然方向推，更多是轻微随机波动或弱偏好信号。

### GRPO-from-RFT vs RFT

- n=224
- mean_delta_h = -0.0074
- median_delta_h = +0.0000
- h_up=53，h_down=63
- acc_up=26，acc_down=19
- acc_up_but_h_down=3
- acc_same_but_h_down=44

解释倾向：GRPO 有一些守准/准确率收益，但没有把 humanness 往上推，甚至让部分样本更像资料归纳或照抄。

### GRPO-from-DPO vs DPO

- n=224
- mean_delta_h = -0.0029
- median_delta_h = +0.0000
- h_up=53，h_down=62
- acc_up=19，acc_down=23
- acc_up_but_h_down=2
- acc_same_but_h_down=42

解释倾向：从 DPO base 再 GRPO 没有明显收益，且 DPO 本身已经没有优于 RFT。

## DPO pair 结构审计

- pair_count=1516
- has_rejected=1516
- chosen_len_mean=821.1，rejected_len_mean=882.2
- chosen_len_median=781.0，rejected_len_median=840.0
- 当前审计只确认结构和长度；没有证明 chosen 在 Kimi humanness/grounded/accuracy 上显著优于 rejected。

重要怀疑：本轮 DPO 复用旧 `60_dpo_pairs.jsonl`，它可能不是针对 `RFT merged base` 重新 rollout/重新判分构造的。DPO 训练日志里 `rewards/margins` 多数很小，甚至有负值，`loss` 约 0.69，提示偏好信号可能弱或噪声大。

## GRPO 训练观察

本轮 GRPO：
- steps=50
- lr=5e-7
- beta=0.08
- K=8
- colocate vLLM
- ZeRO-3

训练日志中 `rewards/HumannessReward/mean` 经常不低，但最终 Kimi `humanness` 没提升。这提示：**训练内 local reward 与最终 Kimi humanness 不一致**，或 reward 主要奖励了表面/长度/格式/准确率代理，而不是最终评测想要的“自然推理”。

另一个观察：GRPO 的 KL 约 `1e-4`，很小。可能说明 50 步/低 LR/beta 下策略移动幅度有限，也可能说明 reward 虽有变化但没有足够推动模型分布。

## 代表性失败模式

### 1. 正确答案但 humanness 大跌：资料归纳/照抄变重

样本 204：`汇算清缴缴纳的所得税如何进行账务处理？`
- base：h=0.85, g=1.0, acc=correct
- new：h=0.3, g=0.7, acc=correct
- 评语：new think 先复述三种准则的具体分录内容，最后才给答案；大量 `verbatim_copy` 和 `ref_enumeration`，呈现“从资料向答案归纳”，不是“从问题向答案推导”。

### 2. correct 不变，但 RAG 痕迹明显增加

样本 46：`计入成本的工资和实际支付给职工的工资有什么不同？`
- base：h=0.75, g=0.95, acc=correct
- new：h=0.3, g=0.95, acc=correct
- 评语：new 按资料条目逐项归纳，并出现近似照搬；忠于参考但不自然。

### 3. acc 改善但 humanness 降低

样本 81：`公司外聘培训人员报销的交通费用如何入账？`
- DPO/GRPO 后 acc 从 incorrect 改到 partial，但 h 从 0.7 降到 0.3。
- 说明模型可能通过更充分罗列政策/场景提高准确性，但这会强化“资料拼接/检索腔”。

### 4. answer 也可能被带偏

样本 195：`小规模企业有季度30万免税额度，为什么餐费收入不包含在内？`
- base：h=0.85, g=0.95, acc=correct
- new：h=0.4, g=0.3, acc=incorrect
- 评语：new 从用户错误前提“餐费收入不包含”出发强行找理由，与标准答案相矛盾，属于脱离参考的臆测。

## 我们目前最可疑的失败根因

请你验证或反驳这些假设：

1. **DPO pairs 信号弱/错位**  
   复用了旧 pair，没有针对 `RFT merged base` 重新 rollout 和重判。chosen/rejected 未必在 Kimi humanness 上有稳定 margin。

2. **DPO 训练目标与目标指标错位**  
   如果 pair 的 chosen 只是 local reward 高，但并不代表 Kimi humanness 高，那么 DPO 不可能提升最终 humanness。

3. **GRPO reward 与 Kimi humanness 不一致**  
   训练内 `HumannessReward` 不低，但终评 humanness 不涨，说明 reward 可能奖励了表面代理，或者对“从问题出发推导”刻画不够。

4. **准确率/grounded 与 humanness 存在局部张力**  
   GRPO from RFT 提升 acc，却降低 h；模型可能通过更详尽罗列参考资料来守准，从而增加 RAG 痕迹。

5. **训练步数/学习率可能太保守，但这不是首要问题**  
   KL 很小，说明策略移动弱；但盲目加大步数/学习率可能只会放大错误 reward。因此必须先修 pair/reward，再谈放量。

## 你要输出的内容

请不要泛泛而谈。请按下面格式输出：

### 第一部分：失败原因诊断

- 按概率排序列出 5 个根因。
- 每个根因给出证据、反证、还需要补的诊断。
- 明确哪些方向不是优先级，例如单纯加步数、单纯调 beta、单纯换 reference。

### 第二部分：v2 技术路线

目标：让 RL 阶段相对 `RFT merged base` 在 humanness 上有明确提升，同时 acc 不低于 RFT 太多。

请至少覆盖：
- 是否重做 rollout。
- 如何重构 DPO pairs。
- chosen/rejected 的筛选规则和 margin。
- 是否使用 Kimi 参与离线 pair 构造。
- GRPO reward v2 如何定义。
- 如何避免 reward 推出资料罗列/照抄。
- 如何保持 grounded 和 acc。

### 第三部分：代码/数据改造清单

请明确：
- 要新增或修改哪些脚本。
- 新输出文件名和目录名，统一使用 `corrected-v2`，不要覆盖 corrected-v1 或历史产物。
- 需要生成哪些诊断文件。

### 第四部分：最小实验计划

要求能在半天到一天内给出方向性证据：
- 先跑哪些诊断。
- 跑多少样本。
- 成功/失败门槛。
- 再决定是否上完整 DPO/GRPO。

### 第五部分：给组长的解释话术

要能解释：
- 为什么 corrected-v1 不是工程失败，而是暴露了旧 RL 信号没有对准目标。
- v2 如何把 RL 信号对准 humanness。
- 预计哪些指标会改善，哪些指标必须守住。

