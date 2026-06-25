# 103 · Stage1 门禁 v2：二值票制（替代 102 号的连续分打分层，Codex 落地稿）

> 原则（与用户达成的共识）：①Kimi 只做二值判断+指认 span，不打连续分——连续分是 prompt 的函数，不是测量；②信任分层：极端交给规则（L0 已修好勿动），中间地带交给多票二值，精密度留给方向敏感的环节（DPO margin、224 终评）；③SFT 种子不需要 DPO 级纯度，门禁复杂度降到任务需要的水平。
> 与 102 号的关系：L0/L1/白名单/修复路由/口癖闸/DPO-GRPO 衔接（§9）全部保留；**替换的只有 L2 打分层、L3 阈值聚合、审计⓪的 AUC 口径，删除审计②③（由人工抽查替代）**。
> 终评尺子不变：224 验收仍用 v3.1 冻结裁判（trace_free 连续分，噪声已标定 sd=0.056/MDE=0.013）——坏掉的是门禁尺 judge_v4.0，不是验收尺。

---

## 1. 为什么换二值（一段话留档）

三代连续分裁判三种死法：humanness 0.85 封顶；derag 格栅版 47% 判分恰为 0.85（默认值盖章）；evidence-first 版变找茬机器（锚集 AUC 0.69，good 均值 0.548）。而人和 agent 亲读样本的好坏判断高度一致——任务本身是可判的，不可判的是小数点。二值票制把裁判用在它能稳定输出的颗粒度上。

## 2. 架构（v2 全流程）

```
737 改写 ─▶ L0（102 号原版，已验证：杀 83，degen 仅 1，img 8——勿动）
            ▼ 654 entrants
        L1 软特征（masked_copy/citation_density/burden——只作裁判上下文与排序，无任何阈值淘汰）
            ▼
        审计⓪-binary：锚集 90 条（旧60 good + 30 实锤 bad）二值定标
            ▼ 过门
        L2 二值票：J-trace-bin k=2（I1 temp0 / I2 temp0.3）＋ J-fact-bin k=1
            ├─ 双票 clean ∧ fact_ok → PASS
            ├─ 双票 traced（各自 ≥1 条 verified span）→ 按 span 类型路由修复（≤1 次）或 FAIL
            └─ 分歧票 / 低置信 / span 验真失败 / fact issue → J-arbiter（T1-T4 清单，二值）→ 三票多数
            ▼
        口癖闸（保留）＋ 人工抽查 30 条 ＋ 产出门 ≥400 → S1 训练集 → 后续按 101 号不变
```

## 3. 审计⓪-binary（定标协议，替换 AUC 版）

- 锚集不变：旧 60（good）+ 30 实锤 bad（12 img + explicit/enum + 高 verbatim）。
- 每条跑 J-trace-bin k=2，verdict 取双票一致，分歧加仲裁票取多数。
- **过门判据：good 通过率 ≥85% ∧ bad 通过率 ≤15%**（等价 balanced accuracy ≥85%，替代连续分 AUC≥0.85）。
- 750-85% 之间：允许把"三票多数"作为锚集测量口径重测一次（即定标时就用完整投票机制）；仍不过 → 走 §8 降级路径，**不许改 prompt 重试超过一次**。
- 成本 ≈ 90×2 + 分歧仲裁 ≈ 200-230 调用。

## 4. 裁判 prompt（二值版）

**J-trace-bin**（max_tokens 700；A-E 痕迹定义与白名单条款 1-7 原样继承 102 号 §4，此处只换输出协议）：

```
…（102 号 J-trace 的痕迹定义 A-E 与白名单条款 1-7 全文照搬）…
你的任务是回答一个问题：这段 think 是否残留机械 RAG 痕迹？
- 若有：列出 trace_spans（逐字摘录+类型 A-E），verdict=traced。
- 若无：trace_spans 为空，verdict=clean。
- 不打分数。不评文风长短。拿不准时如实标 confidence=low。
输出 JSON：{"trace_spans":[{"span":"…","type":"A-E"}],"verdict":"clean|traced","confidence":"high|low"}
一致性要求：verdict=traced 必须至少有一条 span；verdict=clean 则 spans 必须为空。
```
few-shot 三样例沿用 102（idx96 clean / idx519 traced / idx666 traced-C 类）。

**J-fact-bin**（k=1）：
```
…（102 号 J-fact 的 issue 类型定义照搬）…
回答：改写是否丢失/改变了答案成立所依赖的事实落点？
输出 JSON：{"fact_issues":[…],"fact_ok":true|false}
fact_ok=false 必须至少有一条 issue；issue 的 quote 须逐字可查。
```

**J-arbiter**：102 号 T1-T4 二值清单版本本来就是二值的，原样沿用（verdict=pass/fail + fix_type）。

**span 验真、字段顺序校验、重试协议**：102 号 §5 原样保留（traced 票的所有 span 验真失败 → 该票视为 clean-low-confidence 进仲裁，不直接采信）。

## 5. 票决逻辑（替代阈值聚合，伪代码）

```
v1, v2 = J_trace_bin(I1), J_trace_bin(I2)     # span 验真后
if v1==v2==clean and fact_ok:            PASS
elif v1==v2==traced(各有verified span):   route_repair(span types) if fixable else FAIL
else:                                     # 分歧/low-conf/验真失败/fact issue
    v3 = J_arbiter(全部 spans+verified标记+L1特征)
    verdict = majority(v1, v2, v3 仲裁票)   # 仲裁票里 fact 类 issue 由 fix_type=fact 路由修复
PASS 后样本带血统 {bin_unanimous | bin_majority | bin_repaired}
```
修复回路：102 号四路由原版，每条 ≤1 次，修后重过 L0→投票全链；rw_repaired ≤20% 配额保留。

## 6. Stage1 PASS 标准（精简版）

1. 过门改写 **≥400**（不变）；
2. 审计⓪-binary 过门（§3）；
3. **人工抽查**：PASS 集分层抽 30（10 条 masked_copy 最高 + 10 条枚举体 query + 10 随机），盲评"是否残留机械痕迹"——**机械痕迹 ≤3/30**；另抽 20 条被票决 FAIL 的样本记录误杀率（只记录供下轮校准，不作门）；
4. 口癖闸（102 号原版：'综上'≤0.15/千字 ∧ 句首分布偏移 ≤+3pp）；
5. 血统配额：bin_repaired ≤20%。
**删除**：审计②（fresh 重判保持率）、审计③（盲评交错重放）、设计效应门移到 SFT 训后报告里算（不再作开训前置——SFT 对噪声鲁棒，前置门的保护价值低于它的阻塞成本）。

## 7. 预算与产出预估

L2 投票 654×2 + fact 654 + 仲裁 ~130（按 20% 分歧率）+ 修复 ~150×4 ≈ **2,700 调用（~2.5h @3workers）**+ 定标 230 + 抽查 0（人工）。
产出：fact_recall 全 1.0 + 改写质量已多轮亲读验证 → 双票 clean 率预估 65-80%（425-520）+ 仲裁/修复回收 → **450-560，>400 门**。若双票 clean 率 <55%（明显低于亲读印象）→ 停，抽 20 条双票 traced 样本人工裁决：人判 Kimi 对 → 接受产出走 RETRY；人判 Kimi 错 → 走 §8。

## 8. 降级路径（预注册，唯一）

若审计⓪-binary 也不过（good<75% 或 bad>25%）：**判定"Kimi 无法在任何颗粒度上胜任此门禁"**，如实入报告。降级为：L0 + D5 确定性真痕迹路由 + 人工抽查加倍（60 条，机械痕迹 ≤6/60）→ 直接入 SFT。链路不断：SFT 对噪声鲁棒；DPO chosen 资格本就走确定性 fast-path（零 Kimi，102 §9 不变）；224 终评仍是 v3.1 冻结裁判。Kimi 门禁失效不阻塞 RL 主线，只记录为测量结论。

## 9. 代码改动（给 Codex）

| 文件 | 改动 |
|---|---|
| `judge_common.py` | 新增 J_TRACE_BIN / J_FACT_BIN 模板（§4）与解析（verdict/confidence 字段、clean-spans 一致性校验）；judge_version='judge_v4.1_bin'；J_ARBITER 沿用 |
| `step125a_anchor_calibration.py` | 加 `--mode binary`：输出 good_pass_rate / bad_pass_rate / balanced_acc，过门判据 §3；保留旧 AUC 模式代码不删（对照用） |
| `step125_gate_rewrites.py` | L2/L3 替换为 §5 票决逻辑；删除一切 tf_mean/tf_min 阈值分支；行 schema 改 {votes:[{verdict,confidence,spans,verified}], final:{verdict,path∈{unanimous,majority,repaired,fail}}, fact:{fact_ok,issues}} |
| 新增 `step125c_spotcheck_sheet.py` | 生成盲评抽查表（30 PASS 分层 + 20 FAIL，乱序、隐藏判分元数据，markdown 输出）；回收人工裁决 json 并出 spotcheck_report |
| `run_derag_v4.py` | S1 流程换 §2；PASS 标准换 §6；降级路径 §8 写成显式分支（raise 改为 DEGRADED 状态 + 报告标记，不静默） |
| 不动 | `reward_v3.py`（L0/L1/masked_copy/burden 已验证）、DPO/GRPO 衔接（102 §9）、224 终评（v3.1 冻结裁判） |

## 10. 运行序（云机）

```bash
python pipeline/step125a_anchor_calibration.py --mode binary --run_id $RUN_ID     # ~230 调用，过门才继续
python pipeline/step125_gate_rewrites.py --run_id $RUN_ID                         # ~2700 调用过夜
python pipeline/step125c_spotcheck_sheet.py --run_id $RUN_ID                      # 出抽查表 → 人工盲评 → 回填
bash scripts/run_derag_v4.sh --stage s1_train --run_id $RUN_ID                    # 过门后 SFT，后续按 101 号
```

## 11. 一句话给组长

"门禁从'让裁判打小数点'改成'让裁判投票并指认证据'：裁判三代连续分各有系统病（封顶/盖章/找茬），但二值判断+span 指认是它稳定能做的事；规则守极端、投票守中间、精密度留给 DPO margin 和终评——门禁复杂度第一次和任务需要对齐了。"
