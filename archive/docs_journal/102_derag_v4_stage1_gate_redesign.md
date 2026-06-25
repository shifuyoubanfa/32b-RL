# 102 · derag_v4 Stage1 门禁重构定稿（唯一方案，Codex 照图施工）

> 评审：税务业务 / 裁判 Prompt / RL 信号衔接 / 红队 四 agent 并行实算（合计亲读 38+ 样本、737 条全池重放模拟、98 条已有 Kimi 双判重放），主审仲裁三处提案冲突后定稿。
> 所有阈值、prompt、伪代码定死，无分叉。判分溯源/run_id 纪律沿用 101 号 §二 Q6，不重复。

---

## 1. NO-GO 根因结论（带证据）

**结论：本轮 NO-GO 的主因是本地确定性门在税务文本上的系统性误杀（639 杀中估计 ~75% 是误杀），不是改写质量问题。** 改写质量证据：全体 fact_recall=1.0、95/98 双判 correct、进入 Kimi 复审者 61.2% 通过。

四个误杀源（reward_v3.py 行级 + 抽样证实）：
1. **copy_ratio 0.30 硬门错位**：纯 copy 误杀 116 条全部落在 (0.30,0.40] 窄带；亲读 4/4 是完美自然推导，copy 来自必保事实（"13%、9%、6%三档"、UI 菜单路径、表单栏次）。copy(0.40,0.55] 带亲读 3/3 也是业务必须（短文+一段 86 字法条直引的分母效应可把 copy 顶到 0.54）。
2. **verbatim_copy 是 copy 的别名**（:134 `int(copy_ratio>=0.40)`），与 copy 门双重计数，贡献 trace 杀 177 次。
3. **policy_source 全局 list_like 规则杀合法多引用**（:123 全文 ≥3 个文号/《》即 +1）：杀 174 次；三轮规则修复后抽样仍 ≥80% 为合法推理句（嵌套引用《A》所附《B》计 2、准则名作分情形条件、内容词表漏词、表单名《登记表》被当文号）——**确定性 policy 检测有精度天花板，必须退出硬门**。
4. **degenerate 杀分录文体**（:185 8-gram 重复 ≥3 + :183 短句比例按"；"切分）：杀 312 次，全语料真退化仅 ~9 条；杀手 gram 是"应交税费—应交增值税"（分录本性）、"2027年12月31日"（法定日期）、"A105080表"（申报表号）。
5. 次因：Kimi 复审 min(双判)≥0.85 的 min() 聚合在等于目标均值的阈值上，对真值 0.87 样本误杀率 ~60%（38/98 被杀与此吻合）。
6. **红队抓到的反向证据（防放水必须记录）**：新放行集里有 12 条 think 内嵌 `<img>`/OSS 图床链接的实锤机器痕迹，Kimi 双判全放行（旧 pass-60 里混入 idx=250 一条）——URL 类痕迹 Kimi 是盲的，只能确定性硬杀。
7. 枚举体 query（税率表/有哪些类，51/737=6.9%）旧门通过 0/51——整类文体被系统性灭杀；其中 34/51 自带推理组织，抽样是教科书级好推理。

真正无可争议的硬杀仅 ~56 条（intro_nums 27 + ground_floor 23 + explicit 5 + enum 1）。

## 2. 新架构总览

```
737 改写 ─▶ L0 硬门（无可争议项，确定性，预估杀 ~85）
            ├─ 杀 → 审计①分层复核（防硬门误杀）
            ▼
        L1 软特征（masked_copy / citation_density / degen 软指标 / 口癖统计——只入元数据+裁判上下文，不淘汰）
            ▼
        D5 真痕迹确定性路由（图链/客服话术/问答体，13 条 → 直接进定向修复，不杀）
            ▼
        审计⓪ 锚集分位重定标（新 prompt 阈值不许直接继承）→ 先导闸（60 条分层试跑 ≥0.62）
            ▼
        L2 Kimi 双维专项裁判：J-trace k=2（I1 temp0 / I2 temp0.3，证据先行+span 验真）；J-fact k=1（score<0.90 或 masked>0.45 升 k=2）
            ▼
        L3 仲裁：auto-PASS / auto-FAIL（分裂票禁杀）/ 灰区 → J-arbiter（T1-T4 二值清单裁决，k=1）
            ├─ FIX → 定向修复回路（四路由，每条 ≤1 次，修后重过 L0→L3 全链）
            ▼
        口癖闸 + 防放水四审计 → S1 训练集（auto/arb/fix 三类血统标记）
```

## 3. 硬规则、软特征与阈值（终版定义）

**L0 硬门（杀，全部确定性）**：
```
l0_pass = format_ok(40≤len(think)≤2200 ∧ answer非空)
  ∧ introduced_nums == ∅                       # 数字守恒（NFKC 归一化后比对，见下）
  ∧ grounding_floor_ok
  ∧ explicit_ref == 0 ∧ ref_enumeration == 0    # 显式字眼，precision 实测最高
  ∧ no_label_line(^(政策依据|参考文件|参考法规|文件依据)\s*[:：])   # 全池仅 1 条命中，零误杀
  ∧ raw_copy_ratio ≤ 0.55                       # 极端帽（杀 23/737，残余误杀由审计①兜底）
  ∧ not img_trace(<img\b | https?://\S*(aliyuncs|oss-|servu))      # 红队 D1：图床/内嵌图实锤；官方 gov 域名属 W5 白名单不杀
  ∧ not extreme_degen                            # 类归一化版，见下
```
**extreme_degen（类归一化版，仲裁定稿——白名单版实测仍误杀 102-110 条，废弃）**：
```
norm = NFKC(think)；去引号/统一破折号；剔除分录行 ^(借|贷)[:：记] 与 ^【?(财务|预算)会计】?$
norm = re.sub(r'A\d{5,6}','⟨F⟩'); 日期→⟨D⟩; _DOC_TOKEN→⟨C⟩; 数字→⟨N⟩
extreme_degen = distinct2(norm)<0.10 ∨ max_8gram_repeat(norm)≥5 ∨ 同一≥10字归一化句重复≥3
```
实测：737 条杀 ≤23、greedy 本底 3.7%（旧规则 45.6%）、旧 98 det_ok 与旧 60 过门样本 100% 保留。

**必要税务事实白名单 W1-W7（遮罩函数 MASK，命中替换为 □ 后再算 copy）**：W1 法条直引从句（按照/根据…规定…）；W2 税率档/期限/金额（\d+%、万分之X、自\d{4}年…日起、\d+万元/个月/倍）；W3 分录句式+科目名（借/贷：…、应交税费—…、应付职工薪酬 等）；W4 文号/法规名（《…》(第X条)?、财税〔\d{4}〕\d+号）；W5 操作路径/栏次/官方URL（【…】→【…】、A\d{6}表、第\d+行栏、gov 域名）；W6 法定术语串（57 词 config 清单：非正常损失/汇算清缴/应纳税所得额…）；W7 表单凭证名（《…表/单/凭证》）。

**L1 软特征（永不淘汰，入元数据+裁判上下文+排序）**：
- `masked_copy = copy_signal(MASK(think), MASK(refs))`：≤0.25 干净；(0.25,0.35] 灰（spans 注入裁判上下文）；>0.35 → 路由 R-fix-paraphrase。校准锚：PASS60 实测 masked p90=0.256/max=0.273；上线前 PASS60 重放 p99≤0.30 否则阈值整体 +0.05（只调正则不调阈值的例外通道，预注册）。
- `citation_density = standalone_units / 句数`：standalone = 句内引用单元 ≥1 且无内容动词（内容词表扩至 50 词），或单句 ≥3 单元；嵌套引用《A》所附《B》合并计 1；五大会计准则/制度名不计；W7 表单名先剔除。**删除全局 list_like 规则。**
- degen 软指标：short_ratio>0.25（只按[。！？!?\n]切句、排除分录行与含数字句）∨ 类归一化 8gram 重复 ∈{3,4}。
- 口癖统计：'综上|综上所述' 频次/千字、句首二字开头分布。

**真痕迹确定性路由（D5，不杀只修，实测 13/737，PASS60 中 0 条）**：`<img`/OSS 链已入 L0 硬杀；`小贴士|温馨提(示|醒)|参考下图|如下图|哦[~～]|哒[。~～]|亲[，~～]`（客服话术）、`问题\s*\d+\s*[:：]|回答\s*[:：]`（问答体）→ R-fix-trace。

## 4. Kimi 裁判 Prompt 全文

**J-trace（L2 主裁判，k=2：I1 temp0.0 / I2 temp0.3，max_tokens 900；输入截断 ref[:3000]/think[:4000]）**——采纳税务业务 agent D6 版全文：

```
你是税务领域 RAG 痕迹审查员。输入：【问题】【参考问答对】【待审think】【辅助特征】(masked_copy、citation_density、残余匹配spans)。
只判 think 是否机械搬运参考资料，不判答案对错。
第一步 列出 trace_spans（逐条抄录原文片段+类型）：
A 提及检索装置：出现"参考问答对/参考资料/资料显示/根据提供的/检索结果/问题1/回答："等字样；
B 客服话术残留：小贴士/温馨提示/哦~/哒/亲/您可以参考下图/如图/<img/图片链接——这是发给客户的"回答"文体，不是推理文体；
C 答案体复刻：整段复刻参考"回答"的版式（"情况一/情况二"原样照搬且段内无任何推理连接词；或把参考里多个问答的回答原样串联，含与本题无关的内容）；
D 清单式甩文号：单独成句/成行的"参考文件：××号"堆叠，引用不服务于任何推理步骤；
E 无消化照搬：必要事实之外的叙述句、建议语、举例逐字与参考一致成段出现。
第二步 以下永远不是痕迹（必要税务事实白名单）：
1 法定枚举与税率档（"适用0.05%税率规定的：…"、13%/9%/6%三档、印花税税目税率）；
2 推理句内的法条/文号引用，含一句话引两个文号做新旧法对比、"《A》第X条及所附《B》"嵌套引用、《企业会计准则》vs《小企业会计准则》等制度名作分情形条件；
3 会计分录与科目名（借：应交税费—应交增值税 贷：银行存款），同一科目/分录句式反复出现是分录文体本性；
4 法定术语与政策名重复（"非正常损失""研发费用加计扣除政策"多次出现）；
5 操作路径/表单栏次/官方网址（"报表填写"→"股权转让所得"、A105050表第1列、https://etax.chinatax.gov.cn）必须逐字保留；
6 计算公式的文字表述（应纳税额=（每次收入-800）×20%）；
7 若【问题】是枚举体（求税率表/清单/有哪些）或求政策原文/依据，罗列法定项目、汇总文号本身不是痕迹，只看罗列之外有无推理组织（界定概念→分类→收束）。
长度不是评分维度，更长不加分更短不扣分；分数必须完全由你列出的 spans 解释。
输出 JSON：{"trace_spans":[{"span":"…","type":"A-E"}],"trace_free":0~1}
锚点：无span=0.95~1.0；仅1处B/E轻残留=0.85；出现A或D=≤0.6；C成段复刻=≤0.4。
trace_free 必须与所列 spans 一致：无 span 不得低于 0.9，有 A/C/D span 不得高于 0.7。
```
另附 3 条 few-shot（版本化 judge_v4.0，改样例即升版）：idx96（合法多引用好样本，4 个文号全嵌推理句→无 span）、idx519 原 think（四种痕迹形态齐全→A/D spans）、idx666（清单同构搬运→C span，0.5 档教学样本）。quote 已全部本地 string-match 验真。

**J-fact（k=1；score<0.90 或 masked_copy>0.45 升 k=2；输入=原 think+改写 think+参考+gold）**：
```
你是税务事实保持审查员。改写允许精简冗余表述，但不允许丢失答案成立所依赖的落点。
第一步 输出 fact_issues：[{type ∈ {数字丢失,口径改变,极性翻转,新增无据,依据链断裂}, quote(≤30字逐字摘录)}]
- 数字丢失/依据链断裂的 quote 摘自【原think】；新增无据/极性翻转的 quote 摘自【改写think】。
- 数字/税率/期限在白名单内的改写表述变化（如"乘以9%"↔"×9%"）不算问题。
第二步 输出 grounded 0~1。锚点：无 issue=0.95~1.0；1 处次要事实弱化=0.85；关键数字/口径/极性问题=≤0.5；主体臆测=0.2。
给 1.0 的前提是你逐项核对了全部数字事实且 issues 为空。只输出 JSON：{"fact_issues":[...],"grounded":0.x}
```

**J-arbiter（灰区终审，k=1 temp0，max_tokens 1200）**——红队二值清单版与裁判 agent 版合并：
```
你是终审裁判。前两位裁判对这条改写 think 是否残留 RAG 痕迹有分歧。
【问题】【参考材料】【待审think】【机器特征】masked_copy={mc} 最长连续重合={mx}字 引用句统计={cd}
【前两票】J1 trace_free={a} spans={sa}(每条带 verified:true/false)；J2 trace_free={b} spans={sb}；分歧说明={disagreement}
痕迹定义（只按此清单逐条核对，verified=false 的 span 视为不存在）：
T1 显式指代检索材料的措辞；T2 与参考连续重合≥40字且该片段不是税率表/法定枚举/会计分录等唯一正确表述；
T3 think 中出现 URL/<img>/附件名；T4 文号或《文件名》清单式罗列（同句≥2文号无推理动词，或独立成句）。
不评文风、不评长度、不因引用必要税法名/税率/日期判 fail。
逐条输出 T1-T4 是否存在及证据 span，然后输出唯一 JSON：
{"t1":bool,"t2":bool,"t3":bool,"t4":bool,"verdict":"pass|fail","fix_type":"none|trace|paraphrase|citation|fact","reason":"一句话"}
任一 Tx=true 且不可由修复消除 → verdict=fail；可由修复消除 → verdict=fail 但 fix_type≠none（进修复回路）。
```

**定向修复 prompt（R-fix 公共 system + 三模板 + fact 变体）**——裁判 agent D5-D7 全文采纳（R-fix-trace 删来源提及保事实；R-fix-paraphrase 改行文保法定表述与全部数字；R-fix-citation 引用融进推理句、同文号去重但至少保一处支撑结论；fact 变体把丢失落点句最小融回）。统一追加硬约束（红队 D6）：只动指定 spans、数字/税率/日期/文号零改动、**禁用"综上/综上所述/因此可知/需要注意的是"等书面总结语，保持口语化语气**、不出现 URL/<img>、只输出修复后 think。温度 0.3、max_tokens 2048。

## 5. 重复裁判、一致性与噪声处理

- **span 本地验真**（确定性）：quote NFKC 归一化后 ≥6 字且为目标文本子串；一票内 invalid span 占比 >50% → 票作废同温重试 1 次，仍超 → 票=error。分数-证据矛盾（≥0.9 带 verified 机械引用 span / <0.6 零 span / grounded<0.6 零 issue）→ 重试 1 次。每票总重试 ≤2（格式 1+一致性 1）。
- **字段顺序强制**：解析时校验 spans/issues 的 key 在 raw 文本中先于分数 key 出现，违序重试 1 次（违序=分数未以证据为条件生成）。
- **聚合**：tf_mean/tf_min 取有效票；两票差 >0.30 或仅 1 票有效或剔过 invalid span → 一律进仲裁；任一维度有效票=0 → 直接进仲裁（裁判系统性失败不淘汰样本）。
- **审计⓪（一切之前，红队 D4）**：旧判分是格栅分布（47% 判分恰为 0.85），新 prompt 必移分布。锚集=旧 60（good）+30 实锤坏样本（12 URL+explicit/enum/高 verbatim，bad），新 J-trace k2 跑锚集：要求 AUC≥0.85；auto-PASS 线取 max(0.85, 旧60 在新分布上的 P20 分位映射值)；**禁止看分后手调阈值**。锚集失败（旧 60 新均值掉 >0.05 且 AUC<0.85）→ 修 prompt 重跑审计⓪，不改阈值。
- **判分漂移哨兵**：每 L2 批次混入 5 条冻结锚样本，锚均值移动 >0.05 → 停批重校。

## 6. 规则 × Kimi 组合逻辑（最终裁决权）

```
auto_PASS = l0_pass ∧ tf_mean≥T_pass ∧ tf_min≥T_pass−0.05 ∧ gd_mean≥0.85 ∧ gd_min≥0.80    # T_pass 由审计⓪定，下限 0.85
auto_FAIL = 双票有效 ∧ diff≤0.30 ∧ (tf_mean<0.60 ∨ gd_mean<0.70)                            # 分裂票禁止 auto-FAIL
其余 → J-arbiter：verdict 即终审（pass 需 ∧ tf_mean≥0.75 ∧ gd_mean≥0.80）；fix → 修复回路；fail → 淘汰
```
**裁决权规则**：硬失败（L0）规则说了算（Kimi 对 URL 盲已实证）；灰区裁量 Kimi 说了算，但 Kimi 的每次扣分必须有可本地验真的 span 支撑——**规则与 Kimi 冲突时，以"span 是否真实存在"为准**。仲裁放行率 >70% 触发橡皮图章告警并人工复核 10 条。

## 7. 737 条存量的复用、重判与定向修复

1. 全量重过新 L0（零调用）：预估硬杀 ~85（intro27∪ground23∪explicit5∪enum1∪label1∪copy>0.55 23∪img 12∪真退化≤10，含重叠），~650 进 L2。旧 98 det_ok 与旧 60 过门零回归（已重放验证）。
2. 98 条已有 k2 判分**绑定 cand_id 复用不重判**（实测在新 L3 下 auto 61/灰 27/fail 10）。
3. **先导闸**：新放行样本按 copy 分层抽 60 跑完整 L2+仲裁，`auto率+0.5×灰区率 ≥0.62` 才放全量（旧 98 最干净集实测 62%，新集更脏只会更低——预估必须先证伪再烧 ~2400 调用）。
4. 全量 L2（~650×(J-trace k2 + J-fact k1) ≈ 2000 调用 + 升级/重试 ~15% + 仲裁 ~160-220）。
5. 修复回路：路由量预估 ~150（trace 13 + paraphrase ~120 + citation ~20），修后重过 L0→L3 全链，二次 FIX 按 FAIL。修复样本 `ratio(修后,修前)≥0.60 ∧ nums 集合不变` 确定性验收。
6. 产出预估（双锚定）：auto 62-65% ≈ 410-422 + 仲裁回收（27.6%×~50%）+ 修复回收（~150×50%）→ **480-540**，>400 门；若先导闸 <0.62 → 停下修方案，不烧预算。

## 8. Stage1 的 PASS / RETRY / FAIL / NO-GO

**样本级**：PASS（三径：auto/arb/fix）｜RETRY（=进修复回路，每条 ≤1 次）｜FAIL（淘汰，元数据与 span_rulings 落盘入 DPO 负池候选）。
**阶段级 S1 PASS 必要件（全部满足）**：
- 过门改写 ≥400（min_rewrites 不变）；
- 防放水四审计全过：**审计⓪** 锚集 AUC≥0.85+分位定标；**审计①** L0 硬杀分层抽 30（必含 copy-cap 23 条中 ≥10 + img 全部）Kimi k2 复核，分层 precision 各 ≥75% 且合并 ≥80%；**审计②** auto-PASS 抽 50 fresh k2，min(tf)≥0.75 保持率 ≥85%（配漂移哨兵区分裁判漂移与放水）；**审计③** 旧60∪新60 混编同批同 prompt 盲评：新均值 ≥ 旧 −0.02 ∧ 实锤痕迹率（URL/explicit/enum/masked≥0.45 四项可数指标）≤ 旧 +2pp；
- **口癖闸**：train 集 '综上' ≤0.15/千字 ∧ 句首 top-6 二字开头相对 V1 原 think 语料偏移 ≤+3pp（实测 Kimi 已把'综上'写到 ×29——不闸则 SFT 学公文腔）；
- 血统配额：rw_fix ≤20% 过门总数（超帽不回填，告警）；rw_arb >35% → 裁判漂移审计；
- 设计效应预注册 ≥ MDE 0.013（实算 0.0215τ，τ 点估 0.61-0.9 → 0.019，写明 τ 假设）。
**RETRY（阶段级，恰 1 次）**：过门 300-400 → 对 FAIL-paraphrase 桶补一轮定向修复。
**NO-GO（终局）**：retry 后仍 <300——此时门已过四审计，结论是"改写质量问题"，处置=回改写 prompt（不是再松门）。

## 9. SFT / DPO / GRPO 数据与信号构造

- **SFT 训练集**：过门改写（三血统标记 derag_v4_rw_{auto,arb,fix}，同权）+ replay 150（**新 L0 重筛、零 Kimi**，kill>10% 从 123 非池题补足）；query 级去重；修复样本 ≤20%。
- **DPO 种子池**（126_dpo_seed_pools.jsonl）：T1 锚定对=chosen 取 auto/arb-PASS 改写（自带 Kimi 证书）、rejected 取同题源 greedy think，准入 `B_src≥1 ∧ dB≥1`（实算可用 211 对），计重写对帽 ≤5%；T2 硬负对=auto-FAIL(trace 败因) 改写 vs 同题 T1 chosen（同为 Kimi 文体，抵消文体混淆），计手术帽 ≤10%；T3 纯负池=481 条 B≥1 源 think+trace-fail 改写，仅供 S2 引导轮当 rejected。**B=0 的 256 条源 think（34.7%，旧双重计数的假载体）禁入一切负池；rw_fix 禁作 DPO chosen。**
- **margin 重定义**（旧 trace_hits 差 ≥2 废止——explicit/enum 清零后 on-policy 不可达）：痕迹负担 `B = explicit + enum + standalone_citation_units + copy_units(masked: 0.35/0.45/0.55 阶梯 0/1/2/3)`；`margin_ok := B_rej−B_cho ≥2 ∨ (≥1 ∧ Δmasked_copy≥0.12)`，100% 确定性。
- **S2 chosen 资格**（同文件同阈值）：`gate_decision(feats, mode)` 唯一入口；fast-path（零 Kimi，G1-1 只认这条）=B=0 ∧ masked≤0.30 ∧ ¬degen ∧ fact_recall≥0.75 ∧ introduced=∅ ∧ floor ∧ answer_score≥0.55；grey-path=masked∈(0.30,0.45] → Kimi k2 min()，帽 2500 调用；masked>0.45 禁为 chosen。
- **G1-1 联动**：p_clean ≥**60%**（旧 50%+实测 clean 定义机械放松 10.2pp 的等严校准：greedy 36.1%→46.4%）；p_pair := margin 可成对率 ≥40%（heavy 侧新定义实测紧 2.3 倍，不再加码）。
- **GRPO reward_v3 逐行 diff**：删 :134 verbatim 行；:123 删 list_like；policy 罚项 → `0.20×standalone_units + 1.20×max(0,citation_density−0.15)`；copy 罚项 → `2.0×max(0,masked−0.25) + 8.0×max(0,masked−0.40)`（0.55 raw 硬帽保留；点位 masked 0.30→0.93、0.40→0.78、0.50→0.45）；:239 硬门只剩 format ∧ ¬extreme_degen（类归一化版）；degen 软指标只进 L1 不进 reward。
- **痕迹计数器版本冻结**：旧 trace_counts 全套冻结为 `trace_re_v3_frozen`，G1-2/G2/G3 一切 McNemar"计数降"验收只用 frozen 口径（跨阶段可比）；门禁/reward/margin 用 `trace_re_v4_gate`；比较函数 assert 两侧 version 一致。
- **G1-2 预注册修订**（训前改）：删"verbatim 计数降"（训练目标自身 verbatim frozen 口径 105→145，原条款必败）→ "224 评测 copy_ratio 均值升幅 ≤+0.02 ∧ frozen(explicit+enum+policy) 合计降 ≥30%"。

## 10. 代码改动清单

| 文件 | 改动 |
|---|---|
| `pipeline/reward_v3.py` | MASK(W1-W7 正则+57 词术语表入 config)、masked_copy、citation_density（嵌套合并/准则豁免/50 内容词）、extreme_degen 类归一化版、img_trace、no_label_line、删 verbatim/:123 list_like、`gate_decision(feats, mode∈{s1_keep,dpo_chosen})` 唯一入口、derag_reward 罚项按 §9 diff、`trace_counts_v3_frozen` 原样保留并加 version 字段 |
| `pipeline/judge_common.py` | 新增 J_TRACE/J_FACT/J_ARBITER/R_FIX×4 模板（§4 全文）、judge_version='judge_v4.0'、span 验真函数 verify(quote,target)、字段顺序校验、few-shot 三样例常量 |
| `pipeline/step125_gate_rewrites.py` 重写 | L0→L1→D5 路由→L2(J-trace k2+J-fact k1 条件升级)→L3 仲裁→修复回路→口癖闸；98 条 cached 判分按 cand_id 复用；行 schema 增 {l0_reasons, l1:{masked_copy,citation_density,degen_soft}, l2_votes(含 spans+verified), l3_path∈{auto,arb,fix,fail}, fix_type, repair_round, judge_run_id} |
| 新增 `pipeline/step125a_anchor_calibration.py` | 审计⓪：锚集构建（旧60+30 实锤坏）、AUC、分位映射、T_pass 写入 decision；漂移哨兵样本冻结 |
| 新增 `pipeline/step125b_replay_gate_dryrun.py` | §11 离线重放（默认零新调用），输出 125b_replay_report.json + 行级 jsonl |
| 新增 `pipeline/step126_dpo_seed_pools.py` | T1/T2/T3 分层池 + B 负担计算 + margin_evidence 四元组 |
| `run_derag_v4.py` | S1 阶段插入：审计⓪→先导闸→全量 L2→修复→四审计→口癖闸→产出门，全部 raise 硬停；replay 重筛接 gate_decision |
| 回归用例固化 | 17 条亲读好样本（idx=127/330/347/670/45/619/382/256/696/457/535/138/109/690/220/120/327）assert l0_pass=True；idx=250/545 等 12 条 img 样本 assert 硬杀 |

## 11. 离线重放验证与诊断报告

`step125b_replay_gate_dryrun.py`（输入 125 rows+124 rewrites+123 greedy+124 trace_pool；默认零新 Kimi，可选 --fresh_k2_budget 160 按旧杀因分层抽）输出：
- `l0`={entrants, kills_by_reason, combo_top, old98_preserved(assert==98), old60_preserved(assert==60), img_kills}
- `l1_quantiles`=entrant vs killed 的 masked_copy/citation_density/degen 分位对照
- `l3_sim`={auto, grey, fail(cached 98 条), fresh_strata_auto_rate±CI}
- `ab_audit`=审计③盲评结果；`design_effect`={β=0.1044, removal/样本, Δ@τ∈{0.5,0.6,0.7,0.9}}
- `dpo_preview`={anchor_dB1=211, B_src 分布, neg_pool=481}；`yield_matrix`(arb×repair 九宫格)；`repair_routing_counts`

## 12. 自动重试、回滚与停止

- 票级：格式/一致性各重试 1 次；样本级：修复 ≤1 次；阶段级：RETRY 恰 1 次（FAIL-paraphrase 桶补修复）。
- 回滚：审计⓪失败→修 prompt 重跑审计⓪（阈值不动）；漂移哨兵触发→停批、重校、该批判分作废重判；仲裁放行率>70%→人工复核 10 条，确认橡皮图章则该批 arb-PASS 降级灰区重审。
- 停止（NO-GO）：retry 后 <300 过门 → 归因"改写质量"，回 step124 prompt（修复方向由 125b 漏斗给出），不松门；四审计任一不可修复地失败 → 整轮停，出诊断报告。

## 13. 反方审查结论与防护（红队定稿）

红队裁决：**"减少误杀"成立，非变相放水**——证据：583 条新放行样本硬证据可疑仅 4.5%（copy 0.45-0.55 带 64 条无一条 verbatim 覆盖 ≥0.5）；auto-PASS 线在 98 条真实判分上重放仅 +1 翻案（等效旧线，工作点从隐性 0.883 修回声称的 0.85，代价=真值 0.80 样本 +7pp 通过，由下游探针兜底）。
四个必修洞（已全部纳入本方案）：①URL/图床硬杀（L0 新增）；②degenerate 换类归一化（白名单版会再误杀 ~102 条且可被 GRPO 刷分）；③仲裁二值化 T1-T4 清单（防橡皮图章）；④审计⓪阈值分位重定标（防跨 prompt 量纲搬运——**整套重构最可能崩的点**，预警信号=旧 60 在新 prompt 下均值掉 >0.05）。
**放水终审判据（可证伪四开关，任一亮灯=判放水回滚）**：审计③盲评失败；仲裁放行率 >70%；跳过审计⓪或看分调阈；s1 SFT 后痕迹探针 AUC 高于 RFT base（痕迹被学进去）。
残余风险（接受并记录）：masked_copy 正则的查全率依赖 W1-W7 覆盖（PASS60 重放 p99 校准 + 17 条回归用例兜底）；修复样本的 Kimi-on-Kimi 文体放大（配额 20%+口癖闸+永不作 DPO chosen 三重限制）。
