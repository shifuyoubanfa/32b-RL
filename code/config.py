"""32B（公司 V1）推理人性化 RL —— 集中配置（路径 / 接口 / 采样 / 训练超参）。

与 14B 沙盒的关键差异（详见《32B强化学习_技术方案.md》）：
- V1 = 被强化的模型本体（跳过蒸馏），LoRA base = 本地 V1 权重。
- 不连公司内网：V1 改本地 vLLM 推理；参考资料复用 14B 缓存的 user_prompt；Kimi 走公网 DashScope。
- 训练框架 = ms-swift（在 zhjg_rl 环境）；推理/采样 = vLLM（在 vllm_env 环境，独立进程，HTTP 协作）。
- 奖励复用 reward.py（本轮打开 PMI；PMI 尺子由探针 AUC 选优）。
- 采样：全池 + 随机轮换（不再死取前 N）。

所有路径默认指向服务器本地 NVMe，可用环境变量覆盖（便于本机/他机调试）。
"""

import os

# ====================== 路径 ======================
# 代码根：本文件所在目录（上传到服务器 /mnt/pfs/zhjg/code 下）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 产物/权重根：默认落服务器本地 NVMe；本机调试可设 ZHJG_WORK_DIR 指向别处
WORK_DIR = os.environ.get("ZHJG_WORK_DIR", "/home/nvme01/zhjg")
DATA_DIR = os.environ.get("ZHJG_DATA_DIR", os.path.join(ROOT_DIR, "data"))   # 缓存数据(随代码带)
OUTPUT_DIR = os.environ.get("ZHJG_OUTPUT_DIR", os.path.join(WORK_DIR, "output"))
CKPT_DIR = os.environ.get("ZHJG_CKPT_DIR", os.path.join(WORK_DIR, "ckpts"))
LOG_DIR = os.environ.get("ZHJG_LOG_DIR", os.path.join(WORK_DIR, "logs"))

for d in (OUTPUT_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.environ.get("ZHJG_LOG_FILE", os.path.join(LOG_DIR, "pipeline.log"))

# ====================== V1 模型（本地权重，LoRA 底座）======================
# 组长拷入、已解压、已确认 = Qwen2.5-32B 微调（65.5GB / 14 分片 / 无优化器态）。
V1_DIR = os.environ.get("V1_DIR", "/home/nvme01/zhjg/V1-32B/checkpoint-1500")
# 干净底座（PMI 尺子候选之一；V1 微调前的 Qwen2.5-32B-Instruct，机器上 /mnt/pfs/model 下）
CLEAN_BASE_DIR = os.environ.get("CLEAN_BASE_DIR", "/mnt/pfs/model/Qwen2.5-32B-Instruct")

# ====================== 本地 vLLM 服务（V1 推理 / rollout / 评测）======================
# 由 scripts/serve_v1_vllm.sh 在 vllm_env 环境起；本进程(zhjg_rl)走 HTTP 调用。
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "v1")           # 与 serve 的 --served-model-name 对齐
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "600"))
VLLM_CALL_WORKERS = int(os.environ.get("VLLM_CALL_WORKERS", "32"))  # 并发(vLLM 连续批处理吃得下)

# ====================== Kimi 裁判 / 改写（公网 DashScope 兼容模式）======================
# 不落明文 key：从环境变量读，缺失即报错（见 kimi_client）。
KIMI_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi/kimi-k2.6")  # DashScope 实际 model id
# kimi-k2.6 是 thinking 模型：默认正文全在 reasoning_content、content 为空。
# 实测顶层 enable_thinking=False 能关掉思考 → content 直接出答案、省 token。判分/改写都关思考。
KIMI_ENABLE_THINKING = False
KIMI_API_KEY_ENV = "DASHSCOPE_API_KEY"                    # key 只从这个环境变量读，绝不落明文
# 安全：本仓库公开，DashScope key 一律走环境变量，不写进代码。
# 跑之前先 `export DASHSCOPE_API_KEY=sk-xxxx`（各 launcher 脚本也用 ${DASHSCOPE_API_KEY} 透传）。
# 缺 key 时不在此处兜底，交给 kimi_client 调用时报错，避免静默用错 key。
if not os.environ.get(KIMI_API_KEY_ENV):
    import warnings
    warnings.warn(f"环境变量 {KIMI_API_KEY_ENV} 未设置；需要 Kimi 的步骤（改写/判分）会失败。"
                  f"请先 export {KIMI_API_KEY_ENV}=<your-dashscope-key>。", RuntimeWarning)
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 0.7
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "600"))   # 单次 Kimi 请求读超时(秒)；在线 GRPO 想抗 DashScope 抽风可调小(如 120)让其快失败回退而非死等
JUDGE_CALL_WORKERS = int(os.environ.get("JUDGE_CALL_WORKERS", "3"))   # DashScope Kimi 易 429 过载，降并发
SEED_WORKERS = int(os.environ.get("SEED_WORKERS", "3"))   # Kimi 改写并发（同上，宁慢勿被限流）

# ====================== V1 的系统提示（RAG 腔来源；与 14B 一致，保证口径）======================
SYSTEM_PROMPT = (
    "你是一个乐于助人的AI助手。你的任务是基于提供的参考问答对来回答用户的问题。工作流程：\n"
    "1. 仔细分析用户的问题\n"
    "2. 在参考问答对中搜索相关信息\n"
    "3. 基于参考内容组织回答，确保信息的准确性和相关性\n"
    "4. 如果参考问答对中有完全匹配的问题，直接使用对应的答案\n"
    "5. 如果参考问答对中没有直接匹配，但有关联信息，综合这些信息给出回答\n"
    "最后为用户提供答案推理过程和答案分别包含在<think> </think>和<answer> </answer>标签中，"
    "即<think>\n reasoning process here \n</think>\n\n<answer>\n answer here \n</answer>"
)

# 去检索腔的中性系统提示（§4.8 决策落地）：冷启动起的【训练数据 + 训练后模型推理】用它，
# 不把"基于参考问答对搜索"的 RAG 腔 prompt 烤进 32B。V1 数据构建/基线评测仍用上面的 SYSTEM_PROMPT(还原 V1 行为)。
COLDSTART_SYSTEM_PROMPT = (
    "你是一名专业的中文税务助手。你的推理与结论【必须严格依据已知资料中的事实、政策口径与数字】，"
    "不得凭记忆臆测，更不得给出与已知资料相矛盾的结论；只有已知资料未覆盖的细节才可谨慎使用常识补充。"
    "但表达上要像一位资深税务老师那样，把这些依据自然地融进思考、从问题一步步推导到结论，"
    "而不是罗列或复述「参考问答对」「资料编号」「检索结果」这类字样。"
    "把推理过程放在 <think> </think>、最终答案放在 <answer> </answer> 标签中，"
    "即<think>\n推理过程\n</think>\n\n<answer>\n答案\n</answer>。"
)


def system_for(model_name: str) -> str:
    """基线/原始 V1 用 RAG 腔 SYSTEM_PROMPT（还原 V1 行为）；训练后的模型用去检索腔 COLDSTART_SYSTEM_PROMPT。"""
    return SYSTEM_PROMPT if (model_name in ("v1", None, "")) else COLDSTART_SYSTEM_PROMPT


def resolve_adapter(root: str) -> str:
    """swift 把 LoRA 存进 <root>/v*-时间戳/checkpoint-N/。解析出真正含 adapter_config.json 的目录：
    优先 trainer_state.json 的 best_model_checkpoint；否则取 checkpoint 号最大的；找不到原样返回（让下游报清晰错）。
    全链唯一来源（run.py 起 vLLM/续训、step09 探针尺子、step11/12 PMI 都用它），避免各处重复解析。"""
    import glob as _glob, json as _json, re as _re
    def _ck(p):
        m = _re.search(r"checkpoint-(\d+)", str(p))
        return int(m.group(1)) if m else -1
    states = _glob.glob(os.path.join(root, "**", "trainer_state.json"), recursive=True)
    if states:
        try:
            best = _json.loads(open(max(states, key=_ck), encoding="utf-8").read()).get("best_model_checkpoint")
            if best and os.path.exists(os.path.join(best, "adapter_config.json")):
                return best
        except Exception:
            pass
    cfgs = _glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True)
    return os.path.dirname(max(cfgs, key=_ck)) if cfgs else root


def has_adapter(root: str) -> bool:
    """训练【成功完成】才返回 True：以 <root>/.done 标记为准（sft/dpo/grpo.sh 成功后才写，set -e 保证）。
    不用 glob adapter_config.json——因 swift 每个 epoch 就落一个 checkpoint，中途崩溃会留半成品，
    glob 会把半成品误判为"已训完"而跳过训练、拿未收敛权重进下游（冷启动准确率对此敏感）。"""
    return os.path.exists(os.path.join(root, ".done"))

# ====================== 数据文件（命名沿用 14B 习惯，便于对照）======================
# 输入缓存：14B 已生产、随代码带进 data/（含 user_prompt=参考资料）。
CACHED_TEACHER_OUTPUTS = os.path.join(DATA_DIR, "00_data_teacher_outputs.jsonl")  # 2239，含 user_prompt
# 阶段0：本地 V1 重产 think/answer（金标准 answer + 改写原料 think）
V1_OUTPUTS = os.path.join(OUTPUT_DIR, "00_v1_outputs.jsonl")
# 训练/验收切分（与 14B 同口径：固定 224 验收集）
SFT_TRAIN = os.path.join(OUTPUT_DIR, "00_data_sft_train.jsonl")   # 2014（query+参考+V1 think/answer）
SFT_EVAL = os.path.join(OUTPUT_DIR, "00_data_sft_eval.jsonl")     # 224 验收集（全程冻结）
TRAIN_EVAL_RATIO = 0.9

# ====================== 奖励（移植 14B reward.py；本轮打开 PMI）======================
# 阈值/权重沿用 14B tune 出的先验，正式用前由探针在 V1 自然种子上重新证伪（AUC≥0.7）。
REWARD_TAU_ACC = 0.30        # 答案漂移软门阈值
REWARD_W_HUMAN = 0.5
REWARD_W_ACC = 0.5
REWARD_C_TRACE = 0.34        # 每处 RAG 引用痕迹扣分
REWARD_C_COPY = 1.00         # 照抄率扣分系数
REWARD_W_PMI = 0.5           # humanness 里 PMI 的权重
THINK_MIN_CHARS = 40
THINK_MAX_CHARS = 2000

# PMI：探针(step09)选尺子。2026-06-09 实测决策【默认关】——grounding 修复后，自然 think 故意扣参考、
# 高度依赖参考上下文，而 PMI 假设"越不依赖参考越自然"，二者直接冲突 → 探针实测 PMI AUC=0.34/0.47(反相关)、
# 而表面项 s_trace AUC=0.990。结论同 14B：humanness 奖励用【表面项】(检索腔关键词+照抄率)，不用 PMI。
# 如需重新启用 PMI 探针：export PMI_ENABLED=1。
PMI_ENABLED = os.environ.get("PMI_ENABLED", "0") == "1"
# 候选尺子：'clean_base'=干净 Qwen2.5-32B-Instruct / 'stage_model'=V1+当前阶段冻结 adapter。排除原始 V1。
# （pi_ref 与 current 在本流水线里等价——rollout 总由"待改进的那个模型"生成——故合并为 stage_model，省一次 32B 加载）
PMI_RULER_CANDIDATES = ["clean_base", "stage_model"]
PMI_RULER = os.environ.get("PMI_RULER", "")   # 非空=强制指定尺子(覆盖探针)；空=以 output/pmi_ruler.json 为准
PMI_THINK_CAP = 1024
PMI_MAX_LEN = 4096

# ====================== 采样（全池 + 随机轮换）======================
RL_POOL = SFT_TRAIN
SAMPLE_SEED = 42             # 固定 seed 保可复现的随机轮换
ROLLOUT_MAX_QUERIES = 0      # 0 = 全池（不再死取前 N）
SEED_MAX = 0                 # 冷启动改写：0 = 全部 2014
RL_K = 8                     # 每 query 采样候选数（RFT 选样）
RL_GEN_MAX_NEW_TOKENS = 1024
RL_TEMPERATURE = 0.9         # 拉开多样性给筛选留空间
RL_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 1536    # 评测/数据构建用（确定性偏好低温）
GEN_TEMPERATURE = 0.0        # 数据构建用贪心（金标准可复现）；评测可调
GEN_TOP_P = 1.0

# ====================== 冷启动种子（Kimi 改写全部 2014）======================
SEEDS_RAW = os.path.join(OUTPUT_DIR, "30_seeds_rewritten.jsonl")     # 改写后的自然 think
SEEDS_SCORED = os.path.join(OUTPUT_DIR, "30_seeds_scored.jsonl")     # + Kimi humanness
SEED_HUMANNESS_MIN = 0.60    # 冷启动只用 humanness≥此值的高质量自然种子（c）
SEED_FAITHFUL_MIN = float(os.environ.get("SEED_FAITHFUL_MIN", "0.70"))   # 冷启动种子 grounding/忠实度门槛（Kimi 评 think 是否扣参考、不与之矛盾）
COPY_RATIO_MAX = float(os.environ.get("COPY_RATIO_MAX", "0.50"))         # 照抄率上限（grounding 不等于照搬原文，超此判机器腔）


def seed_is_chosen(rec: dict) -> bool:
    """冷启动入选的【唯一判据】——step08 建训练集 / step09 探针正样本共用，防两处定义漂移。
    五道闸：数字未臆造 ∧ 去检索腔(确定性) ∧ 像人 ∧ grounding 忠于参考 ∧ 不照抄。"""
    return (rec.get("facts_ok", False)
            and rec.get("trace_hits", 0) == 0
            and rec.get("kimi_humanness", 0.0) >= SEED_HUMANNESS_MIN
            and (rec.get("grounded") or 0.0) >= SEED_FAITHFUL_MIN
            and rec.get("copy_ratio", 0.0) <= COPY_RATIO_MAX)
COLDSTART_TRAIN = os.path.join(OUTPUT_DIR, "40_coldstart_train.jsonl")
COLDSTART_EVAL = os.path.join(OUTPUT_DIR, "40_coldstart_eval.jsonl")  # 自然腔留出 eval（早停用）
COLDSTART_EVAL_FRAC = 0.10
# 低质量改写(d) + 原始机器腔 → 探针"低分对照"
PROBE_LOW = os.path.join(OUTPUT_DIR, "30_probe_low.jsonl")
PROBE_REPORT = os.path.join(OUTPUT_DIR, "35_probe_report.md")

# ====================== swift 训练超参 ======================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
MAX_LEN = 4096

# 冷启动 SFT
COLDSTART_LR = 5e-5
COLDSTART_EPOCHS = int(os.environ.get("COLDSTART_EPOCHS", "5"))   # 可 env 覆盖；多agent调查建议 k2-930 配 7(对齐历史~110步)
EARLY_STOP_PATIENCE = 2
# RFT（自蒸馏续训）
RFT_TOPN = 1
RFT_ACC_FLOOR = 0.6          # 选样准确率(漂移)硬门
RFT_LR = 3e-5
RFT_EPOCHS = 3
# DPO
DPO_BETA = 0.1
DPO_LR = 5e-6
DPO_EPOCHS = 1
DPO_MARGIN = 0.05            # chosen/rejected 的 R_human 最小差
# GRPO（swift + vLLM rollout + 每步 LoRA 权重同步）
GRPO_LR = 2e-6
GRPO_K = 8                   # 组大小（640G 资源富裕，比 14B 的 4 更大给更稳 advantage）
GRPO_STEPS = 200
GRPO_PROMPTS_PER_STEP = 16
GRPO_KL_THINK = 0.02         # think 段 KL（放开变自然）
GRPO_KL_ANSWER = 0.1         # answer 段 KL（重锚保准确率）——swift 盲区，见技术方案 §4.2 三策
GRPO_TEMPERATURE = 1.0
GRPO_TOP_P = 0.95
GRPO_MAX_LEN = 4096

# LoRA adapter 落点（各阶段）
_TAG = "v1-32b"
COLDSTART_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-coldstart-lora")
CS_RFT_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-cs-rft-lora")
DPO_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-dpo-lora")
GRPO_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-grpo-lora")
# GRPO 从两个基础各跑一遍做对比（RFT 当前最优 acc=0.813 / DPO 略降 0.804），看哪个起点 GRPO 后更好
GRPO_FROM_RFT_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-grpo-from-rft-lora")
GRPO_FROM_DPO_LORA_DIR = os.path.join(CKPT_DIR, f"{_TAG}-grpo-from-dpo-lora")

# 各阶段 rollout / 选样 / 偏好对 / 评测产物
CS_ROLLOUT = os.path.join(OUTPUT_DIR, "50_cs_rollout.jsonl")
CS_RFT_TRAIN = os.path.join(OUTPUT_DIR, "50_cs_rft_train.jsonl")
DPO_ROLLOUT = os.path.join(OUTPUT_DIR, "60_dpo_rollout.jsonl")
DPO_PAIRS = os.path.join(OUTPUT_DIR, "60_dpo_pairs.jsonl")


def stage_eval_paths(tag: str) -> tuple[str, str, str]:
    """某阶段在 224 验收集上的 (推理, 判分, 报告) 产物路径。"""
    return (
        os.path.join(OUTPUT_DIR, f"{tag}_infer.jsonl"),
        os.path.join(OUTPUT_DIR, f"{tag}_judge.jsonl"),
        os.path.join(OUTPUT_DIR, f"{tag}_report.md"),
    )


# 红线常量（阶段1 实测后写回；acc0 在 32B 口径下 ≈1.0，见技术方案 §7-阶段1）
ACC0 = float(os.environ.get("ACC0", "1.0"))
HUMANNESS0 = float(os.environ.get("HUMANNESS0", "0.0"))   # 阶段1 实测填入
EPSILON = float(os.environ.get("EPSILON", "0.03"))        # 守准容差
