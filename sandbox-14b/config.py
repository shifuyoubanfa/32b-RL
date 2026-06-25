"""项目集中配置：路径、接口、采样、训练超参。"""

import os

# ---------- 路径 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CKPT_DIR = os.path.join(ROOT_DIR, "ckpts")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

for d in (OUTPUT_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

RAW_XLSX = os.path.join(DATA_DIR, "A模型纯样本汇总.xlsx")
SUMMARY_XLSX = os.path.join(DATA_DIR, "一阶段模型输出3.31-5.19 v2.xlsx")
USABLE_QUERIES = os.path.join(OUTPUT_DIR, "01_usable_queries.jsonl")
SUMMARY_QUERIES = os.path.join(OUTPUT_DIR, "01b_summary_queries.jsonl")
ALL_QUERIES = os.path.join(OUTPUT_DIR, "01c_all_queries.jsonl")
COMPANY_OUTPUTS = os.path.join(OUTPUT_DIR, "02_company_outputs.jsonl")
SFT_TRAIN = os.path.join(OUTPUT_DIR, "03_sft_train.jsonl")
SFT_EVAL = os.path.join(OUTPUT_DIR, "03_sft_eval.jsonl")
STUDENT_OUTPUTS = os.path.join(OUTPUT_DIR, "05_student_outputs.jsonl")
JUDGE_RESULTS = os.path.join(OUTPUT_DIR, "06_judge_results.jsonl")
REPORT_MD = os.path.join(OUTPUT_DIR, "07_report.md")

# ---------- 公司内部接口（来自 rag_demo.py） ----------
RETRIEVE_URL = "http://mlp.paas.dc.servyou-it.com/agentic_system_service/rag/getRetrieve"
COMPANY_CHAT_URL = "http://mlp.paas.dc.servyou-it.com/llm_finetune/v1/chat/completions"

# ---------- 公司内部 mudgate 网关 ----------
# 鉴权方式：同一把 key 同时塞到三个 header（Authorization / app_id / appId）
# 默认 key 见 local_llm_runtime.json；可通过环境变量 MUDGATE_API_KEY 覆盖
MUDGATE_API_KEY = os.environ.get("MUDGATE_API_KEY", "sk-6b15b039aed34bbe8750fd1c7f26c9c7")
MUDGATE_BASE = "http://mlp.paas.dc.servyou-it.com/mudgate/api/llm"


def mudgate_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": MUDGATE_API_KEY,
        "app_id": MUDGATE_API_KEY,
        "appId": MUDGATE_API_KEY,
    }


# ---------- 评测裁判模型 ----------
# kimi-k2.6（Moonshot）262K 上下文，跨家选 judge 规避同源 bias
# 服务地址来自公司模型广场截图：mudgate/api/llm/moonshot/v1
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", f"{MUDGATE_BASE}/moonshot/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "kimi-k2.6")

JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 0.7
JUDGE_TIMEOUT = 600

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

# ---------- 学生模型（魔搭 ModelScope） ----------
# 选用 R1 蒸馏推理模型：原生输出 <think>...</think> 推理链，契合本项目"端到端 CoT"目标。
# 之前的 deepseek-llm-7b-chat(2023 纯指令模型) 不会稳定输出 think，已弃用。
STUDENT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
_STUDENT_TAG = "ds-r1-qwen-14b"
STUDENT_LOCAL_DIR = os.path.join(CKPT_DIR, _STUDENT_TAG)
STUDENT_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-lora")
STUDENT_MERGED_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-merged")

# ---------- 训练超参 ----------
TRAIN_EVAL_RATIO = 0.9
MAX_LEN = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 1e-4
# RTX PRO 6000 (96G) 显存充足：把 per-device batch 提到 2、grad_acc 降到 8，
# 等效 batch 仍为 16（保持 LR 有效），更充分利用大显存、训练更快。
BATCH_SIZE = 2
GRAD_ACC = 8
EPOCHS = 3

# ---------- 调用并发 ----------
COMPANY_CALL_WORKERS = 4
JUDGE_CALL_WORKERS = 4

# ---------- 推理超参 ----------
GEN_MAX_NEW_TOKENS = 1536
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.9


# ==================== 强化学习（RL）阶段 ====================
# 目标：把 think 优化得更像人（humanness↑），同时靠"按住答案"保住准确率。
# 路线：阶段0 reward 校准 -> 阶段1 RFT 自蒸馏 -> 阶段2 DPO -> 阶段3 GRPO。
# 所有阶段共用 reward.py 打分；rollout 用 HF generate（复用 step05 已验证的加载/生成）。

# rollout 来源：严格用 SFT_TRAIN（绝不碰 SFT_EVAL，eval 留作验收）
RL_POOL = SFT_TRAIN
RL_ROLLOUT = os.path.join(OUTPUT_DIR, "08_rl_rollout.jsonl")
RFT_TRAIN = os.path.join(OUTPUT_DIR, "08b_rft_train.jsonl")
CALIB_REPORT = os.path.join(OUTPUT_DIR, "08c_calibration.md")
DPO_PAIRS = os.path.join(OUTPUT_DIR, "09_dpo_pairs.jsonl")

RFT_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-rft-lora")
DPO_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-dpo-lora")
GRPO_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-grpo-lora")

# 各 RL 阶段在 eval 集上的推理/评测/报告产物（复用 step05/06/07，--lora_dir/--out 指定）
RFT_STUDENT_OUTPUTS = os.path.join(OUTPUT_DIR, "08d_rft_student_outputs.jsonl")
RFT_JUDGE_RESULTS = os.path.join(OUTPUT_DIR, "08e_rft_judge_results.jsonl")
RFT_REPORT = os.path.join(OUTPUT_DIR, "08f_rft_report.md")
DPO_STUDENT_OUTPUTS = os.path.join(OUTPUT_DIR, "09d_dpo_student_outputs.jsonl")
DPO_JUDGE_RESULTS = os.path.join(OUTPUT_DIR, "09e_dpo_judge_results.jsonl")
DPO_REPORT = os.path.join(OUTPUT_DIR, "09f_dpo_report.md")
GRPO_STUDENT_OUTPUTS = os.path.join(OUTPUT_DIR, "10d_grpo_student_outputs.jsonl")
GRPO_JUDGE_RESULTS = os.path.join(OUTPUT_DIR, "10e_grpo_judge_results.jsonl")
GRPO_REPORT = os.path.join(OUTPUT_DIR, "10f_grpo_report.md")

# rollout 采样（128G 内存 + 96G 显存充足，并行生成 K 条、共享 prefill）
RL_K = 8                       # 每条 query 采样数（内存充足，多采样给 RFT 更多优质样本可挑）
RL_ROLLOUT_MAX_QUERIES = 400   # 默认只对前 N 条 query 采样（实验够用，控制时长）；设 0 表示全量
RL_GEN_MAX_NEW_TOKENS = 1024
RL_TEMPERATURE = 0.9           # 比评测的 0.6 高，拉开多样性给筛选留空间
RL_TOP_P = 0.95

# reward 阈值/权重（已由 step08c --tune 在 224 条对齐样本上搜出：
#   humanness 本地均值 0.250 ≈ Kimi 0.212；准确率门控一致率 83.5%）
REWARD_TAU_ACC = 0.30      # tune 最优：一致率 83.5%（降到 0.30 减少误杀"换说法的正确答案"）
REWARD_W_HUMAN = 0.5
REWARD_W_ACC = 0.5
REWARD_C_TRACE = 0.34      # tune 最优：每处 RAG 引用痕迹的扣分
REWARD_C_COPY = 1.00       # tune 最优：照抄率的扣分系数
THINK_MIN_CHARS = 40
THINK_MAX_CHARS = 2000

# RFT（拒绝采样自蒸馏，复用 step04 训练，lr 调低防过拟合自采样本）
RFT_TOPN = 1                   # 每条 query 选前 N 个做新训练样本
RFT_ACC_FLOOR = 0.6            # 选样准确率硬门槛：只在 R_acc≥此值的样本里挑（压住准确率下滑）
RFT_LR = 3e-5
RFT_EPOCHS = 3              # 2→3（上限 4，比冷启动卡更死）。RFT 在自生成数据上续训，过训易自我强化/模式坍塌，
                            # 第一版 RFT 已因 Goodhart 翻车(acc 93.3→85.3)，故保守；最终以 Kimi 验收为准，非 eval_loss。

# ---------- 冷启动种子（rewrite teacher think → 自然推导）----------
# 一份两用：① 验证奖励的"该高分"对照 ② cold-start SFT 种子。改写/打分都走 Kimi(离线、不进训练环)。
SEEDS_RAW = os.path.join(OUTPUT_DIR, "11_natural_seeds.jsonl")       # 改写后的自然 think
SEEDS_SCORED = os.path.join(OUTPUT_DIR, "12_seeds_scored.jsonl")     # + Kimi humanness 标签
SEED_MAX = 800                 # 先改写前 N 条 SFT_TRAIN（实验够用、控制时长）；0=全量
SEED_WORKERS = 4               # Kimi 并发（与 JUDGE_CALL_WORKERS 同量级）

# ---------- 嵌入模型（公司内网 bge-m3 API）----------
# 仅用于 step13 信号证伪探针；正式奖励不依赖嵌入(证伪显示嵌入信号弱，AUC~0.55，已弃用)。
EMBED_URL = os.environ.get("EMBED_URL", "http://mlp.paas.dc.servyou-it.com/text-embedding-bge/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
EMBED_TIMEOUT = 30
EMBED_MAX_RETRIES = 3

# ---------- 冷启动 → RFT 整链 ----------
# 证伪结论：奖励 humanness = s_trace(关键词,AUC0.74) + 字符照抄 + 【可选 PMI(结构,AUC0.73)】；嵌入信号已弃。
SEED_HUMANNESS_MIN = 0.60        # 冷启动只用 Kimi humanness≥此值的高质量自然种子。
                                 # 0.40→0.60：种子分布重度左偏(均值0.64/中位0.80)，弱尾很薄——
                                 # 抬到 0.60 只从 593→525(少~68条)，却砍掉全部"低于均值"的将就样本，
                                 # 几乎白捡的纯度提升(冷启动是教风格，范例必须集中在"像人"那一档)。
COLDSTART_TRAIN = os.path.join(OUTPUT_DIR, "14_coldstart_train.jsonl")
# 自然腔留出验证集：从同一批自然种子里切 ~10% 出来。必须与训练同分布(自然 think)，
# 否则用 SFT_EVAL(teacher 机器腔)当 eval，eval_loss 会随"变自然"反而升高，早停/留最优会朝反方向选(已被对抗验证抓到)。
COLDSTART_EVAL = os.path.join(OUTPUT_DIR, "14_coldstart_eval.jsonl")
COLDSTART_EVAL_FRAC = 0.10       # 留出比例(每 1/FRAC 条抽 1 条进 eval，确定性切分，可复现)
COLDSTART_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-coldstart-lora")
COLDSTART_LR = 5e-5
COLDSTART_EPOCHS = 5             # 3→5（上限 8）。~784 条/eff_batch16 ≈ 49 步/epoch，3 epoch 偏少；
                                 # 真正的安全靠"每 epoch 评+留最优 eval_loss+早停"，不是 epoch 数本身。
EARLY_STOP_PATIENCE = 2          # 连续 N 次 eval_loss 不降则停（配 load_best_model_at_end 回退到最优 ckpt）

# 冷启动模型自生成 → 新奖励选样 → 在冷启动基础上 RFT
CS_ROLLOUT = os.path.join(OUTPUT_DIR, "15_cs_rollout.jsonl")
CS_RFT_TRAIN = os.path.join(OUTPUT_DIR, "16_cs_rft_train.jsonl")
CS_RFT_LORA_DIR = os.path.join(CKPT_DIR, f"{_STUDENT_TAG}-cs-rft-lora")
CS_RFT_OUTPUTS = os.path.join(OUTPUT_DIR, "17_cs_rft_student_outputs.jsonl")
CS_RFT_JUDGE = os.path.join(OUTPUT_DIR, "18_cs_rft_judge_results.jsonl")
CS_RFT_REPORT = os.path.join(OUTPUT_DIR, "19_cs_rft_report.md")

REWARD_W_PMI = 0.5               # 开 PMI 时 humanness 里 PMI 的权重（其余给 s_trace+照抄基础项）

# ---------- 扩 query（防过拟合 + 证泛化）：摄取从未用过的生产问题，扩充 RL 训练池 ----------
NEW_QUERY_XLSX = os.path.join(DATA_DIR, "一阶段模型输出3.31-5.19 v2.xlsx")  # 1773 条生产问题(带答案+满意度)
RL_POOL_EXPANDED = os.path.join(OUTPUT_DIR, "20_rl_pool_expanded.jsonl")    # 老训练池 + 新 query(去重、排验收集)
DPO_ROLLOUT = os.path.join(OUTPUT_DIR, "21_dpo_rollout.jsonl")              # RFT 模型在扩充池上自采(供构 DPO 对)

# DPO（手写实现，基于 transformers+peft，不依赖 trl）
# 关键：DPO 的 πref = 起点策略(CS_RFT)，训练前预计算并冻结缓存（见 step09）；不是裸 base。
DPO_BETA = 0.1
DPO_LR = 5e-6
DPO_EPOCHS = 1
DPO_MAX_PAIRS = 0              # 0 表示用全部可构造的偏好对
DPO_MARGIN = 0.05            # chosen/rejected 的 R_human 最小差。CS_RFT 已很自然(0.78)方差小，
                             # 用 0.05(不是0.1)做安全余量：实测旧 rollout 在 0.1 出 279 对、0.05 出 340 对，0.05 更稳不会配对饥饿。

# GRPO（手写实现，非对称 KL：think 段放开、answer 段重锚）
GRPO_LR = 2e-6
GRPO_K = 4                    # 6→4：显存余量(generate 4 路 vs 6 路)+加快生成；K=4 仍够算组内 advantage。
GRPO_STEPS = 100             # 200→100：实测 ~8-10 分钟/步，200 步要 ~30h 太长；100 步够看 GRPO 是否再加分，每 50 步存档可早停。
GRPO_PROMPTS_PER_STEP = 8
GRPO_KL_THINK = 0.02          # think 段 KL 系数（基本放开变自然；留一丝>0 防"只奖humanness+在线采样"200步发散/奖励黑客。设 0 为纯放开冒烟）
GRPO_KL_ANSWER = 0.1          # answer 段 KL 系数（重锚，保住准确率）。锚到税务策略(CS_RFT/DPO)而非裸 base，见 step10 --ref_lora
GRPO_TEMPERATURE = 1.0        # GRPO 需要组内方差
GRPO_TOP_P = 0.95
GRPO_MAX_FORWARD_LEN = 3072   # 训练前向(policy/ref)的序列上限：超长 prompt 从左截断，防单条长序列 OOM(单卡96G)
