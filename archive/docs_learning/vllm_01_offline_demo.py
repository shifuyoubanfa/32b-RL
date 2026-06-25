"""vLLM 教学 ①：离线批量推理（Offline Batched Inference）—— 不起服务，进程内直接跑。

【什么是 vLLM】
一个【高吞吐 LLM 推理引擎】。它快在三件事：
  1. PagedAttention：把 KV cache 像操作系统分页一样管理，显存利用率高、能塞更多并发；
  2. Continuous batching：动态把多个请求拼批，GPU 一直满载（不像传统 batch 要等齐）；
  3. Tensor Parallel：把一个大模型【切到多张卡】上一起算（我们 32B 切 2 张卡 = TP=2）。

【两种用法】
  ① 离线批量（本文件）：在 Python 里 `LLM(...)` 加载一次模型，喂一批 prompt 一次性出结果。适合"我有一堆数据要批量跑"。
  ② 在线服务（看 vllm_02/03）：`vllm serve` 起一个 HTTP 服务，别的程序发请求。适合"当个 API 用、反复调"。

运行环境：vLLM 装在 vllm_env，不在训练环境。
  /home/nvme02/biyh/vllm_env/bin/python 教学/vllm_01_offline_demo.py
"""

from vllm import LLM, SamplingParams

MODEL = "/home/nvme01/zhjg/V1-32B/checkpoint-1500"   # 本地权重目录（HF 格式）

# ① 加载模型（这一步最慢，要把 65GB 权重读进 2 张卡，约 2-3 分钟）
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,          # 把模型切到 2 张卡（TP=2）；单卡放得下就写 1
    dtype="bfloat16",                # 精度，A800 用 bf16
    gpu_memory_utilization=0.90,     # vLLM 占每张卡 90% 显存（权重 + KV cache 池）
    max_model_len=4096,              # 单条最长 token 数（prompt+生成）
)

# ② 采样参数：决定"怎么生成"
sp = SamplingParams(
    temperature=0.7,                 # 0=贪心(确定)，越大越随机
    top_p=0.9,
    max_tokens=512,                  # 最多生成多少 token
)

# ③ 方式 A：chat 接口（聊天模型用这个，自动套 chat 模板 <|im_start|> 那套）
messages_batch = [
    [{"role": "system", "content": "你是税务助手。"},
     {"role": "user", "content": "小规模纳税人增值税征收率是多少？"}],
    [{"role": "user", "content": "什么是进项税额抵扣？"}],
]
outputs = llm.chat(messages_batch, sp)   # 一次性批量跑这 2 条
for i, out in enumerate(outputs):
    print(f"\n=== 第{i+1}条 ===")
    print(out.outputs[0].text)           # out.outputs[0].text 就是生成的文本

# ③ 方式 B：completion 接口（直接喂纯文本 prompt，不套 chat 模板）
# raw_prompts = ["请用一句话解释增值税：", "印花税的纳税人是："]
# for out in llm.generate(raw_prompts, sp):
#     print(out.prompt, "->", out.outputs[0].text)
