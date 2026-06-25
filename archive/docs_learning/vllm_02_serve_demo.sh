#!/usr/bin/env bash
# vLLM 教学 ②：在线服务 —— 这就是我们项目里说的"起 vLLM"。
#
# 【"起 vLLM" 到底是什么】
# 就是跑 `vllm serve <模型>`，它会：
#   1. 把模型加载进 GPU（和离线一样，2-3 分钟）；
#   2. 起一个【OpenAI 兼容的 HTTP 服务】，默认监听 8000 端口；
#   3. 暴露这些接口：
#        GET  /v1/models                # 看有哪些模型可用
#        POST /v1/chat/completions      # 聊天补全（和 OpenAI API 一模一样）
#        POST /v1/completions           # 文本补全
# 之后【任何程序】（我们的 vllm_client.py、curl、openai SDK）都能发 HTTP 请求来用它，
# 不用每次自己加载 65GB 模型——加载一次、反复调用，这就是"服务化"。
#
# 跑法（前台，会一直占着这个终端）：
#   bash 教学/vllm_02_serve_demo.sh
# 就绪后另开一个终端跑 vllm_03_client_demo.py 发请求。
set -euo pipefail

MODEL="/home/nvme01/zhjg/V1-32B/checkpoint-1500"
VLLM="/home/nvme02/biyh/vllm_env/bin/vllm"   # vLLM 命令在 vllm_env 里

CUDA_VISIBLE_DEVICES=0,1 \
"$VLLM" serve "$MODEL" \
  --served-model-name v1 \          # 客户端请求时 model 字段填 "v1"
  --tensor-parallel-size 2 \        # 切 2 张卡（对应 CUDA_VISIBLE_DEVICES 的 2 张）
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
# 就绪标志：日志出现 "Uvicorn running on http://0.0.0.0:8000"，
# 且 curl http://127.0.0.1:8000/v1/models 能返回 {"data":[{"id":"v1",...}]}。
#
# 【挂 LoRA adapter】（我们项目评测各阶段 adapter 就靠这个）：再加一行
#   --enable-lora --max-lora-rank 16 \
#   --lora-modules coldstart=/home/nvme01/zhjg/ckpts/.../checkpoint-165 \
# 这样客户端 model 填 "coldstart" 就用上了 V1+冷启动adapter；填 "v1" 就是纯底座。
