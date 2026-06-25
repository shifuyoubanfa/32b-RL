# vLLM 入门 + 百炼 vs ModelScope（教学）

## 一、vLLM 是什么
一个**高吞吐的 LLM 推理引擎**。同一个模型、同样的卡，vLLM 比你用 transformers 裸跑快好几倍。快在三件事：

| 技术 | 通俗解释 |
|---|---|
| **PagedAttention** | 把 KV cache（注意力缓存）像操作系统内存分页一样管理，碎片少、显存利用高 → 能同时塞更多并发请求 |
| **Continuous batching** | 请求来一个塞一个、动态拼批，GPU 一直满载（传统做法要"等一批凑齐"才算，GPU 常空转） |
| **Tensor Parallel (TP)** | 把一个装不进单卡的大模型**切到多张卡**上协同计算（我们 32B 用 TP=2，切到 GPU 0,1） |

## 二、vLLM 的两种用法
### ① 离线批量（offline）—— 进程内直接跑
`from vllm import LLM` → `LLM(model=...)` 加载一次 → `llm.chat([一批消息])` 一次性出结果。
适合："我有一堆数据要批量推理"。**没有服务、没有端口**。（见 `vllm_01_offline_demo.py`）

### ② 在线服务（serving）—— 这就是"起 vLLM"
`vllm serve <模型>` → 起一个 **OpenAI 兼容的 HTTP 服务**（默认 8000 端口），暴露：
- `GET  /v1/models`、`POST /v1/chat/completions`、`POST /v1/completions`

之后**任何程序**（我们的 `vllm_client.py`、curl、openai SDK）发 HTTP 请求就能用，**加载一次、反复调用**。
适合："当成一个 API 反复用"。（见 `vllm_02_serve_demo.sh` 起服务、`vllm_03_client_demo.py` 发请求）

> **"起 vLLM" = 跑 `vllm serve` 把模型加载进显存并起 HTTP 服务**。我们项目里 `serve_v1_vllm.sh` 干的就是这事。

## 三、常用启动参数（对着我们项目记）
```bash
vllm serve /path/to/V1 \
  --served-model-name v1 \          # 客户端 model 字段填这个名
  --tensor-parallel-size 2 \        # 切 2 张卡
  --dtype bfloat16 \                # 精度
  --max-model-len 4096 \            # 单条最长 token（prompt+生成）
  --gpu-memory-utilization 0.90 \   # 占每卡 90% 显存（权重+KV池）
  --enable-lora --lora-modules coldstart=/path/adapter   # 挂 LoRA，model 填 coldstart 即用
```
我们的 RL 全链里，vLLM 负责**所有推理/采样**：数据构建、各阶段评测、RFT/DPO 的 rollout。
而 GRPO 阶段比较特殊——swift **自带一个内置 vLLM** 做在线 rollout 并每步同步权重，所以那一步要先把我们这个静态服务停掉、腾卡。

---

## 四、百炼 vs ModelScope —— 一句话先记住
- **百炼（阿里云百炼 / DashScope）= 调云端 API**：不给你权重，你发请求、它返回结果，按 token 付费。**像 OpenAI API**。
- **ModelScope（魔搭）= 下载权重自己跑**：开源模型/数据集/代码的仓库社区，你把权重拉回本地、用自己的卡跑。**像 HuggingFace**。

| | 百炼 / DashScope | ModelScope（魔搭） |
|---|---|---|
| 本质 | **模型即服务**（托管推理 API） | **开源模型仓库 + 社区**（下载用） |
| 你拿到的 | 一个 **API endpoint + key** | **模型权重 / 数据集 / 代码** |
| 算力 | 阿里云的（你不管） | **你自己的 GPU** |
| 计费 | 按 token 付费 | 下载免费（算力你自己出） |
| 类比 | OpenAI API | HuggingFace Hub |
| 我们项目里 | **Kimi 判分/改写** 走百炼（DashScope OpenAI 兼容） | **ms-swift 训练框架**出自魔搭生态；V1 权重也是这类"拿回本地自己跑" |

### 我们项目同时用了两种范式（这点很值得理解）
- **V1（本地权重）→ vLLM 本地起服务**：自己的卡、自己跑（ModelScope 那种"拿回来自己跑"的范式）；
- **Kimi（当老师）→ 百炼 DashScope API 调用**：不下载 Kimi 权重，发 HTTP 请求拿它的判分/改写（百炼那种"调云端 API"的范式）。

也就是说：**"学生" V1 我们本地自托管（vLLM），"老师" Kimi 我们云端调 API（百炼）** —— 一个项目里两种模型使用方式各司其职。

> 补充：百炼上其实也能调 Qwen 系列、甚至上传数据做微调；ModelScope 也提供免费 Notebook 算力。但**核心区分不变**：百炼以"调 API"为主，ModelScope 以"下载自己跑"为主。
