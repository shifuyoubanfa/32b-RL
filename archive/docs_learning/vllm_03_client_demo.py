"""vLLM 教学 ③：客户端 —— 向已经"起好"的 vLLM 服务发请求。

前提：vllm_02_serve_demo.sh 已经在另一个终端跑起来、且就绪。
本文件【不加载模型】，只是发 HTTP 请求，所以用哪个 python 都行（连 requests 即可）。
  python 教学/vllm_03_client_demo.py

关键认知：vLLM 的服务接口和 OpenAI API 完全兼容 —— 所以你既能用 requests 裸发，
也能直接拿 openai 这个 SDK 把 base_url 指过来用。我们项目的 pipeline/vllm_client.py 就是 requests 版。
"""

import requests

BASE = "http://127.0.0.1:8000/v1"

# ① 先看服务里有哪些模型（确认服务活着、模型名对不对）
print("可用模型:", requests.get(f"{BASE}/models").json())

# ② 发一个 chat 请求（请求体格式 = OpenAI /chat/completions）
resp = requests.post(
    f"{BASE}/chat/completions",
    json={
        "model": "v1",                 # 对应 serve 时的 --served-model-name；挂了 adapter 就填 adapter 名
        "messages": [
            {"role": "system", "content": "你是税务助手。"},
            {"role": "user", "content": "小规模纳税人增值税征收率是多少？"},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        # "n": 4,                      # 一次要 4 个不同采样（我们 GRPO/rollout 就用 n=K 一次出多条）
    },
    timeout=120,
).json()

print("\n回答:\n", resp["choices"][0]["message"]["content"])

# ③ 等价地，用 openai SDK 也行（pip install openai）：
# from openai import OpenAI
# client = OpenAI(base_url=BASE, api_key="EMPTY")   # 本地服务不校验 key，填任意值
# r = client.chat.completions.create(model="v1",
#         messages=[{"role": "user", "content": "你好"}], temperature=0.7)
# print(r.choices[0].message.content)
