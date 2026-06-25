"""Step 6: 调用公司内部满血 Claude（claude-opus-4-7）做评测。

每条 eval 记录都把 query / teacher_think / teacher_answer / student_think / student_answer
一并交给 Claude，要求它输出严格 JSON：

{
  "student_accuracy_score":      0~1,
  "student_accuracy_label":      "correct|partial|incorrect",
  "student_accuracy_reason":     "<60 字内>",

  "teacher_reasoning_humanness": 0~1,                       # think 像不像端到端 CoT
  "teacher_humanness_reason":    "<60 字内>",
  "teacher_rag_trace_types":     ["explicit_ref"|"verbatim_copy"|"ref_enumeration"|"policy_source"],

  "student_reasoning_humanness": 0~1,
  "student_humanness_reason":    "<60 字内>",
  "student_rag_trace_types":     [...]
}

设计要点：humanness 是连续分数，便于做分布、做 RL reward 的尺。
"""

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

# 公司内网，绕开本地系统代理
_session = requests.Session()
_session.trust_env = False
_session.proxies = {"http": "", "https": ""}

from config import (
    STUDENT_OUTPUTS,
    JUDGE_RESULTS,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    JUDGE_TIMEOUT,
    JUDGE_CALL_WORKERS,
    mudgate_headers,
)
from pipeline.logger import get_logger

log = get_logger("step06_judge")


_write_lock = threading.Lock()

JUDGE_SYSTEM = """你是一位严格、专业的税务领域大模型评测官。需要评估两个模型对同一个用户问题的回答：
- teacher = 公司微调模型（视为正确答案的基准）
- student = 7B 蒸馏模型

请输出两类指标。

# 指标一：学生答案准确率（仅评 student vs teacher）
以 teacher 的 answer 作为绝对正确，看 student 的 answer 在事实、税率/金额、适用条件、操作步骤上是否一致。
- correct：核心结论正确，关键条件/数值无误。
- partial：方向正确但缺少关键条件或某个数值偏差。
- incorrect：结论错或缺核心要点。

# 指标二：推理过程的"端到端 CoT 感"（reasoning_humanness）
对 teacher 和 student 的 think 段分别打 0~1 连续分。看的是推理过程像不像一个人/端到端模型自然推理出来的（而不是在对照检索资料）。

打分锚点：
- 0.9 ~ 1.0  完全像人/端到端 CoT。从问题出发自然推导，没有任何"查阅/对照参考资料"的气味。
- 0.6 ~ 0.8  推理整体自然，但偶尔有政策原文味道，或隐约暴露了"我查到了"的动作。
- 0.3 ~ 0.5  能明显感觉模型在对照外部材料：先列规定再套用、分点对比多条参考、显式说"参考资料显示"。
- 0.0 ~ 0.2  直接出现"参考问答对1/2/3"、"根据检索结果"等关键词，或大段一字不差照搬政策文本。

要点：
- 不要因为"提到了税法/政策"就扣分——这是任何人都会说的。
- 看的是"气质"：从问题向答案推导 vs 从资料向答案归纳。
- 即使没有任何 RAG 关键词，但大段照搬政策文本，humanness 也应较低。

# rag_trace_types 多标签（多选，无则空数组）
- "explicit_ref"    出现"参考问答对1/2"、"参考资料显示"、"根据检索结果"等明确引用语
- "verbatim_copy"   大段照搬政策原文且与口语化问答口吻不符
- "ref_enumeration" 依次罗列多条参考资料后归纳
- "policy_source"   显式标注"政策依据"、"参考文件"并给编号

# 输出
只输出一个 JSON 对象，不要任何额外文字、不要 markdown 代码块包裹：
{
  "student_accuracy_score": 0~1 小数,
  "student_accuracy_label": "correct" | "partial" | "incorrect",
  "student_accuracy_reason": "<60 字内>",

  "teacher_reasoning_humanness": 0~1 小数,
  "teacher_humanness_reason": "<60 字内>",
  "teacher_rag_trace_types": [<可空>],

  "student_reasoning_humanness": 0~1 小数,
  "student_humanness_reason": "<60 字内>",
  "student_rag_trace_types": [<可空>]
}
"""


def build_user(rec: dict) -> str:
    return (
        "【用户问题】\n" + (rec.get("query") or "") +
        "\n\n【公司微调模型 think（teacher）】\n" + (rec.get("teacher_think") or "") +
        "\n\n【公司微调模型 answer（teacher，视为正确）】\n" + (rec.get("teacher_answer") or "") +
        "\n\n【7B 学生模型 think】\n" + (rec.get("student_think") or "") +
        "\n\n【7B 学生模型 answer】\n" + (rec.get("student_answer") or "") +
        "\n\n请按要求输出一个 JSON。"
    )


def _extract_response_text(data) -> str:
    """从 mudgate 网关响应里抽 content。不同 vendor 路径返回壳子不一样。"""
    if not isinstance(data, dict):
        raise ValueError(f"响应不是 dict: type={type(data).__name__}")

    # 网关有可能 200 OK 但返回错误体
    if data.get("success") is False:
        ctx = data.get("errorContext") or data.get("error") or data
        raise ValueError(f"gateway success=False: {ctx}")
    err = data.get("error") or data.get("err")
    if isinstance(err, dict) or (isinstance(err, str) and ("error" in err.lower() or "fail" in err.lower())):
        raise ValueError(f"gateway returned error body: {err}")

    # 候选路径，按优先级
    def _try(d, *keys):
        cur = d
        for k in keys:
            if isinstance(k, int):
                if not isinstance(cur, list) or len(cur) <= k:
                    return None
                cur = cur[k]
            else:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
        return cur if isinstance(cur, str) and cur else None

    candidates = [
        _try(data, "choices", 0, "message", "content"),
        _try(data, "choices", 0, "text"),
        _try(data, "data", "choices", 0, "message", "content"),
        _try(data, "data", "choices", 0, "text"),
        _try(data, "data", "content"),
        _try(data, "data", "text"),
        _try(data, "data", "answer"),
        _try(data, "content"),
        _try(data, "text"),
        _try(data, "answer"),
        _try(data, "result"),
    ]
    for c in candidates:
        if c:
            return c
    raise ValueError(f"无法定位响应文本，顶层键={list(data.keys())[:10]}")


def _build_payload(messages: list[dict]) -> dict:
    """根据 JUDGE_MODEL 构造对应的请求体，统一处理模型私有约束。"""
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": JUDGE_TEMPERATURE,
        "top_p": JUDGE_TOP_P,
        "stream": False,
    }
    model_lower = JUDGE_MODEL.lower()
    if model_lower.startswith("kimi"):
        # Kimi K2 系列在 mudgate 网关上的强制约束（不可绕开，违反则 400）：
        # 1. 默认输出 <think> 段 → judge 场景纯浪费，关掉
        # 2. temperature 只接受 1
        # 3. top_p 只接受 0.95
        # 4. 关 do_sample → greedy 输出，弥补 1/0.95 采样带来的随机性
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["temperature"] = 1
        payload["top_p"] = 0.95
        payload["do_sample"] = False
    return payload


def call_judge(messages: list[dict], timeout: int = JUDGE_TIMEOUT, retries: int = 3) -> str:
    url = JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    headers = mudgate_headers()
    payload = _build_payload(messages)
    last_err = None
    last_body_snippet = None
    for attempt in range(retries):
        r = None
        try:
            r = _session.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return _extract_response_text(data)
        except Exception as e:
            last_err = e
            if r is not None:
                try:
                    last_body_snippet = (r.text or "")[:500]
                except Exception:
                    last_body_snippet = None
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"裁判模型 {JUDGE_MODEL} 调用失败: {last_err} | "
        f"url={url} status={getattr(r, 'status_code', 'N/A') if r else 'N/A'} | "
        f"body[:500]={last_body_snippet!r}"
    )


def extract_json(text: str) -> dict:
    """从模型输出里抽取 JSON 对象。允许模型有 ```json``` 包裹。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1:
        raise ValueError(f"无法定位 JSON 边界: {text[:200]}")
    return json.loads(text[l : r + 1])


def load_done(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["query"])
            except Exception:
                continue
    return out


def worker(rec: dict):
    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_user(rec)},
        ]
        raw = call_judge(messages)
        judged = extract_json(raw)
        return ("ok", {"query": rec.get("query"), **judged, "_judge_raw": raw}, None)
    except Exception as e:
        return ("err", {"query": rec.get("query")}, repr(e))


def probe():
    """单条 ping 网关，把完整响应打出来，方便排查接口形状。"""
    log.info("PROBE  url=%s model=%s", JUDGE_BASE_URL, JUDGE_MODEL)
    url = JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    payload = _build_payload([{"role": "user", "content": "只回复一个字：好"}])
    log.info("PROBE payload=%s", json.dumps(payload, ensure_ascii=False))
    r = _session.post(url, json=payload, headers=mudgate_headers(), timeout=60)
    log.info("PROBE status=%s", r.status_code)
    log.info("PROBE headers=%s", dict(r.headers))
    body = r.text or ""
    log.info("PROBE body (%d 字符):\n%s", len(body), body[:2000])
    try:
        data = r.json()
        log.info("PROBE parsed keys: %s", list(data.keys()) if isinstance(data, dict) else type(data).__name__)
        log.info("PROBE extracted text: %r", _extract_response_text(data))
    except Exception as e:
        log.error("PROBE 解析失败: %r", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=JUDGE_CALL_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--probe", action="store_true", help="只发一条测试请求，看响应壳子")
    parser.add_argument("--in", dest="in_path", default=STUDENT_OUTPUTS, help="待评测的 student 输出 jsonl")
    parser.add_argument("--out", default=JUDGE_RESULTS, help="评测结果输出 jsonl")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    with open(args.in_path, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    done = load_done(args.out)
    todo = [r for r in recs if r.get("query") not in done]
    log.info("待评测 %d / %d (已完成 %d) workers=%d model=%s url=%s in=%s",
             len(todo), len(recs), len(done), args.workers, JUDGE_MODEL, JUDGE_BASE_URL, args.in_path)

    fout = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    ok = err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, r): r for r in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, payload, msg = fut.result()
            if status == "ok":
                with _write_lock:
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    fout.flush()
                ok += 1
            else:
                err += 1
                log.error("query=%s... -> %s", (payload.get("query") or "")[:40], msg)
            if i % 10 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1e-3)
                log.info("[%d/%d] ok=%d err=%d %.2f q/s", i, len(futures), ok, err, rate)
    fout.close()
    log.info("完成: ok=%d err=%d 写入=%s", ok, err, args.out)


if __name__ == "__main__":
    main()
