"""裸 base 14B 基线推理（独立脚本，不改动现有 step05）。

用途：评测【未做任何微调】的原始 DeepSeek-R1-Distill-Qwen-14B——给它"问题+参考资料"，
让它自由生成 think + answer，落盘成与 step05 完全相同的字段，直接喂 step06(Kimi判分)→step07(报告)。

为什么单独写：
- step05 没有"只加载裸 base、不挂任何 LoRA"的入口；
- 顺手做【批量左填充推理】把并发打高（比逐条快数倍）；
- 用显式 --eval_file，避开本地/服务器命名不一致（本地新名 00_data_sft_eval.jsonl / 服务器旧名 03_sft_eval.jsonl）。

输出每行：{query, student_raw, student_think, student_answer, teacher_think, teacher_answer}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SFT_EVAL,
    STUDENT_LOCAL_DIR,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
)
from pipeline.logger import get_logger

log = get_logger("eval_base14b")


def _extract_tag(tag: str, text: str):
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    i = text.find(open_t)
    j = text.rfind(close_t)
    if i == -1:
        return None
    start = i + len(open_t)
    inner = text[start:j] if (j != -1 and j > start) else text[start:]
    return inner.replace(open_t, "").replace(close_t, "").strip()


def parse_think_answer(text: str):
    """与 step05 同款解析：</think> 之前为 think，<answer>..</answer>(或其后全文)为 answer。
    裸 base 未必严格输出标签，这里有兜底，解析不出 answer 时取尾部全文。"""
    close = text.find("</think>")
    if close != -1:
        think = text[:close].replace("<think>", "").strip()
        tail = text[close + len("</think>"):]
    else:
        think = ""
        tail = text
    answer = _extract_tag("answer", tail)
    if answer is None:
        answer = tail.replace("<answer>", "").replace("</answer>", "").strip()
    return think, answer


def _prompt_ids(tok, msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default=SFT_EVAL,
                        help="测试集 jsonl(默认 SFT_EVAL；服务器命名不一致就显式指定，如 output/03_sft_eval.jsonl)")
    parser.add_argument("--out", required=True, help="输出 jsonl 路径(如 output/baseline_base14b_infer.jsonl)")
    parser.add_argument("--batch_size", type=int, default=8, help="批量推理条数(打高并发；显存不够就调小)")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条(冒烟)")
    args = parser.parse_args()

    log.info("加载【裸 base 14B】(不挂任何 LoRA): %s", STUDENT_LOCAL_DIR)
    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # 解码器生成必须左填充
    try:
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="sdpa",   # 省显存注意力，长 prompt 批量也放得下
        )
    except Exception as e:
        log.warning("sdpa 加载失败(%r)，回退默认注意力", e)
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    dev = model.device
    pad_id = tok.pad_token_id

    with open(args.eval_file, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        samples = samples[: args.limit]

    # 断点续跑：已写过的 query 跳过
    done = set()
    if Path(args.out).exists():
        with open(args.out, "r", encoding="utf-8") as f:
            for l in f:
                try:
                    done.add(json.loads(l)["query"])
                except Exception:
                    pass
    todo = [s for s in samples if s.get("query") not in done]
    log.info("待推理 %d / %d (已完成 %d) batch_size=%d", len(todo), len(samples), len(done), args.batch_size)
    if not todo:
        log.info("无待处理，结束。")
        return

    fout = open(args.out, "a", encoding="utf-8")
    bs = max(1, args.batch_size)
    n_done = 0
    for b0 in range(0, len(todo), bs):
        batch = todo[b0:b0 + bs]
        id_lists = [_prompt_ids(tok, s["messages"][:-1]) for s in batch]   # 去掉 teacher 的 assistant
        maxlen = max(len(x) for x in id_lists)
        input_ids, attn = [], []
        for x in id_lists:
            padn = maxlen - len(x)
            input_ids.append([pad_id] * padn + x)        # 左填充
            attn.append([0] * padn + [1] * len(x))
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=dev)
        attn = torch.tensor(attn, dtype=torch.long, device=dev)

        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids, attention_mask=attn,
                    max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                    temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
                    pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            log.error("batch OOM(从第 %d 条起)。把 --batch_size 调小重跑(已写的会断点续跑跳过)。", b0)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        for k, s in enumerate(batch):
            gen = tok.decode(out[k, maxlen:], skip_special_tokens=True)   # 左填充后所有样本生成段都从 maxlen 开始
            think, answer = parse_think_answer(gen)
            t_think = (s.get("reasoning") or "").strip()
            t_answer = (s.get("answer") or "").strip()
            if not t_think and not t_answer:
                t_think, t_answer = parse_think_answer(s["messages"][-1]["content"])
            rec = {
                "query": s.get("query"),
                "student_raw": gen,
                "student_think": think,
                "student_answer": answer,
                "teacher_think": t_think,
                "teacher_answer": t_answer,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_done += len(batch)
        log.info("[%d/%d]", n_done, len(todo))
    fout.close()
    log.info("写入: %s", args.out)


if __name__ == "__main__":
    main()
