"""Step 08（RL 阶段1a）：对 SFT_TRAIN 的每条 query，用当前学生模型采样 K 个回答并本地打分，落盘。

- 加载方式完全复用 step05（base+LoRA，已验证可跑）；--lora_dir 可指向 RFT/DPO 产出做迭代 rollout。
- 单 prompt + num_return_sequences=K（共享 prefill，省时且避开多 prompt 左 padding 的坑）。
- 断点续跑：按 query 去重，已采过的跳过。
- --limit 小样冒烟：先跑 2~3 条确认不崩，再放量。

产出 20_rft1_rollout.jsonl，每行：
{query, user_prompt, gold_answer, samples:[{text, reward, R_human, R_acc, gate, ...}, ...]}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RL_POOL, RL_ROLLOUT, STUDENT_LOCAL_DIR, STUDENT_LORA_DIR,
    RL_K, RL_ROLLOUT_MAX_QUERIES, RL_GEN_MAX_NEW_TOKENS, RL_TEMPERATURE, RL_TOP_P,
    REWARD_TAU_ACC, REWARD_W_HUMAN, REWARD_W_ACC, THINK_MIN_CHARS, THINK_MAX_CHARS,
)
from pipeline.logger import get_logger
from pipeline import reward as R

log = get_logger("step08_rollout")


def load_done(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    done = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query"])
            except Exception:
                continue
    return done


def load_student(lora_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    log.info("加载 base+LoRA: base=%s lora=%s", STUDENT_LOCAL_DIR, lora_dir)
    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return tok, model


def build_prompt_ids(tok, messages, device):
    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", default=STUDENT_LORA_DIR, help="rollout 用的 LoRA（默认 SFT 产出）")
    parser.add_argument("--pool", default=RL_POOL, help="query 池（默认 SFT_TRAIN；DPO 阶段指向扩充池）")
    parser.add_argument("--k", type=int, default=RL_K)
    parser.add_argument("--max_queries", type=int, default=RL_ROLLOUT_MAX_QUERIES, help="0=全量")
    parser.add_argument("--limit", type=int, default=None, help="冒烟用：只处理前 N 条")
    parser.add_argument("--out", default=RL_ROLLOUT)
    args = parser.parse_args()

    with open(args.pool, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.max_queries and args.max_queries > 0:
        samples = samples[: args.max_queries]
    if args.limit:
        samples = samples[: args.limit]

    done = load_done(args.out)
    todo = [s for s in samples if s.get("query") not in done]
    log.info("rollout 待处理 %d / %d（已完成 %d）K=%d", len(todo), len(samples), len(done), args.k)
    if not todo:
        log.info("无待处理样本，结束。")
        return

    tok, model = load_student(args.lora_dir)

    fout = open(args.out, "a", encoding="utf-8")
    n_ok = 0
    for i, s in enumerate(todo, 1):
        msgs = s["messages"][:2]          # system, user
        user_prompt = s["messages"][1]["content"]
        gold = (s.get("answer") or "").strip()
        prompt_ids = build_prompt_ids(tok, msgs, model.device)
        attn = torch.ones_like(prompt_ids)
        plen = prompt_ids.shape[1]

        # 一次并行生成 K 条（num_return_sequences=K）：共享一次 prefill（长 prompt 只算一次），
        # GPU 批量解码 K 路，比逐条快得多。128G 内存 + 96G 显存足够（14B 28G + K 路 KV ~数 G）。
        try:
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    attention_mask=attn,
                    do_sample=True,
                    temperature=RL_TEMPERATURE,
                    top_p=RL_TOP_P,
                    num_return_sequences=args.k,
                    max_new_tokens=RL_GEN_MAX_NEW_TOKENS,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            texts = [tok.decode(out[k, plen:], skip_special_tokens=True) for k in range(out.shape[0])]
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.error("query=%s... 生成失败：%r", (s.get("query") or "")[:30], e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        scored = []
        for text in texts:
            sc = R.score_rollout(
                text, user_prompt, gold,
                tau_acc=REWARD_TAU_ACC, w_human=REWARD_W_HUMAN, w_acc=REWARD_W_ACC,
                think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS,
            )
            sc["text"] = text
            scored.append(sc)
        if not scored:
            log.error("query=%s... 无有效样本，跳过", (s.get("query") or "")[:30])
            continue

        rec = {
            "query": s.get("query"),
            "user_prompt": user_prompt,
            "gold_answer": gold,
            "samples": scored,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_ok += 1
        if i % 10 == 0 or i == len(todo):
            best = max((x["reward"] for x in scored), default=0.0)
            log.info("[%d/%d] ok=%d 本条最高reward=%.3f", i, len(todo), n_ok, best)
    fout.close()
    log.info("rollout 完成：写入 %d 条 -> %s", n_ok, args.out)


if __name__ == "__main__":
    main()
