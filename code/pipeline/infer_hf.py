"""transformers 兜底推理（vllm cu13 不可用时用它）：rft_merged 底座 + DPO adapter，
在验收集上贪心推理，输出格式严格对齐 step03_eval_infer.py，可直接喂 step_v2_eval.py 三分评测。

【为什么有这个脚本】2026-06-22 AutoDL 上 vLLM 起不来：vllm 0.20.1 的 _C 扩展是 CUDA 13
编译（要 libcudart.so.13），但配套 torch 是 cu128（自带 libcudart.so.12），错配。
AutoDL 的 pip 镜像又拿不到 cu13 的 manylinux wheel（两次 wheel build 失败）。
故放弃 vLLM，用训练栈 zhjg_rl（torch 2.10，GPU 健康）直接 transformers+peft 推理。

【用法】（从含 code/ 的父目录执行；先 export ZHJG_WORK_DIR 到实际工作目录）
  export ZHJG_WORK_DIR=<工作目录>   # 否则用 config 默认（/home/nvme01/zhjg），import 即建目录
  python -X utf8 code/pipeline/infer_hf.py \
    --base    <rft_merged 底座目录> \
    --adapter <DPO adapter checkpoint 目录> \
    --eval_file <验收集.jsonl> \
    --out     <输出.jsonl>

  --model_name 默认 v2-dpo-derag2（非 v1 → system_for 给去检索腔 prompt；用 v1 会套 RAG 腔，分全错）。

输出兼容 step03_eval_infer.py：query / user_prompt / gold_answer / gen_text / think / answer，另带 format_ok。
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # code/ 进 path
os.environ.setdefault("ZHJG_WORK_DIR", "/root/autodl-tmp/dpo")  # 兜底；正式跑请在外部 export

from config import system_for, GEN_MAX_NEW_TOKENS

try:
    from pipeline.reward import parse_think_answer_diagnostic  # 复用线上诊断解析，格式失败不伪装成答案
except Exception:
    def parse_think_answer_diagnostic(raw):
        raw = raw or ""
        close = raw.find("</think>")
        if close != -1:
            think = raw[:close].replace("<think>", "").strip()
            tail = raw[close + len("</think>"):]
            i = tail.find("<answer>")
            if i != -1:
                j = tail.rfind("</answer>")
                inner = tail[i + len("<answer>"): j] if (j != -1 and j > i) else tail[i + len("<answer>"):]
                answer = inner.replace("<answer>", "").replace("</answer>", "").strip()
            else:
                answer = tail.replace("<answer>", "").replace("</answer>", "").strip()
        else:
            think = raw.replace("<think>", "", 1).strip() if "<think>" in raw else ""
            answer = ""
        reasons = (["missing_think_close"] if close == -1 else [])
        reasons += (["empty_think"] if not think else []) + (["empty_answer"] if not answer else [])
        return {"think": think, "answer": answer, "format_ok": not reasons,
                "format_reason": "+".join(reasons) if reasons else "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="rft_merged 全量底座目录")
    ap.add_argument("--adapter", required=True, help="DPO LoRA adapter（checkpoint-N 目录）")
    ap.add_argument("--model_name", default="v2-dpo-derag2", help="决定 system prompt；非 v1 → 去检索腔")
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true", help="忽略已有输出并从头跑；默认按 query 断点续跑")
    args = ap.parse_args()

    base = Path(args.base)
    adapter = Path(args.adapter)
    eval_file = Path(args.eval_file)
    out_path = Path(args.out)
    if not (base / "config.json").is_file():
        raise SystemExit(f"[infer] base 缺 config.json: {base}")
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"[infer] adapter 缺 adapter_config.json: {adapter}")
    if not eval_file.is_file():
        raise SystemExit(f"[infer] eval_file 不存在: {eval_file}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    # 默认断点续跑：每条写完立即 flush；重启后按 query 跳过已完成项。若已有文件损坏则明确停，避免静默漏题。
    done_queries = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fin:
            for lineno, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"[infer] 已有输出第 {lineno} 行损坏；修复或加 --overwrite: {exc}")
                if old.get("query") is not None:
                    done_queries.add(old["query"])

    # 所有轻量校验都放在 65G 模型加载前；已跑完时直接终检退出，不白等模型加载。
    with eval_file.open(encoding="utf-8") as fin:
        recs = [json.loads(l) for l in fin if l.strip()]
    if not recs:
        raise SystemExit(f"[infer] eval_file 为空: {eval_file}")
    queries = [r.get("query") for r in recs]
    query_set = set(queries)
    if any(q is None for q in queries) or len(query_set) != len(queries):
        raise SystemExit("[infer] eval_file 的 query 缺失或重复，无法安全按 query 断点续跑")
    extra_done = done_queries - query_set
    if extra_done:
        raise SystemExit(f"[infer] 已有输出含 {len(extra_done)} 条不属于当前 eval_file 的 query；请换输出路径或加 --overwrite")
    pending = [r for r in recs if r.get("query") not in done_queries]
    print(f"[infer] total={len(recs)} done={len(done_queries)} pending={len(pending)}", flush=True)

    if not pending:
        print(f"[infer] already complete -> {out_path} rows={len(done_queries)}", flush=True)
        return

    # 重依赖延迟到所有文件/续跑校验之后再导入，避免路径写错也先加载训练栈。
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    sys_prompt = system_for(args.model_name)
    print(f"[infer] system={'RAG腔' if args.model_name=='v1' else '去检索腔'} adapter={args.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    print("[infer] loading base (~65G, device_map=auto 自动切多卡)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    device = next(model.parameters()).device
    print(f"[infer] model ready, device={device}", flush=True)

    with out_path.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(pending):
            up = rec.get("user_prompt") or ""
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": up}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tok.eos_token_id)  # 贪心，对齐评测 temp=0
            raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            parsed = parse_think_answer_diagnostic(raw)
            fout.write(json.dumps({"query": rec.get("query"), "user_prompt": up,
                "gold_answer": rec.get("answer", ""), "gen_text": raw,
                "think": parsed["think"], "answer": parsed["answer"],
                "format_ok": parsed["format_ok"],
                "format_reason": parsed["format_reason"]}, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[infer] new={i+1}/{len(pending)} total_done={len(done_queries)+i+1}/{len(recs)}", flush=True)

    # 完整性终检：输出必须恰好覆盖验收集全部 query；空答案单列告警，后续在池规则会判失败。
    final_rows = []
    with out_path.open(encoding="utf-8") as fin:
        final_rows = [json.loads(l) for l in fin if l.strip()]
    final_queries = {r.get("query") for r in final_rows}
    missing = set(queries) - final_queries
    extra = final_queries - query_set
    duplicates = len(final_rows) - len(final_queries)
    if missing or extra or duplicates:
        raise SystemExit(f"[infer] 输出完整性失败 missing={len(missing)} extra={len(extra)} duplicates={duplicates}")
    total_empty = sum(1 for r in final_rows if not (r.get("answer") or "").strip())
    print(f"[infer] done -> {out_path} rows={len(final_rows)} empty_answers={total_empty}", flush=True)


if __name__ == "__main__":
    main()
