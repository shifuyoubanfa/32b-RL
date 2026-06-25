"""Step 5: 用微调后的 7B 学生模型在 eval 集上推理，落盘 think / answer。

直接复用 SFT_EVAL 里准备好的 messages（system + user_prompt），生成 assistant。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SFT_EVAL,
    STUDENT_MERGED_DIR,
    STUDENT_LOCAL_DIR,
    STUDENT_LORA_DIR,
    STUDENT_OUTPUTS,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
)
from pipeline.logger import get_logger

log = get_logger("step05_infer")


def _extract_tag(tag: str, text: str) -> str | None:
    """取第一个 <tag> 到最后一个 </tag> 之间的内容，并清掉残留的同名嵌套标签。"""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    i = text.find(open_t)
    j = text.rfind(close_t)
    if i == -1:
        return None
    start = i + len(open_t)
    inner = text[start:j] if (j != -1 and j > start) else text[start:]
    inner = inner.replace(open_t, "").replace(close_t, "")
    return inner.strip()


def parse_think_answer(text: str) -> tuple[str, str]:
    """解析 R1 学生模型输出。

    R1 模板已在生成起点注入 <think>，所以 decode 出来的生成文本从【推理正文】开始，
    以 </think> 收束推理，随后是 <answer>...</answer>。因此：
      think  = </think> 之前的全部内容（去掉可能残留的 <think>）
      answer = <answer>...</answer> 内的内容；没有标签则取 </think> 之后的全文。
    """
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=STUDENT_MERGED_DIR,
        help="默认用合并后的目录；如未合并，可传 LoRA 目录或 base+adapter。",
    )
    parser.add_argument("--use_lora", action="store_true", help="不合并、直接挂 LoRA 推理")
    parser.add_argument("--lora_dir", default=STUDENT_LORA_DIR,
                        help="base+LoRA 模式用的 adapter 目录（RFT/DPO/GRPO 评测时指向各自产出）")
    parser.add_argument("--out", default=STUDENT_OUTPUTS, help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # 自动识别加载模式：
    # - 显式 --use_lora，或传了非默认 --lora_dir，或合并目录不全  -> base + LoRA
    # - 否则                                                      -> 已合并模型
    # 注意：合并若在 save_pretrained 阶段被 OOM 杀掉，会留下只有 config.json 的半成品目录，
    # 因此必须同时确认【权重】和【tokenizer】都齐全，否则视为未合并、回退 base+LoRA。
    md = Path(args.model_dir)
    has_weights = bool(list(md.glob("*.safetensors"))) or (md / "pytorch_model.bin").exists()
    has_tokenizer = (md / "tokenizer.json").exists() or (md / "tokenizer_config.json").exists()
    merged_ready = md.exists() and (md / "config.json").exists() and has_weights and has_tokenizer
    lora_overridden = args.lora_dir != STUDENT_LORA_DIR
    use_lora_mode = args.use_lora or lora_overridden or not merged_ready

    if use_lora_mode:
        from peft import PeftModel
        if not args.use_lora and not lora_overridden and not merged_ready:
            log.warning("未发现合并模型 %s，回退到 base+LoRA 模式", args.model_dir)
        log.info("以 base+LoRA 模式加载: base=%s lora=%s", STUDENT_LOCAL_DIR, args.lora_dir)
        tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, args.lora_dir)
    else:
        log.info("加载合并模型: %s", args.model_dir)
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(SFT_EVAL, "r", encoding="utf-8") as f:
        eval_samples = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        eval_samples = eval_samples[: args.limit]
    log.info("待推理: %d", len(eval_samples))

    fout = open(args.out, "w", encoding="utf-8")
    for i, sample in enumerate(eval_samples, 1):
        msgs = sample["messages"][:-1]  # 去掉 teacher 的 assistant
        prompt_ids = tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        # 兼容新版返回 dict / BatchEncoding 的情况
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        elif hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        prompt_ids = prompt_ids.to(model.device)
        attn_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                attention_mask=attn_mask,   # pad==eos 时必须显式传，否则 mask 推断不出
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        think, answer = parse_think_answer(gen)

        # teacher 的 think/answer 直接取 step03 存好的字段（不再解析，更稳）
        t_think = (sample.get("reasoning") or "").strip()
        t_answer = (sample.get("answer") or "").strip()
        if not t_think and not t_answer:  # 兼容旧格式样本
            t_think, t_answer = parse_think_answer(sample["messages"][-1]["content"])

        rec = {
            "query": sample.get("query"),
            "student_raw": gen,
            "student_think": think,
            "student_answer": answer,
            "teacher_think": t_think,
            "teacher_answer": t_answer,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if i % 10 == 0 or i == len(eval_samples):
            log.info("[%d/%d]", i, len(eval_samples))
    fout.close()
    log.info("写入: %s", args.out)


if __name__ == "__main__":
    main()
