"""对照实验：定位"生成坍塌"是 base/模板问题 还是 LoRA 问题。

同一个 prompt 分别跑：
  A. base 单独（不挂 LoRA）+ 我们训练用的 chat template
  B. base 单独 + 极简裸 prompt（不套 template，验证模型本体是否正常）
  C. base + LoRA + 我们训练用的 chat template

并排看输出，就能判断：
  - 若 A/B 就崩  -> base 加载/dtype/环境问题（与 LoRA 无关）
  - 若 A/B 正常、C 崩 -> LoRA 挂载/训练把模型带崩了
  - 若 B 正常、A 崩 -> chat template 与该模型不兼容
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_LOCAL_DIR, STUDENT_LORA_DIR, SYSTEM_PROMPT
from pipeline.logger import get_logger

log = get_logger("sanity")

PROMPT = "请简要介绍小规模纳税人增值税的征收率。"


def gen(model, tok, input_ids):
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn,        # 显式传，避免 pad==eos 时推断不出 mask
            max_new_tokens=256,
            do_sample=False,            # greedy，排除采样随机性
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("加载 base ...")
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    templ_ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=False,
    )
    if hasattr(templ_ids, "input_ids"):
        templ_ids = templ_ids.input_ids
    templ_ids = templ_ids.to(base.device)

    # 打印 template 渲染出来的真实文本，肉眼看格式对不对
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    log.info("===== 生成时渲染（add_generation_prompt=True）=====\n%r", rendered)

    # 关键：训练时（带 assistant 目标、add_generation_prompt=False）模板怎么渲染。
    # 这决定 SFT 目标该不该保留开头的 <think>。
    demo_target = "<think>\n这是推理过程\n</think>\n\n<answer>\n这是答案\n</answer>"
    full_msgs = msgs + [{"role": "assistant", "content": demo_target}]
    full_render = tok.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    log.info("===== 训练时全量渲染（add_generation_prompt=False，带 assistant 目标）=====\n%r", full_render)

    raw_ids = tok(PROMPT, return_tensors="pt").input_ids.to(base.device)

    log.info("===== A. base + 训练用 template =====\n%r", gen(base, tok, templ_ids))
    log.info("===== B. base + 裸 prompt（无 template）=====\n%r", gen(base, tok, raw_ids))

    log.info("挂载 LoRA ...")
    lora_model = PeftModel.from_pretrained(base, STUDENT_LORA_DIR).eval()
    log.info("===== C. base+LoRA + 训练用 template =====\n%r", gen(lora_model, tok, templ_ids))

    log.info("对照完成。看哪一组崩了，对照脚本头部注释判断根因。")


if __name__ == "__main__":
    main()
