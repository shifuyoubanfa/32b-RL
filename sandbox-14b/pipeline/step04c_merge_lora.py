"""Step 4c: 把 LoRA 合并回 base，方便后续推理直接 from_pretrained 加载。"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import STUDENT_LOCAL_DIR, STUDENT_LORA_DIR, STUDENT_MERGED_DIR
from pipeline.logger import get_logger

log = get_logger("step04c_merge")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info("加载 base: %s", STUDENT_LOCAL_DIR)
    # 在 GPU 上合并：96G 显存足够，避开之前 CPU 内存 OOM。
    # 显存不足的机器可把 device_map 改回 "cpu"（需 ≥64G 系统内存）。
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)

    log.info("挂载 LoRA: %s", STUDENT_LORA_DIR)
    model = PeftModel.from_pretrained(base, STUDENT_LORA_DIR)
    log.info("merge_and_unload ...")
    model = model.merge_and_unload()

    log.info("保存合并模型到 %s", STUDENT_MERGED_DIR)
    model.save_pretrained(STUDENT_MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(STUDENT_MERGED_DIR)
    log.info("完成")


if __name__ == "__main__":
    main()
