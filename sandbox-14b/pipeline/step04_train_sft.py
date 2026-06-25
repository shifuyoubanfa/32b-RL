"""Step 4b: LoRA SFT 蒸馏训练。

把公司微调模型作为 teacher，用其 (query+检索上下文 -> <think>+<answer>) 输出作为标签，
对 DeepSeek-7B-chat 做 LoRA 微调。只对 assistant 段计算 loss（mask 掉 prompt）。

注意：本脚本依赖 GPU + 单卡 ≥ 24G 显存（bf16 + LoRA + max_len=4096）。
显存吃紧时把 MAX_LEN 调到 2048 / 把 GRAD_ACC 调大 / 用 4bit 量化（需 bitsandbytes）。
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SFT_TRAIN,
    SFT_EVAL,
    STUDENT_LOCAL_DIR,
    STUDENT_LORA_DIR,
    MAX_LEN,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LR,
    BATCH_SIZE,
    GRAD_ACC,
    EPOCHS,
    LOG_DIR,
    EARLY_STOP_PATIENCE,
)
from pipeline.logger import get_logger

log = get_logger("step04_train")


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _apply_chat(tok, msgs, *, add_generation_prompt: bool, max_len: int | None = None) -> list[int]:
    """apply_chat_template 的版本兼容封装，统一返回 list[int]。max_len=None 时不截断。"""
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    if max_len is not None:
        kwargs.update(truncation=True, max_length=max_len)
    out = tok.apply_chat_template(msgs, **kwargs)
    if isinstance(out, dict):
        out = out["input_ids"]
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


class SFTDataset(Dataset):
    """R1 蒸馏 SFT：

    - prompt = chat_template(system+user, add_generation_prompt=True)，
      它会正确注入 R1 的 <think>\\n（和推理时完全一致）；这一段 mask 掉不算 loss。
    - target = "{reasoning}\\n</think>\\n\\n<answer>\\n{answer}\\n</answer>" + eos，手工拼，
      绕开 R1 模板对 assistant 内 <think>...</think> 的剥离；这一段算 loss。
    """

    def __init__(self, samples: list[dict], tokenizer, max_len: int):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        prompt_msgs = s["messages"][:2]  # system, user
        prompt_ids = _apply_chat(self.tok, prompt_msgs, add_generation_prompt=True)

        reasoning = (s.get("reasoning") or "").strip()
        answer = (s.get("answer") or "").strip()
        target_text = f"{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"
        target_ids = self.tok(target_text, add_special_tokens=False)["input_ids"]
        if self.eos_id is not None:
            target_ids = target_ids + [self.eos_id]

        # 长度控制：完整保留 target（含答案），prompt 超预算则从左截断（保留问题 + 注入的 <think>）
        budget = self.max_len - len(target_ids)
        if budget <= 0:
            target_ids = target_ids[: self.max_len]
            prompt_ids = []
        elif len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class PadCollator:
    pad_id: int

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


def main():
    import argparse
    import math
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # 可选参数：默认沿用 SFT 配置；RFT 阶段用 --train_file/--output_dir/--lr/--epochs 复用本脚本
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=SFT_TRAIN)
    parser.add_argument("--eval_file", default=SFT_EVAL)
    parser.add_argument("--output_dir", default=STUDENT_LORA_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--init_adapter", default=None,
                        help="从已有 LoRA 继续训(如 RFT 在冷启动 adapter 基础上续训)；默认 None=从 base 新建 LoRA")
    a = parser.parse_args()

    log.info("加载分词器和模型: %s", STUDENT_LOCAL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if a.init_adapter:
        # 从已有 adapter 继续训（RFT 在冷启动基础上续训，不丢已学到的自然推理）
        from peft import PeftModel
        log.info("从已有 adapter 继续训: %s", a.init_adapter)
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SFTDataset(load_jsonl(a.train_file), tokenizer, MAX_LEN)
    eval_ds = SFTDataset(load_jsonl(a.eval_file), tokenizer, MAX_LEN)
    log.info("train=%d eval=%d max_len=%d lr=%g bs=%d grad_acc=%d epochs=%g out=%s",
             len(train_ds), len(eval_ds), MAX_LEN, a.lr, BATCH_SIZE, GRAD_ACC, a.epochs, a.output_dir)

    args = TrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=a.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        # 多跑 epoch 安全网：每 epoch 评一次，结尾自动回退到 eval_loss 最低的 ckpt（而非最后一个 epoch）。
        # 前提：eval 集必须是自然腔同分布(COLDSTART_EVAL)，否则 eval_loss 方向相反、会选错(见 step14 注释)。
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # save_total_limit 提到 ≥ epoch 数，确保被选中的最优 ckpt 不会被轮转删掉（LoRA adapter 很小，多留无妨）。
        save_total_limit=max(2, math.ceil(a.epochs) + 1),
        report_to="none",
        logging_dir=LOG_DIR,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # transformers v5+ 把 tokenizer 参数改名成了 processing_class，做个兼容
    import inspect
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(pad_id=tokenizer.pad_token_id),
        # 连续 EARLY_STOP_PATIENCE 次 eval_loss 不降则停（配 load_best_model_at_end 回退到最优 ckpt）
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    log.info("LoRA 权重已保存到: %s", a.output_dir)


if __name__ == "__main__":
    main()
