"""Step 09（RL 阶段2）：手写 DPO 偏好对齐（基于 transformers+peft，不依赖 trl）。

从 rollout 里构造同 query 的偏好对：
- chosen   = gate==ok（答案没漂移）里 reward 最高的（通常更自然）
- rejected = gate==ok（答案同样没漂移）里 R_human 最低的（RAG 痕迹最重）
两者答案都正确，差别只在 think 风格 -> DPO 只学"改思考"，学不会"改答案"。

DPO 损失：L = -logσ( β·[(logπ_chosen - logπ_ref_chosen) - (logπ_rejected - logπ_ref_rejected)] )
ref 用 peft disable_adapter 复用同一冻结 base，零额外权重。

--smoke：只用少量对、1 epoch，几分钟验证不崩。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as Fnn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RL_ROLLOUT, DPO_PAIRS, DPO_LORA_DIR, STUDENT_LOCAL_DIR, STUDENT_LORA_DIR,
    RFT_LORA_DIR, DPO_BETA, DPO_LR, DPO_EPOCHS, DPO_MAX_PAIRS, DPO_MARGIN, MAX_LEN, SYSTEM_PROMPT,
)
from pipeline.logger import get_logger

log = get_logger("step09_dpo")


def build_pairs(rollout_path, margin=0.1, max_pairs=0):
    with open(rollout_path, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    pairs = []
    n_too_few = n_too_close = 0
    for rec in recs:
        ok = [s for s in rec.get("samples", []) if s.get("gate") == "ok"]
        if len(ok) < 2:
            n_too_few += 1
            continue
        chosen = max(ok, key=lambda x: x.get("reward", -1))
        rejected = min(ok, key=lambda x: x.get("R_human", 1.0))
        if chosen is rejected:
            n_too_few += 1
            continue
        if (chosen.get("R_human", 0) - rejected.get("R_human", 0)) < margin:
            n_too_close += 1
            continue
        pairs.append({
            "query": rec.get("query"),
            "user_prompt": rec.get("user_prompt", ""),
            "chosen": chosen.get("text", ""),
            "rejected": rejected.get("text", ""),
        })
    # 配对产出率诊断：CS_RFT 已很自然(0.78)时组内 R_human 方差小，margin 太大会大量滤掉 → 看这行决定是否调小 --margin
    log.info("配对: 从 %d 条 query 构造 %d 对 (产出率 %.1f%%, margin=%.2f); 丢弃: 合格样本<2 %d / R_human差<margin %d",
             len(recs), len(pairs), 100.0 * len(pairs) / max(1, len(recs)), margin, n_too_few, n_too_close)
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def prompt_ids_of(tok, user_prompt, device):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids.to(device)


def full_ids_of(tok, prompt_ids, completion_text, device, eos_id):
    comp = tok(completion_text, add_special_tokens=False)["input_ids"]
    if eos_id is not None and (not comp or comp[-1] != eos_id):
        comp = comp + [eos_id]
    comp_t = torch.tensor([comp], dtype=torch.long, device=device)
    full = torch.cat([prompt_ids, comp_t], dim=1)
    if full.shape[1] > MAX_LEN:                       # 极端长度兜底：保留 prompt 头 + 截尾
        full = full[:, :MAX_LEN]
    return full, prompt_ids.shape[1]


def completion_logprob(model, full_ids, prompt_len):
    """对 full_ids 里 completion 段的逐 token logprob 求和。"""
    out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
    logits = out.logits[:, :-1, :]                    # 预测位置 1..T-1
    targets = full_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)   # [1, T-1]
    comp = tok_logp[:, prompt_len - 1:]               # completion 段
    return comp.sum(dim=1).squeeze(0)                 # 标量


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", default=RL_ROLLOUT)
    parser.add_argument("--init_lora", default=None, help="DPO 起点 adapter（默认优先 RFT，其次 SFT）")
    parser.add_argument("--out_dir", default=DPO_LORA_DIR)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument("--lr", type=float, default=DPO_LR)
    parser.add_argument("--epochs", type=int, default=DPO_EPOCHS)
    parser.add_argument("--max_pairs", type=int, default=DPO_MAX_PAIRS)
    parser.add_argument("--margin", type=float, default=DPO_MARGIN,
                        help="chosen/rejected 的 R_human 最小差；CS_RFT 已很自然时方差小，配对少就调小(如 0.05)")
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="少量对 + 1 epoch 冒烟")
    args = parser.parse_args()

    init_lora = args.init_lora
    if init_lora is None:
        init_lora = RFT_LORA_DIR if Path(RFT_LORA_DIR, "adapter_config.json").exists() else STUDENT_LORA_DIR
    log.info("DPO 起点 adapter: %s", init_lora)

    pairs = build_pairs(args.rollout, margin=args.margin, max_pairs=(2 if args.smoke else args.max_pairs))
    Path(DPO_PAIRS).write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    log.info("构造偏好对 %d，落盘 %s", len(pairs), DPO_PAIRS)
    if not pairs:
        log.error("没有可用偏好对（每条 query 需 ≥2 个 gate==ok 且 R_human 差距≥0.1 的样本）。检查 rollout。")
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, init_lora, is_trainable=True)
    model.gradient_checkpointing_enable()
    # grad-ckpt + LoRA：必须让输入 require_grad，否则梯度传不到 adapter
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    epochs = 1 if args.smoke else args.epochs
    eos_id = tok.eos_token_id
    dev = model.device

    # ===== 关键：πref = 起点策略(CS_RFT)，不是裸 base =====
    # DPO 的参考模型必须等于策略初始化(init_lora)，否则失去对 CS_RFT 的 KL 锚、把模型往别处拉。
    # 不能用 disable_adapter()(那是裸 base)；也不能"用同一可训 adapter 现算 ref"(那样 πref≡πθ → logits≡0 → 不学)。
    # 正确做法：训练前用【冻结的 CS_RFT adapter】把每对的 chosen/rejected 参考 logprob 预计算并缓存(πref 固定不变)。
    log.info("预计算 πref(冻结 CS_RFT 起点策略) 的参考 logprob，共 %d 对 …", len(pairs))
    model.eval()
    ref_cache = []
    with torch.no_grad():
        for p in pairs:
            pids = prompt_ids_of(tok, p["user_prompt"], dev)
            ch_full, ch_plen = full_ids_of(tok, pids, p["chosen"], dev, eos_id)
            rej_full, rej_plen = full_ids_of(tok, pids, p["rejected"], dev, eos_id)
            ref_cache.append((
                completion_logprob(model, ch_full, ch_plen).detach(),
                completion_logprob(model, rej_full, rej_plen).detach(),
            ))
    model.train()

    step = 0
    opt.zero_grad()
    for ep in range(epochs):
        n_in_acc = 0
        for i, p in enumerate(pairs, 1):
            lp_ch_ref, lp_rej_ref = ref_cache[i - 1]      # 冻结的 CS_RFT 参考（不随训练变）
            pids = prompt_ids_of(tok, p["user_prompt"], dev)
            ch_full, ch_plen = full_ids_of(tok, pids, p["chosen"], dev, eos_id)
            rej_full, rej_plen = full_ids_of(tok, pids, p["rejected"], dev, eos_id)

            lp_ch = completion_logprob(model, ch_full, ch_plen)
            lp_rej = completion_logprob(model, rej_full, rej_plen)

            logits = args.beta * ((lp_ch - lp_ch_ref) - (lp_rej - lp_rej_ref))
            loss = -Fnn.logsigmoid(logits)
            (loss / args.grad_acc).backward()
            n_in_acc += 1

            if i % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                n_in_acc = 0
            if i % 10 == 0 or i == len(pairs):
                acc = (logits > 0).float().item()
                log.info("ep%d [%d/%d] loss=%.4f margin=%.3f chosen>rejected=%.0f",
                         ep + 1, i, len(pairs), loss.item(), logits.item(), acc)
        # epoch 末 flush 残余梯度：末批不足 grad_acc 时按实际条数重标定，保持有效学习率一致
        if n_in_acc > 0:
            scale = args.grad_acc / n_in_acc
            for pp in params:
                if pp.grad is not None:
                    pp.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    log.info("DPO 完成，adapter 保存到 %s", args.out_dir)


if __name__ == "__main__":
    main()
