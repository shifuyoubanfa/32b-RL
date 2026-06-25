"""Step 10（RL 阶段3）：手写 GRPO 在线强化（基于 transformers+peft，不依赖 trl）。

核心 = 你定的"非对称 KL"：
- 每条 query 采样 K 个回答，本地 reward 打分，组内归一化得 advantage；
- 策略梯度推动高 advantage 的回答；
- 同时对 ref(SFT，disable_adapter) 做 KL 锚定，但 **think 段 KL 系数≈0（放开变自然）、
  answer 段 KL 系数大（按住，保住准确率）**。

显存：单卡 96G，batch=1 逐序列前向 + 梯度累积，不开 grad-ckpt（避免与 generate 的 use_cache 冲突）。
--smoke：2 步 × 2 prompt × K=4，几分钟验证不崩。

注意：这是实验性最强、最该先 --smoke 的一步。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 必须在 import torch 前设：用 expandable_segments 消除显存碎片。
# GRPO 每步 generate(K路) + 几十次大前向/backward 交替，极易把显存切碎；
# 实测 OOM 时仅差 198MiB 却有 10.5GB"保留但未分配"碎片——这条直接回收碎片。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RL_POOL, GRPO_LORA_DIR, STUDENT_LOCAL_DIR, STUDENT_LORA_DIR, RFT_LORA_DIR, DPO_LORA_DIR,
    GRPO_LR, GRPO_K, GRPO_STEPS, GRPO_PROMPTS_PER_STEP, GRPO_KL_THINK, GRPO_KL_ANSWER,
    GRPO_TEMPERATURE, GRPO_TOP_P, GRPO_MAX_FORWARD_LEN, RL_GEN_MAX_NEW_TOKENS, MAX_LEN,
    REWARD_TAU_ACC, REWARD_W_HUMAN, REWARD_W_ACC, THINK_MIN_CHARS, THINK_MAX_CHARS, SYSTEM_PROMPT,
)
from pipeline.logger import get_logger
from pipeline import reward as R

log = get_logger("step10_grpo")


def pick_init_lora(explicit):
    if explicit:
        return explicit
    for d in (DPO_LORA_DIR, RFT_LORA_DIR, STUDENT_LORA_DIR):
        if Path(d, "adapter_config.json").exists():
            return d
    return STUDENT_LORA_DIR


def prompt_ids_of(tok, user_prompt, device):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return ids.to(device)


def comp_token_logprobs(model, full_ids, prompt_len):
    """completion 段逐 token logprob，返回 [n_comp]（带/不带梯度由调用处的上下文决定）。
    省显存关键：只对 completion 段位置做 log_softmax(全词表 float)，不对整条序列做——
    长 prompt(税务可达6000+token)整条 float 会吃 2GB+ 显存且我们根本用不到 prompt 段的 logp。"""
    out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
    # 位置 prompt_len-1 .. T-2 预测 token prompt_len .. T-1（即 completion 各 token）
    comp_logits = out.logits[:, prompt_len - 1:-1, :]                     # [1, n_comp, vocab]
    comp_targets = full_ids[:, prompt_len:]                              # [1, n_comp]
    logp = torch.log_softmax(comp_logits.float(), dim=-1)                # 只 float completion 段
    return torch.gather(logp, 2, comp_targets.unsqueeze(-1)).squeeze(-1).squeeze(0)   # [n_comp]


def kl_weights(tok, comp_ids, n_comp, device):
    """think 段权重 GRPO_KL_THINK、answer 段权重 GRPO_KL_ANSWER。
    按 </think> 的【token-id 子序列】定位边界，避免 decode→retokenize 的 BPE 错位。
    找不到则全按 think(放开)处理(与原行为一致)。"""
    end_ids = tok("</think>", add_special_tokens=False)["input_ids"]
    n_think = n_comp
    if end_ids:
        L = len(end_ids)
        for i in range(0, n_comp - L + 1):
            if list(comp_ids[i:i + L]) == end_ids:
                n_think = i + L            # think 段含 </think> 本身
                break
    n_think = max(0, min(n_think, n_comp))
    w = [GRPO_KL_THINK] * n_think + [GRPO_KL_ANSWER] * (n_comp - n_think)
    return torch.tensor(w[:n_comp], dtype=torch.float32, device=device)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_lora", default=None, help="GRPO 起点 adapter（默认优先 DPO>RFT>SFT）")
    parser.add_argument("--ref_lora", default=None,
                        help="KL 参考 adapter（默认=起点策略）。answer 段 KL 锚到税务策略(CS_RFT/DPO)，不是裸 base")
    parser.add_argument("--pool", default=RL_POOL, help="query 池（默认 SFT_TRAIN；扩 query 时指向扩充池）")
    parser.add_argument("--out_dir", default=GRPO_LORA_DIR)
    parser.add_argument("--lr", type=float, default=GRPO_LR)
    parser.add_argument("--k", type=int, default=GRPO_K)
    parser.add_argument("--steps", type=int, default=GRPO_STEPS)
    parser.add_argument("--prompts_per_step", type=int, default=GRPO_PROMPTS_PER_STEP)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.steps, args.prompts_per_step, args.k, args.save_every = 2, 2, 4, 2

    init_lora = pick_init_lora(args.init_lora)
    if not Path(init_lora, "adapter_config.json").exists():
        log.error("起点 adapter 不存在: %s（GRPO 需要 DPO/RFT 产出，请先跑前置阶段）", init_lora)
        sys.exit(1)
    ref_lora = args.ref_lora or init_lora     # πref 默认锚到起点(税务)策略
    if not Path(ref_lora, "adapter_config.json").exists():
        log.error("参考 adapter 不存在: %s", ref_lora)
        sys.exit(1)
    log.info("GRPO 起点 adapter: %s | πref(KL锚): %s | steps=%d prompts/step=%d K=%d lr=%g KL(think=%g,answer=%g)",
             init_lora, ref_lora, args.steps, args.prompts_per_step, args.k, args.lr, GRPO_KL_THINK, GRPO_KL_ANSWER)

    with open(args.pool, "r", encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    if not pool:
        log.error("query 池为空：%s", args.pool)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(STUDENT_LOCAL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # attn_implementation="sdpa"：省显存注意力，避免 eager 把 T×T 注意力矩阵整块 materialize。
    # 长 prompt(T~3500)时 eager 的 T² 矩阵可达 ~47GB，是这次 OOM 的主因；SDPA 把它降到 O(T)。
    # 若环境不支持 sdpa(极少)则回退 eager 重试，避免加载即失败。
    try:
        base = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="sdpa",
        )
    except Exception as e:
        log.warning("sdpa 注意力加载失败(%r)，回退默认实现", e)
        base = AutoModelForCausalLM.from_pretrained(
            STUDENT_LOCAL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    # 可训策略="default"；冻结参考="ref"(独立副本，起点时与策略相同，训练中不变)。
    # 共享同一份 base，只多挂一个小 LoRA adapter，显存几乎无增。
    model = PeftModel.from_pretrained(base, init_lora, is_trainable=True)
    model.load_adapter(ref_lora, adapter_name="ref")
    model.set_adapter("default")
    model.eval()                       # 关 dropout：让采样分布与 logprob 前向一致；LoRA 仍可训
    dev = model.device
    eos_id = tok.eos_token_id

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pool_idx = 0
    for step in range(1, args.steps + 1):
        # 取本步的 prompts
        batch = []
        for _ in range(args.prompts_per_step):
            batch.append(pool[pool_idx % len(pool)])
            pool_idx += 1

        seqs = []   # 每条：(full_ids, prompt_len, comp_ids, advantage)
        step_rewards = []
        for s in batch:
            user_prompt = s["messages"][1]["content"]
            gold = (s.get("answer") or "").strip()
            pids = prompt_ids_of(tok, user_prompt, dev)
            attn = torch.ones_like(pids)
            with torch.no_grad():
                gen = model.generate(
                    pids, attention_mask=attn, do_sample=True,
                    temperature=GRPO_TEMPERATURE, top_p=GRPO_TOP_P,
                    num_return_sequences=args.k, max_new_tokens=RL_GEN_MAX_NEW_TOKENS,
                    pad_token_id=tok.pad_token_id, eos_token_id=eos_id,
                )
            plen = pids.shape[1]
            rewards, fulls, compids = [], [], []
            for k in range(gen.shape[0]):
                full = gen[k:k + 1, :]                     # [1, T]
                comp_ids = gen[k, plen:].tolist()
                text = tok.decode(comp_ids, skip_special_tokens=True)
                sc = R.score_rollout(text, user_prompt, gold,
                                     tau_acc=REWARD_TAU_ACC, w_human=REWARD_W_HUMAN, w_acc=REWARD_W_ACC,
                                     think_min=THINK_MIN_CHARS, think_max=THINK_MAX_CHARS)
                rewards.append(sc["reward"])
                fulls.append(full)
                compids.append(comp_ids)
            step_rewards.extend(rewards)
            # 组内归一化 advantage
            rt = torch.tensor(rewards, dtype=torch.float32)
            mean, std = rt.mean().item(), rt.std().item()
            if std < 1e-4:
                continue                                   # 组内无方差，无学习信号，跳过
            for k in range(len(rewards)):
                adv = (rewards[k] - mean) / (std + 1e-4)
                seqs.append((fulls[k], plen, compids[k], adv))

        if not seqs:
            log.info("step %d：所有组组内零方差，跳过", step)
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()        # 生成阶段结束、进反向前清一次碎片，配合 expandable_segments 防 OOM
        opt.zero_grad()
        total_pg = total_kl = 0.0
        for full, plen, comp_ids, adv in seqs:
            n_comp = full.shape[1] - plen
            if n_comp <= 0:
                continue
            # 训练前向长度上限：税务 prompt 最长可达 6000+，整条带梯度前向会 OOM。
            # 超限则从左侧截断 prompt(保留最近 prompt 上下文 + 完整 completion，n_comp 不变)，
            # policy/ref 用同一截断输入→KL 一致；这是显存的确定性兜底。
            if full.shape[1] > GRPO_MAX_FORWARD_LEN:
                cut = full.shape[1] - GRPO_MAX_FORWARD_LEN
                full = full[:, cut:]
                plen = plen - cut                                         # completion 段长度不变
            # 先算冻结参考(no_grad，算完立即释放)，再算带梯度策略——避免两者激活同时驻留导致 OOM
            with torch.no_grad():
                model.set_adapter("ref")                                  # 切到冻结的税务参考策略
                ref_logp = comp_token_logprobs(model, full, plen)         # ref，无梯度
                model.set_adapter("default")                              # 切回可训策略
            comp_logp = comp_token_logprobs(model, full, plen)            # 带梯度（"default"策略）
            n = comp_logp.shape[0]
            klw = kl_weights(tok, comp_ids, n, dev)[:n]
            ratio = (ref_logp - comp_logp).clamp(-10, 10)
            kl_tok = torch.exp(ratio) - ratio - 1.0                       # k3 估计，≥0
            kl_term = (klw * kl_tok).sum() / max(1, n)
            pg = -(adv * comp_logp.mean())
            loss = (pg + kl_term) / len(seqs)
            loss.backward()
            total_pg += float(pg.detach())
            total_kl += float(kl_term.detach())

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        avg_r = sum(step_rewards) / max(1, len(step_rewards))
        log.info("step %d/%d | seqs=%d avg_reward=%.3f pg=%.4f kl=%.4f",
                 step, args.steps, len(seqs), avg_r, total_pg / len(seqs), total_kl / len(seqs))

        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(args.out_dir)
            tok.save_pretrained(args.out_dir)
            log.info("已存 checkpoint -> %s (step %d)", args.out_dir, step)

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    log.info("GRPO 完成，adapter 保存到 %s", args.out_dir)


if __name__ == "__main__":
    main()
