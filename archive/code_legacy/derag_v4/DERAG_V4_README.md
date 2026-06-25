# derag_v4: mandatory DPO -> GRPO chain

This is the 2026-06-12 implementation of `101_derag_v4_final_blueprint.md`.
It starts from the merged RFT full model and keeps all artifacts isolated by
`run_id`.

Run in the main terminal:

```bash
cd /mnt/pfs/zhjg/code
bash scripts/run_derag_v4.sh
```

Monitor in another terminal:

```bash
cd /mnt/pfs/zhjg/code
DERAG_V4_RUN_ID=<run_id printed by main terminal> bash scripts/monitor_derag_v4.sh
```

If `DERAG_V4_RUN_ID` is omitted, the monitor picks the newest derag_v4 run.

Default chain:

```text
RFT merged base
  -> residual-trace Kimi rewrite
  -> v4 deterministic replay + binary-vote judge anchor calibration
  -> J-trace-bin k2 + J-fact-bin k1 -> arbiter majority -> one repair
  -> blind spot-check sheet + phrase gate
  -> explicit L0 deterministic fallback if Kimi binary calibration is unfit
  -> Stage1 SFT LoRA and merged S1 model
  -> G1-1 pool density probe with K=16,32,64,128
  -> deterministic DPO pair build
  -> DPO variants a/b/c until gate passes
  -> GRPO variants a/b/c until gate passes
  -> deterministic final report
```

Output layout:

```text
/home/nvme01/zhjg/output/derag_v4/<run_id>/
/home/nvme01/zhjg/logs/derag_v4/<run_id>/
/home/nvme01/zhjg/ckpts/derag_v4/<run_id>/
/home/nvme01/zhjg/models/derag_v4/<run_id>/
```

Important knobs:

```bash
export DERAG_V4_G11_KS=16,32,64,128
export DERAG_V4_ROLLOUT_CHUNK_K=16
export DERAG_V4_MIN_REWRITES=400
export DERAG_V4_LEGACY_GATE_ROWS=/home/nvme01/zhjg/output/derag_v4/20260612_022656/125_gate_rewrites.rows.jsonl
export DERAG_V4_STAGE1_REUSE_DIR=/home/nvme01/zhjg/output/derag_v4/20260612_022656
# Set DERAG_V4_REUSE_STAGE1=0 only when a completely fresh RFT rollout/rewrite is required.
export DERAG_V4_JUDGE_WORKERS=3
export DERAG_V4_MIN_DPO_PAIRS=160
export DERAG_V4_MAX_ANSWER_DROP=0.02
```

Stage1 is intentionally auditable. Local rules only hard-reject unambiguous
failures. Necessary tax facts are masked before copy scoring. Kimi no longer
emits continuous gate scores: it casts evidence-backed binary trace/fact votes,
with grey-zone arbitration and one targeted repair. If binary anchor
calibration fails, the run records `DEGRADED-GO`, uses the validated L0 gate,
generates a larger blind spot-check sheet, and continues the RL chain.
