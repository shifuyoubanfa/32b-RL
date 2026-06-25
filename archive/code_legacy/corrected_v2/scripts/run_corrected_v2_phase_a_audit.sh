#!/usr/bin/env bash
# corrected-v2 Phase A only: no training, no checkpoint writes.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export PMI_ENABLED=0

PAIR_AUDIT_N="${PAIR_AUDIT_N:-150}"
PAIR_AUDIT_WORKERS="${PAIR_AUDIT_WORKERS:-1}"

echo "===== corrected-v2 Phase A audit ====="
echo "1) offline local reward vs Kimi judge alignment"
echo "2) Kimi audit of old DPO pairs, sample=${PAIR_AUDIT_N}, workers=${PAIR_AUDIT_WORKERS}"
echo "Outputs:"
echo "  \$ZHJG_WORK_DIR/output/90_corrected_v2_reward_alignment_audit.md"
echo "  \$ZHJG_WORK_DIR/output/91_corrected_v2_dpo_pair_kimi_audit.md"
echo

"$ZHJG_ENV/bin/python" -X utf8 pipeline/step90_audit_reward_alignment.py

: "${DASHSCOPE_API_KEY:?Please export DASHSCOPE_API_KEY before running step91 Kimi pair audit.}"
DPO_PAIR_AUDIT_WORKERS="$PAIR_AUDIT_WORKERS" "$ZHJG_ENV/bin/python" -X utf8 \
  pipeline/step91_audit_dpo_pairs_kimi.py --limit "$PAIR_AUDIT_N"

echo
echo "===== Phase A complete ====="
echo "Read:"
echo "  $ZHJG_WORK_DIR/output/90_corrected_v2_reward_alignment_audit.md"
echo "  $ZHJG_WORK_DIR/output/91_corrected_v2_dpo_pair_kimi_audit.md"
