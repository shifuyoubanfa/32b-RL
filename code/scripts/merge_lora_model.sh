#!/usr/bin/env bash
# Merge one final LoRA checkpoint into its full base model.
# Writes to <output>.partial first and only publishes <output> after validation.
set -euo pipefail

BASE_MODEL="$1"
ADAPTER="$2"
OUT="$3"
ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
SWIFT="$ZHJG_ENV/bin/swift"
TMP="${OUT}.partial"

[ -f "$BASE_MODEL/config.json" ] || { echo "[merge] missing base config: $BASE_MODEL"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "[merge] missing adapter_config.json: $ADAPTER"; exit 1; }
[ -x "$SWIFT" ] || { echo "[merge] swift not found: $SWIFT"; exit 1; }

if [ -f "$OUT/.done" ] && [ -f "$OUT/config.json" ]; then
  echo "[merge] already complete: $OUT"
  exit 0
fi
if [ -e "$OUT" ]; then
  echo "[merge] refusing incomplete/existing output: $OUT"
  echo "[merge] move it aside before retrying; no automatic deletion is performed."
  exit 1
fi
if [ -e "$TMP" ]; then
  STALE="${TMP}.interrupted-$(date +%Y%m%d-%H%M%S)"
  echo "[merge] preserving stale partial output -> $STALE"
  mv "$TMP" "$STALE"
fi

mkdir -p "$(dirname "$OUT")"
echo "[merge] base=$BASE_MODEL"
echo "[merge] adapter=$ADAPTER"
echo "[merge] output=$OUT"
echo "[merge] CPU merge avoids occupying training GPUs; this may take several minutes."

CUDA_VISIBLE_DEVICES="" "$SWIFT" export \
  --model "$BASE_MODEL" \
  --model_type "${V1_MODEL_TYPE:-qwen2}" \
  --template "${V1_TEMPLATE:-qwen2_5}" \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --device_map cpu \
  --torch_dtype bfloat16 \
  --safe_serialization true \
  --max_shard_size 5GB \
  --output_dir "$TMP"

[ -f "$TMP/config.json" ] || { echo "[merge] merged config missing: $TMP/config.json"; exit 1; }
find "$TMP" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . \
  || { echo "[merge] merged safetensors missing: $TMP"; exit 1; }
echo done > "$TMP/.done"
mv "$TMP" "$OUT"
echo "[merge] complete -> $OUT"
