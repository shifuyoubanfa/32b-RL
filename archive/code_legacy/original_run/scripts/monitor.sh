#!/usr/bin/env bash
# 实时健康监控：每 10s 刷 当前阶段 / GPU 占用 / 最新训练指标 / 各阶段验收报告摘要。
# 用法：bash scripts/monitor.sh   （另开一个窗口，与 run_all.sh 并行看）
WORK="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
LOG_DIR="${ZHJG_LOG_DIR:-$WORK/logs}"
OUT_DIR="${ZHJG_OUTPUT_DIR:-$WORK/output}"
PLOG="$LOG_DIR/pipeline.log"

while true; do
  clear
  echo "================= $(date '+%F %T')  32B RL 健康监控 ================="
  echo "--- 流水线进度（pipeline.log 关键行）---"
  grep -E "阶段 \[|START|END |FAIL|跳过|PIPELINE" "$PLOG" 2>/dev/null | tail -6
  echo ""
  echo "--- 训练/采样最新指标（loss / reward / kl / it/s）---"
  grep -iE "loss|reward|'kl'|it/s|epoch" "$PLOG" 2>/dev/null | tail -4
  echo ""
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "  GPU%s 利用率%s%% 显存%s/%sMB\n",$1,$2,$3,$4}'
  echo ""
  echo "--- 各阶段验收报告（humanness / 准确率 / 照抄）---"
  for f in $(ls -t "$OUT_DIR"/*_report.md 2>/dev/null | head -4 | tac); do
    echo "[$(basename "$f")]"
    grep -E "humanness 均值|准确率|correct\+partial|verbatim_copy" "$f" 2>/dev/null | head -3 | sed 's/^/   /'
  done
  echo ""
  echo "(Ctrl+C 退出 ｜ 详细：tail -f $PLOG)"
  sleep 10
done
