#!/usr/bin/env bash
# 前台启动 V2 训练链：V1 -> 冷启动SFT(2σ/3σ) -> RFT(2σ/3σ) -> DPO(2σ/3σ)，跑到 DPO 结束。
# 长跑请放 tmux/screen 里：  tmux new -s v2  然后在里面 bash scripts/run_v2.sh
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_ROOT"

export ZHJG_WORK_DIR="${ZHJG_WORK_DIR:-/home/nvme01/zhjg}"
export ZHJG_LOG_DIR="${ZHJG_LOG_DIR:-$ZHJG_WORK_DIR/logs}"
export ZHJG_OUTPUT_DIR="${ZHJG_OUTPUT_DIR:-$ZHJG_WORK_DIR/output}"
export ZHJG_CKPT_DIR="${ZHJG_CKPT_DIR:-$ZHJG_WORK_DIR/ckpts}"
export ZHJG_MODEL_DIR="${ZHJG_MODEL_DIR:-$ZHJG_WORK_DIR/models}"
export ZHJG_ENV="${ZHJG_ENV:-/home/nvme02/conda/zhjg_rl}"
export ZHJG_CONSOLE_LOG_LEVEL="${ZHJG_CONSOLE_LOG_LEVEL:-WARNING}"

# V2 旋钮（都有代码默认，这里显式列出便于调；一般保持默认即可）
export V2_RFT_SELFSAMPLE_K="${V2_RFT_SELFSAMPLE_K:-32}"   # RFT 自采样候选数（用户定 32）
export V2_DPO_ROLLOUT_K="${V2_DPO_ROLLOUT_K:-16}"
export V2_EVAL_INTERMEDIATE="${V2_EVAL_INTERMEDIATE:-1}"  # 1=也评 SFT/RFT 中间态（看分阶段曲线），0=只评 8 个叶
export V2_MIN_FREE_GIB="${V2_MIN_FREE_GIB:-200}"          # 磁盘下限硬闸（镜像 derag_v4）
export KIMI_BUDGET_YUAN="${KIMI_BUDGET_YUAN:-0}"          # Kimi 预算围栏：0=只计量不设上限；要硬闸改成如 5000
# 早停"凑够就停"目标（凸显 RL：SFT 压低留 headroom / RFT 小验证 / DPO 主 RL 加量）；0=不限跑满
export V2_COLDSTART_TARGET="${V2_COLDSTART_TARGET:-700}"  # 冷启动 2σ 攒够即停（学透风格不封顶）
export V2_RFT_TARGET="${V2_RFT_TARGET:-200}"             # RFT 每线 2σ（小验证）
export V2_DPO_TARGET="${V2_DPO_TARGET:-900}"             # DPO 每线 2σ 偏好对（主 RL 加量）
# export ZHJG_V2_ROOT_BASE=/path/to/V1            # 二叉树根 base，默认=config.V1_DIR（V1 重开），一般不动

# 明文兜底 key（与其它 launcher 同款）：没 export 就用项目内置的，免每次手动设。
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-REDACTED-ROTATE-ME}"

echo "===== V2 训练链：V1 -> 冷启动SFT(2σ/3σ) -> RFT(2σ/3σ) -> DPO(2σ/3σ) ====="
echo "root_base = ${ZHJG_V2_ROOT_BASE:-<config.V1_DIR>}（V1 重开）"
echo "RFT自采K=$V2_RFT_SELFSAMPLE_K | DPO rollout K=$V2_DPO_ROLLOUT_K | 评测中间态=$V2_EVAL_INTERMEDIATE | 磁盘下限=${V2_MIN_FREE_GIB}G"
echo "Kimi围栏 KIMI_BUDGET_YUAN=$KIMI_BUDGET_YUAN（0=只计量）| 选样k=16 / 评测k=3 | 无损去重缓存默认开"
echo "Logs:      $ZHJG_LOG_DIR/v2/        state=$ZHJG_LOG_DIR/v2/state.json"
echo "Outputs:   $ZHJG_OUTPUT_DIR/v2/     Kimi计量=$ZHJG_OUTPUT_DIR/kimi_budget.json"
echo "监控(另开一个终端):  cd $CODE_ROOT && bash scripts/monitor_v2.sh"
echo ""

exec "$ZHJG_ENV/bin/python" -X utf8 run_v2.py
