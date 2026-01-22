#!/usr/bin/env bash
set -Eeuo pipefail

# ---- settings you may change
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-/work/outputs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

run() {
  local script="$1"; shift
  local name="$1"; shift
  local log="${LOG_DIR}/$(timestamp)_${name}.log"

  echo "=================================================================="
  echo "[RUN] ${name}"
  echo "[LOG] ${log}"
  echo "CMD: $PYTHON $script $*"
  echo "=================================================================="

  # stdout+stderr -> log (and also show on terminal)
  $PYTHON "$script" "$@" 2>&1 | tee "$log"
}

# Epsilon and alpha values
MNIST_EPS=0.3
MNIST_ALPHA=0.01

# Common parameters
N_EXAMPLES=5
NUM_RESTARTS=9
TOTAL_ITER=100
DF_MAX_ITER=50
DF_OVERSHOOT=0.02

# Experiment name
EXP_NAME=run_mnist_nat_and_adv

# ==================================================================
# src/main.py: random init (MNIST nat_and_adv)
# ==================================================================
run "src/main.py" "src_mnist_nat_and_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA"

# ==================================================================
# src/main.py: deepfool init (MNIST nat_and_adv)
# ==================================================================
run "src/main.py" "src_mnist_nat_and_adv_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot

# ==================================================================
# loss_curves.py: multi_deepfool init (MNIST nat_and_adv)
# ==================================================================
run "loss_curves.py" "lc_mnist_nat_and_adv_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot

echo "ALL DONE."
