#!/usr/bin/env bash
set -Eeuo pipefail

# ---- settings you may change
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-loss_curves.py}"
OUT_DIR="${OUT_DIR:-/work/outputs_df_iter}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

run() {
  local name="$1"; shift
  local log="${LOG_DIR}/$(timestamp)_${name}.log"

  echo "=================================================================="
  echo "[RUN] ${name}"
  echo "[LOG] ${log}"
  echo "CMD: $PYTHON $SCRIPT $*"
  echo "=================================================================="

  # stdout+stderr -> log (and also show on terminal)
  $PYTHON "$SCRIPT" "$@" 2>&1 | tee "$log"
}

# df_max_iter values to test
DF_ITERS=(1 5 10 15 20 25 30 50)

# Epsilon and alpha values
MNIST_EPS=0.3
MNIST_ALPHA=0.01
CIFAR10_EPS=0.03137254901960784   # 8/255
CIFAR10_ALPHA=0.00784313725490196 # 2/255

# Common parameters
N_EXAMPLES=5
NUM_RESTARTS=9
TOTAL_ITER=100
DF_OVERSHOOT=0.02

# Experiment name
EXP_NAME=run_mdf_iter

# ------------------------------------------------------------------
# MNIST (nat) - random baseline
# ------------------------------------------------------------------
run "mnist_nat_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA"

# ------------------------------------------------------------------
# MNIST (nat) - multi_deepfool with varying df_max_iter
# ------------------------------------------------------------------
for df_iter in "${DF_ITERS[@]}"; do
  run "mnist_nat_multi_deepfool_dfiter${df_iter}" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --exp_name "$EXP_NAME" \
    --dataset mnist \
    --init multi_deepfool --seed 0 \
    --n_examples "$N_EXAMPLES" \
    --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$df_iter" \
    --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
    --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot
done

# ------------------------------------------------------------------
# MNIST (adv) - random baseline
# ------------------------------------------------------------------
run "mnist_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA"

# ------------------------------------------------------------------
# MNIST (adv) - multi_deepfool with varying df_max_iter
# ------------------------------------------------------------------
for df_iter in "${DF_ITERS[@]}"; do
  run "mnist_adv_multi_deepfool_dfiter${df_iter}" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --exp_name "$EXP_NAME" \
    --dataset mnist \
    --init multi_deepfool --seed 0 \
    --n_examples "$N_EXAMPLES" \
    --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$df_iter" \
    --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
    --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot
done

# ------------------------------------------------------------------
# CIFAR10 (nat) - random baseline
# ------------------------------------------------------------------
run "cifar10_nat_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA"

# ------------------------------------------------------------------
# CIFAR10 (nat) - multi_deepfool with varying df_max_iter
# ------------------------------------------------------------------
for df_iter in "${DF_ITERS[@]}"; do
  run "cifar10_nat_multi_deepfool_dfiter${df_iter}" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --exp_name "$EXP_NAME" \
    --dataset cifar10 \
    --init multi_deepfool --seed 0 \
    --n_examples "$N_EXAMPLES" \
    --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$df_iter" \
    --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
    --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot
done

# ------------------------------------------------------------------
# CIFAR10 (adv) - random baseline
# ------------------------------------------------------------------
run "cifar10_adv_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA"

# ------------------------------------------------------------------
# CIFAR10 (adv) - multi_deepfool with varying df_max_iter
# ------------------------------------------------------------------
for df_iter in "${DF_ITERS[@]}"; do
  run "cifar10_adv_multi_deepfool_dfiter${df_iter}" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --exp_name "$EXP_NAME" \
    --dataset cifar10 \
    --init multi_deepfool --seed 0 \
    --n_examples "$N_EXAMPLES" \
    --num_restarts "$NUM_RESTARTS" --total_iter "$TOTAL_ITER" --df_max_iter "$df_iter" \
    --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
    --df_overshoot "$DF_OVERSHOOT" --init_sanity_plot
done

echo "ALL DONE."
