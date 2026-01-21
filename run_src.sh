#!/usr/bin/env bash
set -Eeuo pipefail

# ---- settings you may change
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-src/main.py}"
OUT_DIR="${OUT_DIR:-/work/outputs_src}"
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

# Base epsilon values
MNIST_EPS_1X=0.3
MNIST_EPS_2X=0.6
MNIST_EPS_3X=0.9
MNIST_ALPHA=0.01

CIFAR10_EPS_1X=0.03137254901960784   # 8/255
CIFAR10_EPS_2X=0.06274509803921569   # 16/255
CIFAR10_EPS_3X=0.09411764705882353   # 24/255
CIFAR10_ALPHA=0.00784313725490196    # 2/255

# ------------------------------------------------------------------
# MNIST (nat) - epsilon 1x, 2x, 3x
# ------------------------------------------------------------------
for mult in 1 2 3; do
  eval "eps=\$MNIST_EPS_${mult}X"
  run "mnist_nat_random_eps${mult}x" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --dataset mnist \
    --init random --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 \
    --epsilon "$eps" --alpha "$MNIST_ALPHA"

  run "mnist_nat_deepfool_eps${mult}x" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --dataset mnist \
    --init deepfool --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 --df_max_iter 50 \
    --epsilon "$eps" --alpha "$MNIST_ALPHA" \
    --df_overshoot 0.02 --init_sanity_plot
done

# ------------------------------------------------------------------
# MNIST (adv) - epsilon 1x, 2x, 3x
# ------------------------------------------------------------------
for mult in 1 2 3; do
  eval "eps=\$MNIST_EPS_${mult}X"
  run "mnist_adv_random_eps${mult}x" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --dataset mnist \
    --init random --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 \
    --epsilon "$eps" --alpha "$MNIST_ALPHA"

  run "mnist_adv_deepfool_eps${mult}x" \
    --model_src_dir model_src/mnist_challenge \
    --ckpt_dir model_src/mnist_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --dataset mnist \
    --init deepfool --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 --df_max_iter 50 \
    --epsilon "$eps" --alpha "$MNIST_ALPHA" \
    --df_overshoot 0.02 --init_sanity_plot
done

# ------------------------------------------------------------------
# CIFAR10 (nat) - epsilon 1x, 2x, 3x
# ------------------------------------------------------------------
for mult in 1 2 3; do
  eval "eps=\$CIFAR10_EPS_${mult}X"
  run "cifar10_nat_random_eps${mult}x" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --dataset cifar10 \
    --init random --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 \
    --epsilon "$eps" --alpha "$CIFAR10_ALPHA"

  run "cifar10_nat_deepfool_eps${mult}x" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/nat \
    --out_dir "$OUT_DIR" \
    --dataset cifar10 \
    --init deepfool --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 --df_max_iter 50 \
    --epsilon "$eps" --alpha "$CIFAR10_ALPHA" \
    --df_overshoot 0.02 --init_sanity_plot
done

# ------------------------------------------------------------------
# CIFAR10 (adv) - epsilon 1x, 2x, 3x
# ------------------------------------------------------------------
for mult in 1 2 3; do
  eval "eps=\$CIFAR10_EPS_${mult}X"
  run "cifar10_adv_random_eps${mult}x" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --dataset cifar10 \
    --init random --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 \
    --epsilon "$eps" --alpha "$CIFAR10_ALPHA"

  run "cifar10_adv_deepfool_eps${mult}x" \
    --model_src_dir model_src/cifar10_challenge \
    --ckpt_dir model_src/cifar10_challenge/models/adv \
    --out_dir "$OUT_DIR" \
    --dataset cifar10 \
    --init deepfool --seed 0 \
    --n_examples 5 \
    --num_restarts 20 --total_iter 100 --df_max_iter 50 \
    --epsilon "$eps" --alpha "$CIFAR10_ALPHA" \
    --df_overshoot 0.02 --init_sanity_plot
done

echo "ALL DONE."
