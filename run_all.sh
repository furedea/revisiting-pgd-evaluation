#!/usr/bin/env bash
set -Eeuo pipefail

# ---- settings you may change
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-loss_curves.py}"
OUT_DIR="${OUT_DIR:-/work/outputs}"
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

# ------------------------------------------------------------------
# MNIST (nat)
# ------------------------------------------------------------------
run "mnist_nat_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 \
  --epsilon 0.3 --alpha 0.01

run "mnist_nat_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 --df_max_iter 1 \
  --epsilon 0.3 --alpha 0.01 \
  --df_overshoot 0.02 --init_sanity_plot

# ------------------------------------------------------------------
# MNIST (adv)
# ------------------------------------------------------------------
run "mnist_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 \
  --epsilon 0.3 --alpha 0.01

run "mnist_adv_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 --df_max_iter 1 \
  --epsilon 0.3 --alpha 0.01 \
  --df_overshoot 0.02 --init_sanity_plot

# ------------------------------------------------------------------
# CIFAR10 (nat)
# ------------------------------------------------------------------
run "cifar10_nat_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 \
  --epsilon 0.03137254901960784 --alpha 0.00784313725490196

run "cifar10_nat_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 --df_max_iter 1 \
  --epsilon 0.03137254901960784 --alpha 0.00784313725490196 \
  --df_overshoot 0.02 --init_sanity_plot

# ------------------------------------------------------------------
# CIFAR10 (adv)
# ------------------------------------------------------------------
run "cifar10_adv_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 \
  --epsilon 0.03137254901960784 --alpha 0.00784313725490196

run "cifar10_adv_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples 5 \
  --num_restarts 9 --total_iter 100 --df_max_iter 1 \
  --epsilon 0.03137254901960784 --alpha 0.00784313725490196 \
  --df_overshoot 0.02 --init_sanity_plot

echo "ALL DONE."

