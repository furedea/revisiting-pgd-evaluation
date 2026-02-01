#!/usr/bin/env bash
set -Eeuo pipefail

# ---- settings you may change
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-/work/outputs_ex5}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
INDICES_DIR="${INDICES_DIR:-docs}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$INDICES_DIR"

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
CIFAR10_EPS=0.03137254901960784   # 8/255
CIFAR10_ALPHA=0.00784313725490196 # 2/255

# Common parameters
N_EXAMPLES=5
TOTAL_ITER=100
DF_MAX_ITER=50
DF_OVERSHOOT=0.02

# Restarts per init type
NUM_RESTARTS_CLEAN=1
NUM_RESTARTS_RANDOM=20
NUM_RESTARTS_DEEPFOOL=1
NUM_RESTARTS_MULTI_DEEPFOOL=9

# Experiment name
EXP_NAME=run_all_ex5

# Common indices files
MNIST_INDICES="${INDICES_DIR}/common_correct_indices_mnist.json"
CIFAR_INDICES="${INDICES_DIR}/common_correct_indices_cifar10.json"

# ==================================================================
# Step 0: Generate common correct indices for all models
# ==================================================================
echo "=================================================================="
echo "[STEP 0] Generating common correct indices..."
echo "=================================================================="

run "find_common_correct_samples.py" "find_common_mnist" \
  --dataset mnist \
  --out_dir "$INDICES_DIR"

run "find_common_correct_samples.py" "find_common_cifar10" \
  --dataset cifar10 \
  --out_dir "$INDICES_DIR"

# ==================================================================
# src/main.py: random init
# ==================================================================
run "src/main.py" "src_mnist_nat_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_nat_and_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_weak_adv_random" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_cifar10_nat_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_adv_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_nat_and_adv_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_weak_adv_random" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init random --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_RANDOM" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

# ==================================================================
# src/main.py: deepfool init
# ==================================================================
run "src/main.py" "src_mnist_nat_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_adv_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_nat_and_adv_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_weak_adv_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_cifar10_nat_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_adv_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_nat_and_adv_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_weak_adv_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

# ==================================================================
# loss_curves.py: multi_deepfool init
# ==================================================================
run "loss_curves.py" "lc_mnist_nat_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "loss_curves.py" "lc_mnist_adv_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "loss_curves.py" "lc_mnist_nat_and_adv_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "loss_curves.py" "lc_mnist_weak_adv_multi_deepfool" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$MNIST_INDICES"

run "loss_curves.py" "lc_cifar10_nat_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "loss_curves.py" "lc_cifar10_adv_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "loss_curves.py" "lc_cifar10_nat_and_adv_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

run "loss_curves.py" "lc_cifar10_weak_adv_multi_deepfool" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init multi_deepfool --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_MULTI_DEEPFOOL" --total_iter "$TOTAL_ITER" --df_max_iter "$DF_MAX_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --df_overshoot "$DF_OVERSHOOT"  \
  --common_indices_file "$CIFAR_INDICES"

# ==================================================================
# src/main.py: clean init
# ==================================================================
run "src/main.py" "src_mnist_nat_clean" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_adv_clean" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_nat_and_adv_clean" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_mnist_weak_adv_clean" \
  --model_src_dir model_src/mnist_challenge \
  --ckpt_dir model_src/mnist_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset mnist \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$MNIST_EPS" --alpha "$MNIST_ALPHA" \
  --common_indices_file "$MNIST_INDICES"

run "src/main.py" "src_cifar10_nat_clean" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_adv_clean" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_nat_and_adv_clean" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/nat_and_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

run "src/main.py" "src_cifar10_weak_adv_clean" \
  --model_src_dir model_src/cifar10_challenge \
  --ckpt_dir model_src/cifar10_challenge/models/weak_adv \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --dataset cifar10 \
  --init clean --seed 0 \
  --n_examples "$N_EXAMPLES" \
  --num_restarts "$NUM_RESTARTS_CLEAN" --total_iter "$TOTAL_ITER" \
  --epsilon "$CIFAR10_EPS" --alpha "$CIFAR10_ALPHA" \
  --common_indices_file "$CIFAR_INDICES"

echo "ALL DONE."
