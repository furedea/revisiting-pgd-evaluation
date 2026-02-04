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

  $PYTHON "$script" "$@" 2>&1 | tee "$log"
}

# Experiment name
EXP_NAME=timing_ex10

# Common indices files
MNIST_INDICES_FILE="docs/common_correct_indices_mnist_n10.json"
CIFAR10_INDICES_FILE="docs/common_correct_indices_cifar10_n10.json"

# ==================================================================
# Generate common correct indices (if not exists)
# ==================================================================
if [ ! -f "$MNIST_INDICES_FILE" ]; then
  echo "Generating MNIST indices..."
  $PYTHON find_common_correct_samples.py --dataset mnist --samples_per_class 1 --seed 0
fi

if [ ! -f "$CIFAR10_INDICES_FILE" ]; then
  echo "Generating CIFAR10 indices..."
  $PYTHON find_common_correct_samples.py --dataset cifar10 --samples_per_class 1 --seed 0
fi

# ==================================================================
# MNIST timing measurements
# ==================================================================
for MODEL in nat adv nat_and_adv weak_adv; do
  # n=1 comparison: random vs deepfool
  run "measure_timing.py" "timing_mnist_${MODEL}_random_n1" \
    --dataset mnist --model "$MODEL" --init random --num_restarts 1 \
    --common_indices_file "$MNIST_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  run "measure_timing.py" "timing_mnist_${MODEL}_deepfool_n1" \
    --dataset mnist --model "$MODEL" --init deepfool --num_restarts 1 \
    --common_indices_file "$MNIST_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  # n=9 comparison: random vs multi_deepfool
  run "measure_timing.py" "timing_mnist_${MODEL}_random_n9" \
    --dataset mnist --model "$MODEL" --init random --num_restarts 9 \
    --common_indices_file "$MNIST_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  run "measure_timing.py" "timing_mnist_${MODEL}_multi_deepfool_n9" \
    --dataset mnist --model "$MODEL" --init multi_deepfool --num_restarts 9 \
    --common_indices_file "$MNIST_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"
done

# ==================================================================
# CIFAR10 timing measurements
# ==================================================================
for MODEL in nat adv nat_and_adv weak_adv; do
  # n=1 comparison: random vs deepfool
  run "measure_timing.py" "timing_cifar10_${MODEL}_random_n1" \
    --dataset cifar10 --model "$MODEL" --init random --num_restarts 1 \
    --common_indices_file "$CIFAR10_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  run "measure_timing.py" "timing_cifar10_${MODEL}_deepfool_n1" \
    --dataset cifar10 --model "$MODEL" --init deepfool --num_restarts 1 \
    --common_indices_file "$CIFAR10_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  # n=9 comparison: random vs multi_deepfool
  run "measure_timing.py" "timing_cifar10_${MODEL}_random_n9" \
    --dataset cifar10 --model "$MODEL" --init random --num_restarts 9 \
    --common_indices_file "$CIFAR10_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"

  run "measure_timing.py" "timing_cifar10_${MODEL}_multi_deepfool_n9" \
    --dataset cifar10 --model "$MODEL" --init multi_deepfool --num_restarts 9 \
    --common_indices_file "$CIFAR10_INDICES_FILE" \
    --out_dir "$OUT_DIR" --exp_name "$EXP_NAME"
done

# ==================================================================
# Analyze results
# ==================================================================
echo "=================================================================="
echo "[ANALYZE] Running analyze_timing.py"
echo "=================================================================="
run "analyze_timing.py" "analyze_timing" \
  --input_dir "${OUT_DIR}/timing/${EXP_NAME}" --out_dir "$OUT_DIR"

echo "ALL DONE."
