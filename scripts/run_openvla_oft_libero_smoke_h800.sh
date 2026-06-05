#!/usr/bin/env bash
#SBATCH --job-name=oft-libero-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/oft-libero-smoke-%j.out
#SBATCH --error=logs/oft-libero-smoke-%j.err

set -euo pipefail

# Run from the openvla-oft repository root:
#   sbatch /path/to/robosuite/scripts/run_openvla_oft_libero_smoke_h800.sh
#
# Override examples:
#   TASK_SUITE=libero_object CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-object \
#     sbatch /path/to/robosuite/scripts/run_openvla_oft_libero_smoke_h800.sh

mkdir -p logs

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-7}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[env] host=$(hostname)"
echo "[env] date=$(date)"
echo "[env] cwd=$PWD"
echo "[env] cuda devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run] checkpoint=$CHECKPOINT task_suite=$TASK_SUITE trials=$TRIALS seed=$SEED"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$TRIALS" \
  --seed "$SEED"

