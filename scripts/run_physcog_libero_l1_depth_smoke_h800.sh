#!/usr/bin/env bash
#SBATCH --job-name=physcog-libero-l1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/physcog-libero-l1-%j.out
#SBATCH --error=logs/physcog-libero-l1-%j.err

set -euo pipefail

# Run from the openvla-oft repository root after you create:
#   experiments/robot/libero/run_physcog_libero_l1_eval.py
#
# The script intentionally mirrors native OpenVLA-OFT LIBERO evaluation so that
# PhysCogSafe L1 only changes the task suite and safety oracle layer.

mkdir -p logs

TASK_SUITE="${TASK_SUITE:-physcog_libero_l1_depth}"
CHECKPOINT="${CHECKPOINT:-moojink/openvla-7b-oft-finetuned-libero-spatial}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-7}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f experiments/robot/libero/run_physcog_libero_l1_eval.py ]]; then
  echo "[error] Missing experiments/robot/libero/run_physcog_libero_l1_eval.py"
  echo "Create it by copying run_libero_eval.py and adding PhysCogSafe safety oracles."
  exit 2
fi

echo "[env] host=$(hostname)"
echo "[env] date=$(date)"
echo "[env] cwd=$PWD"
echo "[env] cuda devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run] checkpoint=$CHECKPOINT task_suite=$TASK_SUITE trials=$TRIALS seed=$SEED"

python experiments/robot/libero/run_physcog_libero_l1_eval.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$TRIALS" \
  --seed "$SEED"

