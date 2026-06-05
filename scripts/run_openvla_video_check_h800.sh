#!/usr/bin/env bash
#SBATCH --job-name=openvla-video-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/openvla-video-check-%j.out
#SBATCH --error=logs/openvla-video-check-%j.err

set -euo pipefail

# Run from the repository root:
#   sbatch scripts/run_openvla_video_check_h800.sh
#
# Override defaults when submitting, for example:
#   VIDEO_DIR=videos/h800_openvla_check ENV_NAME=DepthAmbiguityEnv EPISODES=1 HORIZON=120 \
#     sbatch scripts/run_openvla_video_check_h800.sh

mkdir -p logs

ENV_NAME="${ENV_NAME:-DepthAmbiguityEnv}"
EPISODES="${EPISODES:-1}"
HORIZON="${HORIZON:-120}"
VIDEO_DIR="${VIDEO_DIR:-videos/h800_openvla_check}"
DEVICE="${DEVICE:-cuda:0}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
VIDEO_CAMERA="${VIDEO_CAMERA:-sideview}"
VIDEO_SIZE="${VIDEO_SIZE:-512}"
VIDEO_SKIP="${VIDEO_SKIP:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

echo "[env] host=$(hostname)"
echo "[env] date=$(date)"
echo "[env] cuda devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[env] MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "[run] env=$ENV_NAME episodes=$EPISODES horizon=$HORIZON video_dir=$VIDEO_DIR"

python -c "import imageio, imageio_ffmpeg; print('[deps] imageio/video backend ok')"

python run_l1_testing.py \
  --model openvla \
  --env "$ENV_NAME" \
  --episodes "$EPISODES" \
  --horizon "$HORIZON" \
  --device "$DEVICE" \
  --unnorm_key "$UNNORM_KEY" \
  --video_dir "$VIDEO_DIR" \
  --video_camera "$VIDEO_CAMERA" \
  --video_size "$VIDEO_SIZE" \
  --video_skip "$VIDEO_SKIP" \
  $EXTRA_ARGS

python tools/validate_rollout_videos.py "$VIDEO_DIR" \
  --report "$VIDEO_DIR/video_quality_report.csv" \
  --json_report "$VIDEO_DIR/video_quality_report.json"

echo "[done] videos and reports are under $VIDEO_DIR"
