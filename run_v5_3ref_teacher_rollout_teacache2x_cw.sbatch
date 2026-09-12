#!/usr/bin/env bash
#SBATCH --job-name=h3-v5-3ref-teacher
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2
#SBATCH --output=logs/v5_3ref_teacher_%A_%a.out
#SBATCH --error=logs/v5_3ref_teacher_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && "${SCRIPT_DIR}" == *spool* ]]; then
  DEFAULT_ROOT=${SLURM_SUBMIT_DIR}
else
  DEFAULT_ROOT=${SCRIPT_DIR}
fi
ROOT=${ROOT:-${DEFAULT_ROOT}}
RAVEN_ROOT=${RAVEN_ROOT:-${ROOT}/RAVEN}
MODEL_BASE=${MODEL_BASE:-/home/yitongl/code/models}
MODEL_ROOT=${MODEL_ROOT:-${MODEL_BASE}/MiniMax-H3}
H3_DIT_DCP=${H3_DIT_DCP:-${MODEL_BASE}/MiniMax-H3-DCP/Ref2VA/transformer}
TEXT_ENCODER_DCP=${TEXT_ENCODER_DCP:-${MODEL_BASE}/MiniMax-H3-DCP/Ref2VA/text_encoder}
PYTHON_BIN=${PYTHON_BIN:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/miniconda3/envs/sana_t27/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/miniconda3/envs/sana_t27/bin/torchrun}
FFMPEG_EXE=${FFMPEG_EXE:-/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/miniconda3/envs/sana_t27/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
CONFIG=${CONFIG:-${ROOT}/minimax_h3_ref2va_teacher_rollout_worker_50nfe.yaml}
INDEX=${INDEX:-${ROOT}/v5_3ref_teacher_rollout_index.json}
OUT_ROOT=${OUT_ROOT:-${ROOT}/artifacts/v5_3ref_teacher_rollout_teacache_2x}
CASES_PER_WORKER=${CASES_PER_WORKER:-4}
CASE_START=${CASE_START:-$(( (${SLURM_ARRAY_TASK_ID:-0}) * CASES_PER_WORKER ))}
CASE_COUNT=${CASE_COUNT:-${CASES_PER_WORKER}}
STEPS=${STEPS:-50}
TCACHE_COMPUTE_STEPS=${TCACHE_COMPUTE_STEPS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,49}
FORCE=${FORCE:-false}
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"

RUN_ROOT="${OUT_ROOT}/runs"
LATENT_ROOT="${OUT_ROOT}/latents"
VIDEO_ROOT="${OUT_ROOT}/videos"
EXP_NAME="v5-3ref-teacher-teacache2x-${CASE_START}-${CASE_COUNT}-${JOB_KEY}"
RUN_DIR="${RUN_ROOT}/minimax_h3_ref2va_rgb_depth/${EXP_NAME}"
TIMING_PATH="${RUN_DIR}/metrics/hot_timing_rank0.jsonl"
MASTER_PORT=$((37000 + (${SLURM_JOB_ID:-$$} % 9000) + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

mkdir -p "${ROOT}/logs" "${RUN_ROOT}" "${LATENT_ROOT}" "${VIDEO_ROOT}" \
  "${ROOT}/.cache/tmp" "${ROOT}/.cache/triton" "${ROOT}/.cache/huggingface" "${ROOT}/.cache/bin"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
test -d "${RAVEN_ROOT}"
test -s "${INDEX}"
test -s "${H3_DIT_DCP}/.metadata"
test -s "${TEXT_ENCODER_DCP}/.metadata"
test -d "${MODEL_ROOT}/Ref2VA/processor"
test -d "${MODEL_ROOT}/Ref2VA/tokenizer"
test -d "${MODEL_ROOT}/Ref2VA/text_encoder"
test -s "${MODEL_ROOT}/Ref2VA/video_vae/source/model.safetensors"
test -s "${MODEL_ROOT}/Ref2VA/audio_vae/model.safetensors"
test -x "${FFMPEG_EXE}"

ln -sf "${FFMPEG_EXE}" "${ROOT}/.cache/bin/ffmpeg"
export PATH="${ROOT}/.cache/bin:${PATH}"
export TMPDIR="${ROOT}/.cache/tmp"
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export HF_HOME="${ROOT}/.cache/huggingface"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export PYTHONPATH="${RAVEN_ROOT}:${PYTHONPATH:-}"

export H3_REF2VA_CONDITION_KV_CACHE=0
unset H3_REF2VA_CONDITION_KV_CACHE_REFRESH_STEPS
unset H3_REF2VA_CONDITION_KV_CACHE_REFRESH_COUNT
unset H3_REFCACHEBLEND_EXPORT_DIR
unset H3_REFCACHEBLEND_IMPORT_TEXT_DIR
unset H3_REFCACHEBLEND_IMPORT_ASSET_DIRS

export H3_REF2VA_HOT_TIMING=1
export H3_REF2VA_HOT_TIMING_PATH="${TIMING_PATH}"
export H3_REF2VA_TCACHE=1
export H3_REF2VA_TCACHE_COMPUTE_STEPS="${TCACHE_COMPUTE_STEPS}"

echo "V5_3REF_TEACHER_TCACHE2X root=${ROOT}"
echo "case_start=${CASE_START} case_count=${CASE_COUNT} steps=${STEPS} force=${FORCE}"
echo "index=${INDEX}"
echo "out_root=${OUT_ROOT}"
echo "tcache_compute_steps=${TCACHE_COMPUTE_STEPS}"
echo "condition_cache=${H3_REF2VA_CONDITION_KV_CACHE}"
echo "node=${SLURMD_NODENAME:-unknown} job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-0}"

cd "${RAVEN_ROOT}"
"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node 4 \
  --master-port "${MASTER_PORT}" \
  -m common.launch --config "${CONFIG}" \
  validation.rollout_index "${INDEX}" \
  validation.case_start "${CASE_START}" \
  validation.case_count "${CASE_COUNT}" \
  validation.save_latent true \
  validation.save_video true \
  validation.force "${FORCE}" \
  validation.latent_root "${LATENT_ROOT}" \
  validation.video_root "${VIDEO_ROOT}" \
  validation.processor_path "${MODEL_ROOT}/Ref2VA/processor" \
  validation.visual_anchor "${VISUAL_ANCHOR:-0.999}" \
  validation.audio_anchor "${AUDIO_ANCHOR:-1.0}" \
  logging.plugins [] \
  diffusion.sampling_timesteps.args.num_sampling_steps "${STEPS}" \
  diffusion.audio_sampling_timesteps.args.num_sampling_steps "${STEPS}" \
  data.args.num_frames 124 \
  data.args.height 768 \
  data.args.width 1344 \
  data.args.max_seqlen "${MAX_SEQLEN:-262144}" \
  data.args.max_seqlen_per_sample "${MAX_SEQLEN:-262144}" \
  data.args.tokenizer_path "${MODEL_ROOT}/Ref2VA/tokenizer" \
  models.backbone.placement.plugins.0.path "${H3_DIT_DCP}" \
  models.backbone.placement.weight_dtype bfloat16 \
  models.backbone.placement.fsdp.param_dtype bfloat16 \
  models.backbone.placement.cast_frozen true \
  models.text_encoder.args.pretrained_model_name_or_path "${MODEL_ROOT}/Ref2VA/text_encoder" \
  models.text_encoder.placement.plugins.0.path "${TEXT_ENCODER_DCP}" \
  models.text_encoder.placement.cast_frozen true \
  models.video_vae.weight.path "${MODEL_ROOT}/Ref2VA/video_vae/source/model.safetensors" \
  models.audio_vae.weight.path "${MODEL_ROOT}/Ref2VA/audio_vae/model.safetensors" \
  persistence.output_dir "${RUN_ROOT}" \
  persistence.exp_name "${EXP_NAME}"

if [[ -s "${TIMING_PATH}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/summarize_ref2va_hot_timing.py" "${TIMING_PATH}" "${RUN_DIR}/metrics/hot_timing_summary.json" >/dev/null
  echo "timing_summary=${RUN_DIR}/metrics/hot_timing_summary.json"
fi

echo "latent_root=${LATENT_ROOT}"
echo "video_root=${VIDEO_ROOT}"
echo "done case_start=${CASE_START} case_count=${CASE_COUNT}"
