#!/usr/bin/env bash
#SBATCH --job-name=h3-ref2va-lora-distill
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/ref2va_lora_distill_%j.out
#SBATCH --error=logs/ref2va_lora_distill_%j.err

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
CONFIG=${CONFIG:-${ROOT}/minimax_h3_ref2va_decoupled_distill.yaml}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${ROOT}/ref2va_decoupled_distill_train_manifest.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/artifacts/ref2va_decoupled_distill_lora/runs}
EXP_NAME=${EXP_NAME:-lora-bsz16-sp4-ref2va-decoupled}
FINAL32=${FINAL32:-${ROOT}/artifacts/covebench_validation_final_32.json}
COVE_INDEX=${COVE_INDEX:-${ROOT}/covebench_teacher_rollout_index.json}
REF_DECODE_CACHE=${REF_DECODE_CACHE:-${ROOT}/artifacts/ref_decode_cache/covebench_final32_1344x768_124f}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=$((39000 + (${SLURM_JOB_ID:-$$} % 20000)))
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)}

mkdir -p "${ROOT}/logs" "${OUTPUT_ROOT}" \
  "${ROOT}/.cache/tmp" "${ROOT}/.cache/triton" "${ROOT}/.cache/huggingface" "${ROOT}/.cache/bin"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
test -s "${TRAIN_MANIFEST}"
test -d "${RAVEN_ROOT}"
test -s "${FINAL32}"
test -s "${COVE_INDEX}"
test -s "${H3_DIT_DCP}/.metadata"
test -s "${TEXT_ENCODER_DCP}/.metadata"
test -d "${MODEL_ROOT}/Ref2VA/processor"
test -d "${MODEL_ROOT}/Ref2VA/tokenizer"
test -d "${MODEL_ROOT}/Ref2VA/text_encoder"
test -s "${MODEL_ROOT}/Ref2VA/video_vae/source/model.safetensors"
test -s "${MODEL_ROOT}/Ref2VA/audio_vae/model.safetensors"
test -x "${FFMPEG_EXE}"
"${PYTHON_BIN}" - <<'PY'
import peft
import torch
print("peft ok", getattr(peft, "__version__", "unknown"))
print("torch", torch.__version__)
PY

ln -sf "${FFMPEG_EXE}" "${ROOT}/.cache/bin/ffmpeg"
export PATH="${ROOT}/.cache/bin:${PATH}"
export TMPDIR="${ROOT}/.cache/tmp"
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export HF_HOME="${ROOT}/.cache/huggingface"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAVEN_TORCH_NUM_THREADS=${RAVEN_TORCH_NUM_THREADS:-1}
export RAVEN_TORCH_NUM_INTEROP_THREADS=${RAVEN_TORCH_NUM_INTEROP_THREADS:-1}
export RAVEN_DISABLE_AUTOGRAD_MULTITHREADING=${RAVEN_DISABLE_AUTOGRAD_MULTITHREADING:-0}
export RAVEN_LOG_BACKWARD_RESOURCES=${RAVEN_LOG_BACKWARD_RESOURCES:-1}
export RAVEN_LOG_PARAM_UPDATES=${RAVEN_LOG_PARAM_UPDATES:-1}
export RAVEN_UP_GROUP_BACKEND=${RAVEN_UP_GROUP_BACKEND:-nccl}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export PYTHONPATH="${RAVEN_ROOT}:${PYTHONPATH:-}"
export H3_REF2VA_CONDITION_KV_CACHE=0
export ROOT RAVEN_ROOT MODEL_BASE MODEL_ROOT H3_DIT_DCP TEXT_ENCODER_DCP
export PYTHON_BIN TORCHRUN_BIN FFMPEG_EXE CONFIG TRAIN_MANIFEST OUTPUT_ROOT EXP_NAME
export FINAL32 COVE_INDEX NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export MINIMAX_H3_REF_DECODE_CACHE="${REF_DECODE_CACHE}"
export VAL_BEFORE_TRAIN
VALIDATION_TEACACHE=${VALIDATION_TEACACHE:-1}
if [[ "${VALIDATION_TEACACHE}" == "1" ]]; then
  export H3_REF2VA_TCACHE=1
  export H3_REF2VA_TCACHE_COMPUTE_STEPS=${H3_REF2VA_TCACHE_COMPUTE_STEPS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,49}
else
  unset H3_REF2VA_TCACHE
  unset H3_REF2VA_TCACHE_COMPUTE_STEPS
fi
export VALIDATION_TEACACHE
unset H3_REFCACHEBLEND_EXPORT_DIR
unset H3_REFCACHEBLEND_IMPORT_TEXT_DIR
unset H3_REFCACHEBLEND_IMPORT_ASSET_DIRS

echo "REF2VA_LORA_DISTILL root=${ROOT}"
echo "nodes=${SLURM_NNODES:-1} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "config=${CONFIG}"
echo "train_manifest=${TRAIN_MANIFEST}"
echo "output_root=${OUTPUT_ROOT}"
echo "exp_name=${EXP_NAME}"
echo "algorithm_batch_size=16 sp=4"
echo "ref_decode_cache=${MINIMAX_H3_REF_DECODE_CACHE}"
echo "val_before_train=${VAL_BEFORE_TRAIN}"
echo "validation_teacache=${VALIDATION_TEACACHE} compute_steps=${H3_REF2VA_TCACHE_COMPUTE_STEPS:-all}"
echo "job=${SLURM_JOB_ID:-manual} nodelist=${SLURM_JOB_NODELIST:-unknown}"

cd "${RAVEN_ROOT}"
if [[ "${PREDECODE_VALIDATION_REFS:-1}" == "1" ]]; then
  echo "predecoding validation references before model load"
  srun --kill-on-bad-exit=1 \
    --ntasks="${SLURM_NNODES:-1}" \
    --ntasks-per-node=1 \
    "${PYTHON_BIN}" "${ROOT}/scripts/predecode_covebench_validation_refs.py" \
      --manifest "${FINAL32}" \
      --rollout-index "${COVE_INDEX}" \
      --reference-root "${ROOT}/artifacts/covebench_teacher_rollout/cases" \
      --cache-dir "${MINIMAX_H3_REF_DECODE_CACHE}"
  echo "predecoding validation references complete"
fi

srun --kill-on-bad-exit=1 \
  --ntasks="${SLURM_NNODES:-1}" \
  --ntasks-per-node=1 \
  bash -lc '"${TORCHRUN_BIN}" \
    --nnodes "${SLURM_NNODES:-1}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${SLURM_NODEID:-0}" \
    --rdzv-backend c10d \
    --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    -m common.launch --config "${CONFIG}" \
    data.args.manifest_path "${TRAIN_MANIFEST}" \
    data.args.exclude_ids_path "${FINAL32}" \
    validation.covebench_manifest "${FINAL32}" \
    validation.covebench_rollout_index "${COVE_INDEX}" \
    engine.val_before_train "${VAL_BEFORE_TRAIN}" \
    engine.training_steps "${TRAINING_STEPS:-100000}" \
    engine.log_interval "${LOG_INTERVAL:-10}" \
    engine.early_stop_hours "${EARLY_STOP_HOURS:-1.8}" \
    validation.processor_path "${MODEL_ROOT}/Ref2VA/processor" \
    validation.covebench_reference_root "${ROOT}/artifacts/covebench_teacher_rollout/cases" \
    validation.covebench_teacher_video_root "${ROOT}/artifacts/covebench_teacher_rollout_teacache_2x/videos" \
    meta_model.packer_data.args.tokenizer_path "${MODEL_ROOT}/Ref2VA/tokenizer" \
    data.args.num_frames 124 \
    data.args.height 768 \
    data.args.width 1344 \
    data.args.max_seqlen "${MAX_SEQLEN:-262144}" \
    data.args.max_seqlen_per_sample "${MAX_SEQLEN:-262144}" \
    models.backbone.placement.plugins.0.path "${H3_DIT_DCP}" \
    models.teacher_backbone.placement.plugins.0.path "${H3_DIT_DCP}" \
    models.text_encoder.args.pretrained_model_name_or_path "${MODEL_ROOT}/Ref2VA/text_encoder" \
    models.text_encoder.placement.plugins.0.path "${TEXT_ENCODER_DCP}" \
    models.video_vae.weight.path "${MODEL_ROOT}/Ref2VA/video_vae/source/model.safetensors" \
    models.audio_vae.weight.path "${MODEL_ROOT}/Ref2VA/audio_vae/model.safetensors" \
    persistence.output_dir "${OUTPUT_ROOT}" \
    persistence.exp_name "${EXP_NAME}"'
