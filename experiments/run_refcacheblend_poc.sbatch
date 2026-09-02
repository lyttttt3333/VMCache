#!/usr/bin/env bash
#SBATCH --job-name=h3-refcacheblend-poc
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/refcacheblend_poc_%j.out
#SBATCH --error=logs/refcacheblend_poc_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && "${SCRIPT_DIR}" == /cm/local/apps/slurm/var/spool/* ]]; then
  DEFAULT_ROOT=${SLURM_SUBMIT_DIR}
else
  DEFAULT_ROOT=${SCRIPT_DIR}
fi
ROOT=${ROOT:-${DEFAULT_ROOT}}
RAVEN_ROOT=${RAVEN_ROOT:-${ROOT}/../RAVEN}
MODEL_BASE=${MODEL_BASE:-/home/yitongl/code/models}
MODEL_ROOT=${MODEL_ROOT:-${MODEL_BASE}/MiniMax-H3}
H3_DIT_DCP=${H3_DIT_DCP:-${MODEL_BASE}/MiniMax-H3-DCP/Ref2VA/transformer}
TEXT_ENCODER_DCP=${TEXT_ENCODER_DCP:-${MODEL_BASE}/MiniMax-H3-DCP/Ref2VA/text_encoder}
PYTHON_BIN=${PYTHON_BIN:-/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/h3-nrt/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/h3-nrt/bin/torchrun}
CASE_PYTHON=${CASE_PYTHON:-python3}
FFMPEG_EXE=${FFMPEG_EXE:-/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/h3-nrt/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
CONFIG=${CONFIG:-${ROOT}/minimax_h3_ref2va_3video_50nfe.yaml}
OUTPUT_ROOT=${ROOT}/outputs
POC_ROOT=${ROOT}/artifacts/refcacheblend_poc_case00

mkdir -p "${ROOT}/logs" "${OUTPUT_ROOT}" "${ROOT}/.cache/tmp" "${ROOT}/.cache/triton" "${ROOT}/.cache/huggingface" "${ROOT}/.cache/bin" "${POC_ROOT}"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${CONFIG}"
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
export H3_REFCACHEBLEND_FORCE_VISUAL_CONDITION_COUNT=3

emit_case_env() {
  local manifest=$1
  local env_path=$2
  "${CASE_PYTHON}" - "${manifest}" "${ROOT}" >"${env_path}" <<'PY'
import json
from pathlib import Path
import shlex
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
case = manifest["cases"][0]
target = manifest["target"]

def relocate(value):
    path = Path(value)
    if path.exists():
        return str(path)
    text = str(value)
    for marker in ("assets_3video16/", "artifacts/"):
        if marker in text:
            candidate = root / marker / text.split(marker, 1)[1]
            if candidate.exists():
                return str(candidate)
    return text

conditions = []
media_probes = []
resolved_shapes = []
for condition in case["conditions"]:
    relocated = dict(condition)
    relocated["path"] = relocate(condition["path"])
    conditions.append(
        {
            "type": relocated["type"],
            "path": relocated["path"],
            "start_time_seconds": relocated["start_time_seconds"],
        }
    )
    media_probes.append(relocated["video_probe"])
    resolved_shapes.append(relocated["resolved_shape"])

values = {
    "CASE_ID": case["id"],
    "SEED": case["seed"],
    "WIDTH": target["width"],
    "HEIGHT": target["height"],
    "FPS": target["fps"],
    "NUM_FRAMES": target["num_frames"],
    "PROMPTS_JSON": json.dumps([case["prompt"]]),
    "CONDITIONS_JSON": json.dumps(conditions),
    "MEDIA_PROBES_JSON": json.dumps(media_probes),
    "RESOLVED_SHAPES_JSON": json.dumps(resolved_shapes),
}
for item in conditions:
    if not Path(item["path"]).is_file():
        raise SystemExit(f"missing condition path: {item['path']}")
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

run_case() {
  local manifest=$1
  local tag=$2
  local steps=$3
  local cache_enabled=$4
  local export_dir=${5:-}
  local import_text_dir=${6:-}
  local import_asset_dirs=${7:-}
  local env_path="${ROOT}/.cache/refcacheblend_${tag}_${SLURM_JOB_ID:-manual}.env"
  emit_case_env "${manifest}" "${env_path}"
  source "${env_path}"

  unset H3_REFCACHEBLEND_EXPORT_DIR
  unset H3_REFCACHEBLEND_IMPORT_TEXT_DIR
  unset H3_REFCACHEBLEND_IMPORT_ASSET_DIRS
  export H3_REF2VA_CONDITION_KV_CACHE="${cache_enabled}"
  if [[ -n "${export_dir}" ]]; then
    rm -rf "${export_dir}"
    mkdir -p "${export_dir}"
    export H3_REFCACHEBLEND_EXPORT_DIR="${export_dir}"
  fi
  if [[ -n "${import_text_dir}" ]]; then
    export H3_REFCACHEBLEND_IMPORT_TEXT_DIR="${import_text_dir}"
  fi
  if [[ -n "${import_asset_dirs}" ]]; then
    export H3_REFCACHEBLEND_IMPORT_ASSET_DIRS="${import_asset_dirs}"
  fi

  local exp_name="refcacheblend-poc-${tag}-${CASE_ID}-${SLURM_JOB_ID:-manual}"
  local latent_t=$(( ((NUM_FRAMES - 5) / 17) * 5 + 2 ))
  local latent_h=$(( HEIGHT / 16 ))
  local latent_w=$(( WIDTH / 16 ))
  local target_rows=$(( latent_t * (latent_h / 2) * (latent_w / 2) ))
  local audio_t=$(( NUM_FRAMES * 40 / FPS ))
  local media_rows=$(( target_rows * 4 + 2 * audio_t ))
  local max_seqlen=${MAX_SEQLEN:-262144}
  local master_port=$((33000 + (${SLURM_JOB_ID:-$$} % 20000)))

  local phase_start
  phase_start=$(date +%s)
  echo "PHASE=${tag} CASE=${CASE_ID} steps=${steps} cache=${H3_REF2VA_CONDITION_KV_CACHE} export=${export_dir:-none} import_text=${import_text_dir:-none} import_assets=${import_asset_dirs:-none}"
  cd "${RAVEN_ROOT}"
  "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node 4 \
    --master-port "${master_port}" \
    -m common.launch --config "${CONFIG}" \
    validation.seed "${SEED}" \
    validation.ref_noise_seed "${SEED}" \
    validation.num_frames "${NUM_FRAMES}" \
    validation.height "${HEIGHT}" \
    validation.width "${WIDTH}" \
    validation.fps "${FPS}" \
    validation.conditions "${CONDITIONS_JSON}" \
    validation.media_probes "${MEDIA_PROBES_JSON}" \
    validation.resolved_shapes "${RESOLVED_SHAPES_JSON}" \
    validation.processor_path "${MODEL_ROOT}/Ref2VA/processor" \
    validation.visual_anchor "${VISUAL_ANCHOR:-0.999}" \
    validation.audio_anchor "${AUDIO_ANCHOR:-1.0}" \
    validation.prompts "${PROMPTS_JSON}" \
    validation.num_prompts 1 \
    validation.batch_size 1 \
    logging.plugins [] \
    diffusion.sampling_timesteps.args.num_sampling_steps "${steps}" \
    diffusion.audio_sampling_timesteps.args.num_sampling_steps "${steps}" \
    data.args.num_frames "${NUM_FRAMES}" \
    data.args.height "${HEIGHT}" \
    data.args.width "${WIDTH}" \
    data.args.max_seqlen "${max_seqlen}" \
    data.args.max_seqlen_per_sample "${max_seqlen}" \
    data.args.tokenizer_path "${MODEL_ROOT}/Ref2VA/tokenizer" \
    models.backbone.placement.plugins.0.path "${H3_DIT_DCP}" \
    models.text_encoder.args.pretrained_model_name_or_path "${MODEL_ROOT}/Ref2VA/text_encoder" \
    models.text_encoder.placement.plugins.0.path "${TEXT_ENCODER_DCP}" \
    models.video_vae.weight.path "${MODEL_ROOT}/Ref2VA/video_vae/source/model.safetensors" \
    models.audio_vae.weight.path "${MODEL_ROOT}/Ref2VA/audio_vae/model.safetensors" \
    persistence.output_dir "${OUTPUT_ROOT}" \
    persistence.exp_name "${exp_name}"

  local result="${OUTPUT_ROOT}/minimax_h3_ref2va_rgb_depth/${exp_name}/media/0000000/validation/backbone/prompt0000.mp4"
  test -s "${result}"
  echo "RESULT_${tag}=${result}"
  echo "ELAPSED_${tag}=$(( $(date +%s) - phase_start ))"
}

ABC="${ROOT}/refcacheblend_poc_manifests/abc.json"
A="${ROOT}/refcacheblend_poc_manifests/a.json"
B="${ROOT}/refcacheblend_poc_manifests/b.json"
C="${ROOT}/refcacheblend_poc_manifests/c.json"

run_case "${ABC}" "teacher_abc" 50 0
run_case "${ABC}" "fullcache_abc_export" 50 1 "${POC_ROOT}/cache_abc"
run_case "${A}" "source_a_export" 1 1 "${POC_ROOT}/cache_a"
run_case "${B}" "source_b_export" 1 1 "${POC_ROOT}/cache_b"
run_case "${C}" "source_c_export" 1 1 "${POC_ROOT}/cache_c"
run_case "${ABC}" "composed_abc_from_a_b_c" 50 1 "" "${POC_ROOT}/cache_abc" "${POC_ROOT}/cache_a:${POC_ROOT}/cache_b:${POC_ROOT}/cache_c"

echo "POC_ROOT=${POC_ROOT}"
