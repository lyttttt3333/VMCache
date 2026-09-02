#!/usr/bin/env bash
#SBATCH --job-name=h3-ref2va-dcp
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ref2va_dcp_%j.out
#SBATCH --error=logs/ref2va_dcp_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && "${SCRIPT_DIR}" == *spool* ]]; then
  ROOT=${ROOT:-${SLURM_SUBMIT_DIR}}
else
  ROOT=${ROOT:-${SCRIPT_DIR}}
fi

RAVEN_ROOT=${RAVEN_ROOT:-${ROOT}/../RAVEN}
PYTHON_BIN=${PYTHON_BIN:-/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/sana/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/sana/bin/torchrun}
HF_ROOT=${HF_ROOT:-/home/yitongl/code/models/MiniMax-H3-hf-9d710aed/FL2VA}
DCP_ROOT=${DCP_ROOT:-/home/yitongl/code/models/MiniMax-H3-DCP/Ref2VA}
NPROC=${NPROC:-32}

mkdir -p "${ROOT}/logs" "${ROOT}/.cache/tmp" "${DCP_ROOT}"
export TMPDIR="${ROOT}/.cache/tmp"
export PYTHONPATH="${RAVEN_ROOT}:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${HF_ROOT}/transformer/model.safetensors.index.json"
test -s "${HF_ROOT}/text_encoder/model.safetensors.index.json"

convert_one() {
  local name=$1
  local src=$2
  local out=$3
  shift 3
  if [[ -s "${out}/.metadata" ]]; then
    echo "SKIP_${name}=${out}"
    return
  fi
  if [[ -d "${out}" ]]; then
    mv "${out}" "${out}.incomplete.${SLURM_JOB_ID:-manual}.$(date +%s)"
  fi
  mkdir -p "$(dirname "${out}")"
  echo "CONVERT_${name}_SRC=${src}"
  echo "CONVERT_${name}_OUT=${out}"
  cd "${RAVEN_ROOT}"
  "${TORCHRUN_BIN}" --standalone --nproc_per_node "${NPROC}" \
    projects/minimax_h3/tools/convert_to_dcp.py \
    --src "${src}" \
    --out "${out}" \
    "$@"
  test -s "${out}/.metadata"
}

convert_one transformer \
  "${HF_ROOT}/transformer/model.safetensors.index.json" \
  "${DCP_ROOT}/transformer" \
  --minimax-h3-dit --dtype float32

convert_one text_encoder \
  "${HF_ROOT}/text_encoder/model.safetensors.index.json" \
  "${DCP_ROOT}/text_encoder"

echo "DCP_ROOT=${DCP_ROOT}"
