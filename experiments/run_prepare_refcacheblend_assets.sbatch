#!/usr/bin/env bash
#SBATCH --job-name=refcacheblend-assets
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/refcacheblend_assets_%j.out
#SBATCH --error=logs/refcacheblend_assets_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${SLURM_SUBMIT_DIR:-}" && "${SCRIPT_DIR}" == *spool* ]]; then
  ROOT=${ROOT:-${SLURM_SUBMIT_DIR}}
else
  ROOT=${ROOT:-${SCRIPT_DIR}}
fi

cd "${ROOT}"
mkdir -p logs
export H3_CONDITION_CACHE_ROOT="${ROOT}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 prepare_heavy_condition_asset_dataset_v6_8_pexels.py
python3 build_v6_ref2va_run_manifest.py
python3 build_refcacheblend_poc_manifests.py

echo "ASSET_ROOT=${ROOT}/artifacts/heavy_condition_asset_dataset_v6_8_organic_pexels_hotmatch_32aligned"
