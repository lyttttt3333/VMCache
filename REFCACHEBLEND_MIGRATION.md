# RefCacheBlend POC Migration

## Goal

Run a tuning-free composable reference-cache proof of concept for MiniMax-H3 Ref2VA:

- `teacher_abc`: full `[A,B,C,G]` self-attention.
- `fullcache_abc_export`: normal ABC condition cache, exporting ABC text/reference cache parts.
- `source_a_export`, `source_b_export`, `source_c_export`: single-reference 1-step cache-source runs.
- `composed_abc_from_a_b_c`: ABC generation using `text_ABC + cache_A + cache_B + cache_C`.

This tests whether asset-level reference K/V caches computed independently can be recomposed for a multi-reference request without training.

## Code Base

Base RAVEN:

```bash
git clone https://github.com/mvp-ai-lab/RAVEN.git RAVEN
cd RAVEN
git checkout 58a133e095e1fa335633fec50f4c166db08d7867
```

Required patched files:

```text
RAVEN/projects/minimax_h3/modeling/transformer/model_sp.py
RAVEN/projects/minimax_h3/meta_models/minimax_h3_ref2va.py
RAVEN/projects/minimax_h3/modeling/ref2va_reference.py
```

Experiment files:

```text
experiments/prepare_heavy_condition_asset_dataset_v5_32_pexels.py
experiments/prepare_heavy_condition_asset_dataset_v6_8_pexels.py
experiments/build_v6_ref2va_run_manifest.py
experiments/build_refcacheblend_poc_manifests.py
experiments/run_refcacheblend_poc.sbatch
experiments/summarize_refcacheblend_poc.py
experiments/minimax_h3_ref2va_3video_50nfe.yaml
experiments/heavy_ref2va_v6_8_organic_pexels_hotmatch_3video_cases.json
```

## Runtime Env Vars

POC-only vars:

```bash
H3_REFCACHEBLEND_FORCE_VISUAL_CONDITION_COUNT=3
H3_REFCACHEBLEND_EXPORT_DIR=/path/to/export/cache
H3_REFCACHEBLEND_IMPORT_TEXT_DIR=/path/to/cache_abc
H3_REFCACHEBLEND_IMPORT_ASSET_DIRS=/path/to/cache_a:/path/to/cache_b:/path/to/cache_c
```

Existing condition-cache vars:

```bash
H3_REF2VA_CONDITION_KV_CACHE=1
H3_REF2VA_CONDITION_KV_CACHE_REFRESH_STEPS=
H3_REF2VA_TCACHE=0
```

## Expected Directory

On target clusters:

```text
~/code/condition_cache/
  RAVEN/
  experiments/
```

The current runner expects model assets in:

```text
~/code/models/MiniMax-H3/Ref2VA
~/code/models/MiniMax-H3-DCP/Ref2VA/transformer
~/code/models/MiniMax-H3-DCP/Ref2VA/text_encoder
```

CS and DRACO currently have some MiniMax-H3 model directories, but not this exact Ref2VA/DCP layout. Re-download or convert those assets before launching the POC.

## Reference Assets

The POC manifest can be regenerated from the V6 Pexels dataset. Do not run the download/transcode script on a login node; submit it to a CPU/data partition.

```bash
cd ~/code/condition_cache/experiments
export H3_CONDITION_CACHE_ROOT=$PWD
python3 prepare_heavy_condition_asset_dataset_v6_8_pexels.py
python3 build_v6_ref2va_run_manifest.py
python3 build_refcacheblend_poc_manifests.py
```

Expected asset root:

```text
experiments/artifacts/heavy_condition_asset_dataset_v6_8_organic_pexels_hotmatch_32aligned/
```

## Run

```bash
cd ~/code/condition_cache/experiments
python3 build_refcacheblend_poc_manifests.py
sbatch -p <partition> -t 02:00:00 --mem=512G run_refcacheblend_poc.sbatch
```

After completion:

```bash
python3 summarize_refcacheblend_poc.py <job_id>
```

Summary outputs:

```text
experiments/summaries/refcacheblend_poc_<job_id>/
  teacher_abc.mp4
  fullcache_abc_export.mp4
  composed_abc_from_a_b_c.mp4
  metrics.csv
  metrics.json
```

## Current NRT Status

NRT job `6518159` was submitted on `batch_short` but remained pending with `QOSGrpGRES`.

Previous failed NRT job `6517947` failed in 6 seconds because Slurm executed the script from its spool directory and the runner resolved `ROOT` incorrectly. This has been fixed by using `SLURM_SUBMIT_DIR` when `BASH_SOURCE[0]` points into `/cm/local/apps/slurm/var/spool/`.

## 2026-09-02 CS/DRACO Update

GitHub source is available at:

```text
https://github.com/lyttttt3333/VMCache
```

Latest pushed commit during migration:

```text
926ae0f Use torch 2.8 env for DCP conversion
```

CS target directory:

```text
/home/yitongl/code/condition_cache
```

Current CS status:

- Reference assets were regenerated successfully by Slurm job `33664350`.
- Generated manifest: `experiments/heavy_ref2va_v6_8_organic_pexels_hotmatch_3video_cases.json`.
- Generated POC manifests: `experiments/refcacheblend_poc_manifests/{abc,a,b,c}.json`.
- Missing Ref2VA support files were restored from a previous successful run archive into `RAVEN/projects/minimax_h3/modeling/ref2va_encoder.py` and `RAVEN/projects/minimax_h3/trials/base/minimax_h3_ref2va/minimax_h3_ref2va_rgb_depth_50nfe.yaml`.
- A CS-compatible GPU launcher is staged at `experiments/run_refcacheblend_poc_cs.sbatch`; it defaults to the `sana` conda env and `polar4` with 4 GPUs.
- DCP conversion jobs `33664848` and `33667026` were cancelled while pending.
- DCP conversion job `33667350` started but failed because the `sana` PyTorch 2.5 environment does not support `distribute_tensor(..., src_data_rank=None)`.
- Active replacement DCP conversion job `33668723` is pending across `cpu_short,cpu,cpu_long` and uses the `lmflow` PyTorch 2.8 environment, which supports that API. It will write:
  - `/home/yitongl/code/models/MiniMax-H3-DCP/Ref2VA/transformer`
  - `/home/yitongl/code/models/MiniMax-H3-DCP/Ref2VA/text_encoder`

NRT is currently not reachable from `cw-dfw-cs-001-vscode-02`; SSH returns `Permission denied (publickey,password)`.

Once DCP conversion completes, launch the POC on CS:

```bash
cd /home/yitongl/code/condition_cache/experiments
sbatch run_refcacheblend_poc_cs.sbatch
```
