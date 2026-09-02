# RefCacheBlend Condition Cache POC

This repository contains the minimal code needed to reproduce the MiniMax-H3 Ref2VA composable reference-cache proof of concept.

It does not include model weights, generated videos, or reference assets.

## Contents

- `RAVEN_patch/`: patched files to overlay on top of `mvp-ai-lab/RAVEN`.
- `experiments/`: POC manifests, asset-prep script, Slurm runner, and summary script.
- `REFCACHEBLEND_MIGRATION.md`: cluster setup and run notes.

## Base Repository

```bash
git clone https://github.com/mvp-ai-lab/RAVEN.git RAVEN
cd RAVEN
git checkout 58a133e095e1fa335633fec50f4c166db08d7867
cd ..
rsync -a RAVEN_patch/ RAVEN/
```

## Run

```bash
cd experiments
python3 build_refcacheblend_poc_manifests.py
sbatch -p <gpu_partition> -t 02:00:00 --mem=512G run_refcacheblend_poc.sbatch
```

After completion:

```bash
python3 summarize_refcacheblend_poc.py <job_id>
```

The runner expects MiniMax-H3 Ref2VA model assets under `~/code/models`; see `REFCACHEBLEND_MIGRATION.md`.

Reference assets are intentionally excluded. Recreate them on a CPU/data node:

```bash
cd experiments
sbatch -p cpu_short run_prepare_refcacheblend_assets.sbatch
```
