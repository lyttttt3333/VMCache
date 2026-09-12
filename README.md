# RefCacheBlend Condition Cache POC

This repository contains the MiniMax-H3 Ref2VA condition-cache proof of concept
and the LoRA distillation pipeline for a decomposable reference-attention model.

It does not include model weights, generated videos, or reference assets.

## Contents

- `RAVEN_patch/`: patched files to overlay on top of `mvp-ai-lab/RAVEN`.
- `experiments/`: POC manifests, asset-prep script, Slurm runner, and summary script.
- `minimax_h3_ref2va_decoupled_distill.yaml`: LoRA-only teacher/student training configuration.
- `run_ref2va_decoupled_distill_lora_cw.sbatch`: resumable 16-GPU CW launcher.
- `scripts/`: manifest checks and cached reference-media preprocessing.
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

## Decoupled Reference Training

The frozen MiniMax-H3 backbone is used as the coupled teacher. The student uses
the same backbone with trainable rank-16 LoRA adapters, but reference image/video
tokens only self-attend within each reference group. Generation and text tokens
remain dense and may attend all reference groups. See
[`DECOUPLED_TRAINING.md`](DECOUPLED_TRAINING.md) for the exact objective,
parallelism invariants, validation protocol, and launch procedure.
