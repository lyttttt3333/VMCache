# Decoupled Reference-Attention Distillation

## Objective

For the same clean target latent, sampled timestep, and noise, run two DiT
forwards:

- Coupled teacher: native MiniMax-H3 joint self-attention over text, every
  reference image/video, and noisy generation tokens.
- Decoupled student: each reference group self-attends independently; text and
  noisy generation tokens keep dense access to all reference groups.

The student minimizes FP32 MSE against the detached teacher video and audio
velocity predictions. Only LoRA adapters on the DiT are trainable. The teacher,
student base weights, text encoder, video VAE, and audio VAE are frozen.

## Training Configuration

- LoRA: rank 16, alpha 16, BF16, 158,044,160 trainable parameters.
- Optimizer: AdamW, learning rate `1e-4`, no weight decay.
- Sequence parallelism: SP=4.
- Algorithm batch size: 16, independent of allocated GPU count.
- CW 16-GPU layout: four SP groups process four samples concurrently and use
  four gradient-accumulation rounds per optimizer step.
- Checkpoints: trainable LoRA, optimizer, dataloader, RNG, and engine state every
  10 optimizer steps; retain the latest two and resume automatically.
- Validation: CoVeBench final-32 at step 0 and every 100 optimizer steps. The 32
  cases must be excluded from training.
- Validation sampling: 50 denoising indices with TeaCache computing 25 indices,
  approximately 2x denoising acceleration. Each result is decoded to MP4 and
  accompanied by JSON metadata.

The validation fingerprint covers prompts, seeds, reference file metadata,
manifest/index hashes, target shape, LoRA topology, and all TeaCache settings.
Interrupted validation can therefore reuse only matching, complete MP4 results.

## Workspace Inputs

The repository intentionally excludes model weights, latent rollouts, reference
media, generated MP4 files, and checkpoints. Supply these paths through the
launcher environment:

```bash
export ROOT=/path/to/runtime_workspace
export MODEL_BASE=/path/to/models
export TRAIN_MANIFEST=/path/to/ref2va_decoupled_distill_train_manifest.json
export FINAL32=/path/to/covebench_validation_final_32.json
export COVE_INDEX=/path/to/covebench_teacher_rollout_index.json
export OUTPUT_ROOT=/path/to/output/runs
export EXP_NAME=lora-r16-bf16-bsz16-sp4-decoupled
sbatch -p interactive -N 2 --gres=gpu:8 run_ref2va_decoupled_distill_lora_cw.sbatch
```

`ROOT` must contain the overlaid `RAVEN/` checkout, this configuration, the
launcher, and `scripts/predecode_covebench_validation_refs.py`. The launcher
stops before the two-hour partition limit, and a dependency chain can submit the
same command repeatedly; `engine.resume=auto` restores the latest complete
checkpoint.

## Data Roadmap

The initial 600-case training manifest uses one reference video per case. It is
the stability baseline for the distillation and resume machinery. The next data
stage should add the curated V5 32-case set with three unique reference videos
per case, generate native coupled-teacher targets, and oversample those cases so
training directly covers reference-reference interaction removal and composable
`A/B/C` cache reuse.

`scripts/build_v5_3ref_teacher_rollout_index.py` validates the 32-case source,
requires all 96 references to be globally unique and present, and emits the
rollout-worker schema. `run_v5_3ref_teacher_rollout_teacache2x_cw.sbatch` then
runs four cases per persistent 4-GPU worker on `batch_short`, saving both the
teacher latent and an MP4 for review. Its default array concurrency is two
workers; increase it only when those GPUs will not delay the main training
resume chain.
