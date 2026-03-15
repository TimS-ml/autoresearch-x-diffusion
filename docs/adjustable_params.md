# x-DDPM Adjustable Parameters

Complete reference for all tunable parameters in `train_diffusion.py`.
Parameters are sourced from x-DDPM's `Unet1D` and `GaussianDiffusion1D` classes.
Goal: minimize `val_loss` (denoising MSE) within the 5-minute time budget.

---

## Data Parameters

| Constant | Default | Description |
|----------|---------|-------------|
| `SEQ_LEN` | `128` | Bytes per sample. Must be divisible by `2^(len(UNET_DIM_MULTS)-1)`. Longer = more context but fewer steps in budget. |
| `VOCAB_SIZE` | `256` | Fixed. enwik8 is byte-level. Do not change. |

---

## Embedding

| Constant | Default | Description |
|----------|---------|-------------|
| `EMB_DIM` | `32` | Continuous dimension per byte token. Maps to `Unet1D(channels=EMB_DIM)`. Larger = richer representation but more Unet channels. Try 16, 64. |

---

## Unet1D Architecture

All Unet1D constructor parameters, from `denoising_diffusion_pytorch_1d.py`:

| Constant | Unet1D kwarg | Default | Description |
|----------|-------------|---------|-------------|
| `UNET_DIM` | `dim` | `64` | Base channel dimension. All layer widths are `dim * mult` for each resolution level. Wider = more expressive. Try 32, 128, 256. |
| `UNET_INIT_DIM` | `init_dim` | `None` | Channel dim after the first Conv1d projection. `None` = same as `dim`. Can be set smaller for efficiency. |
| `UNET_DIM_MULTS` | `dim_mults` | `(1, 2, 4)` | Channel multipliers at each resolution level. Each level downsamples by 2x. `SEQ_LEN` must be divisible by `2^(len-1)`. |
| `EMB_DIM` | `channels` | `32` | Input/output data channels. Must match embedding dimension. |
| `UNET_DROPOUT` | `dropout` | `0.0` | Dropout in ResNet blocks. Try 0.1 if overfitting. |
| `SELF_CONDITION` | `self_condition` | `False` | Feed previous denoising prediction back as extra input. Doubles input channels, costs ~25% compute, often improves quality. |
| `LEARNED_SINUSOIDAL` | `learned_sinusoidal_cond` | `False` | Use learned (trainable) sinusoidal timestep embeddings instead of fixed. |
| `LEARNED_SINUSOIDAL_DIM` | `learned_sinusoidal_dim` | `16` | Dimension for learned sinusoidal embeddings (only used when `LEARNED_SINUSOIDAL=True`). |
| `ATTN_DIM_HEAD` | `attn_dim_head` | `32` | Dimension per attention head in the bottleneck (full attention, not linear). |
| `ATTN_HEADS` | `attn_heads` | `4` | Number of attention heads in the bottleneck. More heads = more capacity but slightly slower. |

### Parameters NOT exposed (hardcoded in x-DDPM)

These are set internally and cannot be tuned without modifying x-DDPM:

| Unet1D internal | Value | Notes |
|----------------|-------|-------|
| `time_dim` | `dim * 4` | Timestep embedding MLP width. Scales with `dim`. |
| Encoder blocks per level | 2 ResNet + 1 LinearAttention + Downsample | Fixed architecture. |
| Bottleneck | 1 ResNet + full Attention + 1 ResNet | Uses `attn_dim_head`/`attn_heads`. |
| Decoder blocks per level | 2 ResNet (with skip concat) + 1 LinearAttention + Upsample | Mirrors encoder. |

### dim_mults Guide

| `UNET_DIM_MULTS` | Min `SEQ_LEN` | Approx params (dim=64, emb=32) | Approx params (dim=128, emb=32) |
|------------------|---------------|-------------------------------|-------------------------------|
| `(1, 2)` | 2 | ~1.5M | ~6M |
| `(1, 2, 4)` | 4 | ~4.6M | ~18M |
| `(1, 2, 4, 8)` | 8 | ~10M | ~57M |

---

## Diffusion Process (GaussianDiffusion1D)

All GaussianDiffusion1D constructor parameters, from `denoising_diffusion_pytorch_1d.py`:

| Constant | GaussianDiffusion1D kwarg | Default | Description |
|----------|--------------------------|---------|-------------|
| `TIMESTEPS` | `timesteps` | `1000` | Number of diffusion noise steps T. More = smoother schedule, slower per-step but same budget. Try 500. |
| `SAMPLING_TIMESTEPS` | `sampling_timesteps` | `50` | DDIM sampling steps (generation only). Does NOT affect training. Fewer = faster generation. |
| `OBJECTIVE` | `objective` | `'pred_v'` | Training objective. See comparison below. |
| `BETA_SCHEDULE` | `beta_schedule` | `'cosine'` | Noise schedule shape. `'cosine'` or `'linear'`. |
| `DDIM_SAMPLING_ETA` | `ddim_sampling_eta` | `0.0` | DDIM stochasticity: 0=deterministic, 1=equivalent to full DDPM sampling. Generation only. |
| (hardcoded) | `auto_normalize` | `True` | Scales input from [0,1] to [-1,1] internally. Always True. |
| (hardcoded) | `channel_first` | `True` | Data layout (B, C, L). Always True. |

### Objective Comparison (empirically validated)

| Objective | Description | Empirical val_loss (dim=64, 5min) | Best for |
|-----------|-------------|----------------------------------|----------|
| `'pred_v'` | Velocity: `v = √ᾱ·ε − √(1-ᾱ)·x_0` | **0.0094** | Cosine schedule (default, recommended) |
| `'pred_noise'` | Predict noise ε (original DDPM) | 0.0142 | Linear schedule; simpler but worse with cosine |
| `'pred_x0'` | Predict clean data x_0 directly | (untested) | Can be unstable; worth trying |

### Beta Schedule Comparison

| Schedule | Description | When to use |
|----------|-------------|-------------|
| `'cosine'` | Gradual SNR decay, preserves signal longer | Default. Better for short sequences and pred_v. |
| `'linear'` | Aggressive early noise, gentle late noise | Use with `pred_noise`. |

---

## Optimization

| Constant | Default | Description |
|----------|---------|-------------|
| `LEARNING_RATE` | `1e-4` | Initial LR for AdamW. Try 3e-4 (aggressive) or 5e-5 (conservative). |
| `BATCH_SIZE` | `64` | Micro-batch size. Larger = more stable gradients. VRAM is cheap here; try 128 or 256. |
| `GRADIENT_ACCUMULATE_EVERY` | `1` | Gradient accumulation steps. Effective batch = `BATCH_SIZE × GRADIENT_ACCUMULATE_EVERY`. |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay. |
| `GRAD_CLIP` | `1.0` | Gradient norm clipping. |

---

## LR Schedule

| Constant | Default | Description |
|----------|---------|-------------|
| `WARMUP_RATIO` | `0.05` | Fraction of time budget for linear LR warmup. |
| `WARMDOWN_RATIO` | `0.40` | Fraction of time budget for cosine LR cooldown. |
| `FINAL_LR_FRAC` | `0.1` | Final LR as fraction of initial (end of cooldown). |

---

## Measured VRAM (RTX 4090, BF16, SEQ_LEN=128, BATCH_SIZE=64)

Real measurements from test runs:

| Config | Params | VRAM | val_loss (5min) | steps/5min |
|--------|--------|------|----------------|------------|
| dim=64, mults=(1,2,4) | 4.6M | **0.2 GB** | 0.0094 | ~9,900 |
| dim=128, mults=(1,2,4,8) | 57.4M | **1.4 GB** | 0.0047 | ~5,800 |

VRAM is extremely low because:
- No KV cache (diffusion, not autoregressive)
- Small sequences (128 tokens)
- Conv1d layers are memory-efficient

**The 24 GB budget is barely touched.** You can safely try dim=256 or BATCH_SIZE=256.

---

## Key Differences from x-transformers LM

| Property | x-transformers (AR LM) | x-DDPM (diffusion) |
|----------|----------------------|---------------------|
| Data | Discrete tokens | Continuous embeddings |
| Forward pass | Autoregressive (causal) | U-Net denoising at random timestep t |
| Loss | Cross-entropy | MSE (denoising) |
| Metric | `val_bpc` (bits/char) | `val_loss` (MSE) |
| Generation | Token-by-token | Iterative denoising (50-1000 steps) |
| Optimizer | MuonAdamAtan2 | AdamW (Conv layers, no Muon benefit) |
| Batch size | 24 (long context) | 64+ (short context, no KV cache) |
| VRAM | ~8 GB | ~0.2-1.4 GB |

---

## Experiment Ideas (Ordered by Expected Impact)

### Quick Wins (likely to improve val_loss)
1. Increase model size: `UNET_DIM=128`, `UNET_DIM_MULTS=(1,2,4,8)` — **validated: 0.0047 vs 0.0094**
2. Scale further: `UNET_DIM=256` with `BATCH_SIZE=32`
3. Tune `LEARNING_RATE`: try `3e-4`, `1e-3` (with clip)
4. Increase `BATCH_SIZE` to 128 or 256 (VRAM allows it)

### Architecture
5. Enable `SELF_CONDITION=True` (~25% more compute, often helps)
6. Try `EMB_DIM=64` — richer byte representation
7. Try `SEQ_LEN=256` with `BATCH_SIZE=32`
8. Try `ATTN_HEADS=8` or `ATTN_DIM_HEAD=64` (more bottleneck capacity)

### Diffusion Process
9. Try `TIMESTEPS=500` (more gradient steps per budget)
10. Try `OBJECTIVE='pred_x0'` (compare vs pred_v)
11. Try `LEARNED_SINUSOIDAL=True` (learnable timestep features)

### Regularization
12. Add `UNET_DROPOUT=0.1` (if overfitting)
13. Increase `WEIGHT_DECAY` to `1e-3`

### LR Schedule
14. Try `WARMDOWN_RATIO=0.5` (longer cooldown)
15. Try `FINAL_LR_FRAC=0.01` (more aggressive decay)
