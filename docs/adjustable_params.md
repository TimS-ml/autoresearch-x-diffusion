# x-DDPM Adjustable Parameters

Reference for all parameters in `train_diffusion.py`. The goal is to lower `val_loss`
(denoising MSE) within the 5-minute time budget.

---

## Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQ_LEN` | `128` | Bytes per sample. Must be divisible by `2^(len(UNET_DIM_MULTS)-1)`. Longer = more context, larger memory. |
| `VOCAB_SIZE` | `256` | Fixed. enwik8 is byte-level. Do not change. |

---

## Embedding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMB_DIM` | `32` | Continuous dimension per byte token. Larger = richer but more VRAM. Try 16, 64. Must match `Unet1D(channels=EMB_DIM)`. |

---

## Unet1D Architecture

These parameters control the backbone denoising network.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNET_DIM` | `64` | Base channel dimension. Wider = more expressive, more VRAM. Try 32, 128. |
| `UNET_DIM_MULTS` | `(1, 2, 4)` | Channel multipliers at each resolution level. More levels = deeper U-Net but `SEQ_LEN` must be divisible by `2^(len-1)`. |
| `channels` | `EMB_DIM` | Input channel count. Set via `EMB_DIM`. |
| `self_condition` | `False` | Self-conditioning: feed previous prediction back as extra input. Costs ~25% more compute, improves quality. |
| `learned_sinusoidal_cond` | `False` | Use learned sinusoidal timestep embeddings instead of fixed. |
| `random_fourier_features` | `False` | Use random Fourier features for timestep embedding (fixed random). |
| `attn_dim_head` | `32` | Dimension per attention head in bottleneck. |
| `attn_heads` | `4` | Number of attention heads in bottleneck. |
| `dropout` | `0.` | Dropout in ResNet blocks. Try 0.1 if overfitting. |

### dim_mults Guide

| `UNET_DIM_MULTS` | Min `SEQ_LEN` divisor | Approx params (dim=64) | Notes |
|------------------|-----------------------|------------------------|-------|
| `(1, 2, 4)` | 4 | ~5M | Default (3 levels) |
| `(1, 2, 4, 8)` | 8 | ~10M | Deeper, needs `SEQ_LEN >= 8` |
| `(1, 2)` | 2 | ~2M | Shallow, fast |

---

## Diffusion Process

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIMESTEPS` | `1000` | Number of diffusion steps T during training. More = smoother noise schedule. Try 500 for faster per-step speed. |
| `SAMPLING_TIMESTEPS` | `50` | Steps for DDIM sampling (generation only). Does NOT affect training speed. |
| `OBJECTIVE` | `'pred_v'` | Training objective. See table below. |
| `BETA_SCHEDULE` | `'cosine'` | Noise schedule. `'cosine'` (default) or `'linear'`. |
| `DDIM_SAMPLING_ETA` | `0.` | DDIM stochasticity: 0=deterministic, 1=DDPM-equivalent. |

### Objective Comparison

| Objective | Description | Best for |
|-----------|-------------|----------|
| `pred_noise` | Predict the noise ε added at step t (original DDPM) | Linear schedule, simple baseline |
| `pred_x0` | Predict clean data x_0 directly | Can be unstable; try if pred_v underperforms |
| `pred_v` | Predict velocity v = √ᾱ·ε − √(1-ᾱ)·x_0 | Cosine schedule, modern models (SD 2.x, Imagen-Video) |

### Beta Schedule Comparison

| Schedule | Description | When to use |
|----------|-------------|-------------|
| `'cosine'` | Gradual SNR decay, preserves structure longer | Default, better for short sequences |
| `'linear'` | Aggressive noise increase | Works well with pred_noise objective |

---

## Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | `1e-4` | Initial LR for AdamW. Diffusion models are sensitive: try 3e-4 (fast) or 5e-5 (conservative). |
| `BATCH_SIZE` | `64` | Micro-batch size. Larger = more stable gradients but more VRAM. Try 32, 128. |
| `GRADIENT_ACCUMULATE_EVERY` | `1` | Gradient accumulation steps. Effective batch = BATCH_SIZE × this. |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay. |
| `GRAD_CLIP` | `1.0` | Gradient norm clipping. Diffusion models can have spiky gradients. |

---

## LR Schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WARMUP_RATIO` | `0.05` | Fraction of budget for linear warmup. |
| `WARMDOWN_RATIO` | `0.40` | Fraction of budget for cosine cooldown at end. |
| `FINAL_LR_FRAC` | `0.1` | Final LR as fraction of initial (at end of cooldown). |

---

## Memory Estimates (RTX 4090, BF16)

Rough peak VRAM for `SEQ_LEN=128`:

| Config | BATCH_SIZE | VRAM (est.) |
|--------|-----------|-------------|
| dim=64, mults=(1,2,4) | 64 | ~4 GB |
| dim=128, mults=(1,2,4) | 32 | ~6 GB |
| dim=64, mults=(1,2,4,8) | 32 | ~5 GB |
| dim=64, mults=(1,2,4), SEQ_LEN=256 | 32 | ~5 GB |

Diffusion models generally need less VRAM than transformers per sample because there are
no KV caches. The main bottleneck is the batch size and the Unet channel width.

---

## Key Differences from x-transformers LM

| Property | x-transformers (AR LM) | x-DDPM (diffusion) |
|----------|----------------------|---------------------|
| Data | Discrete tokens | Continuous embeddings |
| Forward pass | Autoregressive (causal) | U-Net denoising at random timestep t |
| Loss | Cross-entropy | MSE (denoising) |
| Metric | val_bpc (bits/char) | val_loss (MSE) |
| Generation | Token-by-token | Iterative denoising (50-1000 steps) |
| Optimizer | MuonAdamAtan2 (Muon for linear) | AdamW (Conv layers, no Muon advantage) |
| Batch size | 24 (large context) | 64 (short context, no KV cache) |

---

## Experiment Ideas (Ordered by Expected Impact)

### Quick Wins
1. Tune `LEARNING_RATE`: try `3e-4`, `5e-5`
2. Increase `BATCH_SIZE` to `128` if VRAM allows
3. Try `UNET_DIM=128` with `BATCH_SIZE=32`
4. Enable `self_condition=True` in Unet1D

### Architecture
5. Try `UNET_DIM_MULTS=(1, 2, 4, 8)` with `SEQ_LEN=128`
6. Try `EMB_DIM=64` with adjusted `UNET_DIM`
7. Try `SEQ_LEN=256` with `BATCH_SIZE=32`

### Diffusion
8. Try `OBJECTIVE='pred_noise'` (compare to pred_v)
9. Try `TIMESTEPS=500` (more steps per time budget)
10. Try `BETA_SCHEDULE='linear'` with `pred_noise`

### Advanced
11. Try `learned_sinusoidal_cond=True` for timestep embedding
12. Try `random_fourier_features=True`
13. Add `dropout=0.1` to ResNet blocks
