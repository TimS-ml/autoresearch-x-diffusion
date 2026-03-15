# x-DDPM Autoresearch — Design Decisions

This document records the design choices made when extending the autoresearch framework
from x-transformers (autoregressive LM) to x-DDPM (denoising diffusion).

---

## 1. Task Framing: Discrete Text → Continuous Diffusion

**Problem**: DDPM operates on continuous data. enwik8 is discrete bytes (vocab=256).

**Options considered**:

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| A: Bit-representation (256→8 bits, treat as 8-channel float) | Simple, exact | Unnatural geometry for diffusion | Rejected |
| B: One-hot → diffuse (256-dim) | Exact representation | Very high-dimensional, expensive | Rejected |
| C: Learned embedding (256→D-dim) | Compact, learnable geometry | Decode requires nearest-neighbor | **Chosen** |
| D: Absorbing/masked diffusion on discrete tokens | Theoretically principled | Requires reimplementing DDPM; out of scope for this library | Deferred |

**Choice C** — learned embedding diffusion:
- Embed each byte token into a `EMB_DIM=32` dimensional continuous vector.
- The Unet1D processes `(B, EMB_DIM, SEQ_LEN)` shaped tensors.
- After sampling, decode back to bytes via nearest-neighbor in embedding space (cosine similarity).
- Simple, works with unmodified x-DDPM library.

---

## 2. Architecture: Unet1D (not 2D)

**Why 1D**: enwik8 is a 1D byte sequence. `Unet1D` operates on `(B, C, L)` tensors with Conv1d.
The 2D `Unet` would require reshaping to an artificial image grid — unnecessary.

**Unet1D config** (baseline):
```
dim = 64
dim_mults = (1, 2, 4)   # 3 levels, sequence shrinks: L → L/2 → L/4
channels = EMB_DIM       # input channels = embedding dim
```

The input sequence length must be divisible by `2^(len(dim_mults)-1) = 4`.
We use `SEQ_LEN = 128`, which is divisible by 4.

---

## 3. Diffusion Objective: `pred_v`

**Options**:
- `pred_noise` (original DDPM): Predict noise at each timestep. Numerically stable but
  known to blur at low noise levels.
- `pred_x0`: Directly predict clean data. Can be unstable early in training.
- `pred_v` (v-parameterization, Salimans & Ho 2022): Predict velocity
  `v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x_0`. Stable across all noise levels,
  works well for cosine schedule, used in Imagen-Video and SD 2.x.

**Choice**: `pred_v` — best for cosine schedule, most stable, well-validated.

---

## 4. Noise Schedule: Cosine

**Linear** schedule (original DDPM): Destroys structure too aggressively for short sequences.
**Cosine** schedule (Nichol & Dhariwal 2021): Gentler at high SNR, better for small sequences.

**Choice**: `cosine` — standard for modern DDPM, better empirical performance.

---

## 5. Sequence Length: 128

**Trade-off**: Longer sequences capture more text context; shorter sequences allow larger
batches and more steps in the 5-minute budget.

**Choice**: `SEQ_LEN = 128`.
- 128 bytes = enough to capture local character patterns (words, common phrases).
- Divisible by 4 (required by 3-level Unet1D).
- With `BATCH_SIZE = 64` and `EMB_DIM = 32`: ~64 MB activation memory per step.

---

## 6. Embedding Dimension: 32

**Trade-off**: Higher `EMB_DIM` → richer representation but more Unet channels to process.
**Choice**: `EMB_DIM = 32`.
- Enough to capture byte identity (256 → 32 is comparable to word2vec compression).
- Keeps the Unet width manageable (base `dim=64`, channels=32).
- The embedding matrix itself is only 256 × 32 = 8K parameters.

---

## 7. DDIM Sampling: 50 steps

Training uses `T=1000` timesteps. Full DDPM sampling would also take 1000 steps.
DDIM (denoising diffusion implicit models, Song et al. 2020) enables deterministic
sampling in fewer steps with minimal quality loss.

**Choice**: `sampling_timesteps = 50` — fast enough for the GENERATE_EVERY callback.

---

## 8. Optimizer: AdamW (not MuonAdamAtan2)

The Muon optimizer is designed for Transformer weight matrices. The Unet1D has Conv1d
layers, not linear projection matrices. Muon's gradient orthogonalization is not
obviously beneficial for conv kernels.

**Choice**: Standard `AdamW` with `lr=1e-4`, `betas=(0.9, 0.99)`.

---

## 9. Metric: `val_loss` (denoising MSE) and `val_bpd`

**Key difference from autoregressive LM**: There is no direct BPC equivalent for diffusion.

- `val_loss`: Mean denoising MSE over the validation set. This is the ELBO-like training objective.
  Lower is better. Comparable across runs on the same architecture.
- `val_bpd`: `val_loss / ln(2)` — rough conversion to bits-per-dim.
  NOT the same as AR BPC; cannot compare across LM and diffusion models.

The primary tracking metric for the autoresearch loop is `val_loss`.
We also store `val_bpd` for legibility.

**Note**: A proper BPD estimate would require evaluating the full ELBO
(`sum of E[log p(x_t | x_{t+1})]` over all T timesteps), which is too expensive
for a 5-minute loop. We use the one-step denoising loss as a proxy.

---

## 10. Self-Conditioning: Disabled (baseline)

Self-conditioning (Chin-Yew Lin, 2022) — concatenating the previous denoised prediction
as an extra input — improves sample quality at ~25% compute cost.

**Choice**: Disabled in baseline. Can enable with `self_condition=True` in `Unet1D`.

---

## 11. Auto-normalize: True

`GaussianDiffusion1D(auto_normalize=True)` scales inputs from `[0, 1]` to `[-1, 1]`.
The embedding output has arbitrary scale, so we rely on the embedding initialization
and let the diffusion process's internal normalization handle the rest.

---

## 12. Dataset and Data Path

Reuses the existing `enwik8.gz` at `./x-transformers/data/enwik8.gz`.
No new data download required. Same 90M train / 5M val split.

---

## File Structure

```
train_diffusion.py          # DDPM training script (edit this for experiments)
train.py                    # Original AR LM script (unchanged)
x-DDPM/                     # x-DDPM library (read-only, gitignored)
docs/design.md              # This file
docs/adjustable_params.md   # x-DDPM parameter reference
program_diffusion.md        # Agent instructions for DDPM loop
results_diffusion.tsv       # Experiment log (gitignored, created at runtime)
```

---

## Empirical Results (Test Runs)

| Run | Config | val_loss | peak_vram | note |
|-----|--------|----------|-----------|------|
| Baseline | dim=64, mults=(1,2,4), pred_v | 0.009394 | 0.2 GB | |
| Larger model | dim=128, mults=(1,2,4,8), pred_v | 0.004723 | 1.4 GB | significantly better |
| pred_noise | dim=64, mults=(1,2,4), pred_noise | 0.014233 | 0.2 GB | worse than pred_v |

Key findings:
- VRAM usage is extremely low (0.2-1.4 GB) — can scale up aggressively.
- `pred_v` substantially outperforms `pred_noise` (as expected for cosine schedule).
- Bigger model (57M, dim=128, 4 levels) achieves 2x lower val_loss with only 7x more VRAM.
- ~10K steps/5min for small model, ~5K steps/5min for 57M model.

## Experiment Ideas (Priority Order)

1. **Baseline**: dim=64, mults=(1,2,4), pred_v, seq=128 — val_loss=0.009394 (established)
2. **Larger model**: dim=128, mults=(1,2,4,8) — val_loss=0.004723 ✓ (better)
3. **Even larger**: dim=256, mults=(1,2,4,8) with BATCH_SIZE=32 (VRAM allows it)
4. **Sequence length**: Try `SEQ_LEN=256` with `BATCH_SIZE=32`
5. **Batch size**: Try `BATCH_SIZE=128` (more throughput per step)
6. **Self-conditioning**: Enable `self_condition=True` in `Unet1D` (~25% more compute)
7. **EMB_DIM=64**: Richer byte representation
8. **LR tuning**: Try `LEARNING_RATE=3e-4` or `5e-5`
9. **Timesteps**: Try `TIMESTEPS=500` (more gradient steps per budget)
10. **pred_x0**: Compare vs pred_v
