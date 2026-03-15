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

## Experiment Ideas (Priority Order)

1. **Baseline**: Establish baseline `val_loss` with default config.
2. **Batch size**: Try `BATCH_SIZE=32` (larger model) or `BATCH_SIZE=128` (more throughput).
3. **Unet dim**: Try `UNET_DIM=128` (deeper/wider) — watch VRAM.
4. **Objective**: Try `pred_noise` and `pred_x0` vs `pred_v`.
5. **Sequence length**: Try `SEQ_LEN=256` with `BATCH_SIZE=32`.
6. **EMB_DIM**: Try 64 or 16.
7. **Self-conditioning**: Enable `self_condition=True` in `Unet1D`.
8. **dim_mults**: Try `(1, 2, 4, 8)` — adds one more resolution level.
9. **Timesteps**: Try `T=500` (faster per-step, more steps in budget).
10. **DDIM eta**: Try `ddim_sampling_eta=1.0` (DDPM sampling).
