# autoresearch (x-DDPM edition)

Autonomous LLM-driven research on denoising diffusion models for enwik8 byte sequences,
using [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`).
   The branch `autoresearch-diff/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-diff/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AGENTS.md` — **machine-specific overrides** (Python path, GPU VRAM, etc.). Read first.
   - `train_diffusion.py` — the file you modify. All config is in this file.
   - `docs/adjustable_params.md` — x-DDPM parameter reference.
   - `docs/design.md` — design decisions and rationale.
4. **Verify data exists**: Check that `./x-transformers/data/enwik8.gz` exists.
5. **Verify dependencies**: Run:
   ```bash
   python -c "from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D; print('OK')"
   ```
   If missing, install:
   ```bash
   pip install denoising-diffusion-pytorch einops accelerate ema-pytorch tqdm
   ```
6. **Initialize results_diffusion.tsv**: Create with just the header row.
7. **Confirm and go**.

---

## Experimentation

Each experiment runs on a single GPU for a **fixed 5-minute time budget**.

```bash
python train_diffusion.py > run_diff.log 2>&1
```

**What you CAN do:**
- Modify `train_diffusion.py` — this is the only file you edit.
  Everything is fair game: model dimensions, diffusion parameters, optimizer,
  batch size, embedding dim, sequence length.

**What you CANNOT do:**
- Modify files inside `x-DDPM/`. The library is read-only reference.
- Break the output format (the `---` summary block at the end must remain parseable).

**Goal**: Minimize `val_loss` (denoising MSE). Lower = better denoising = better model.
`val_bpd = val_loss / ln(2)` is logged as a secondary metric.

**VRAM** is a hard constraint. OOM = crash.

**Simplicity criterion**: Same as the AR loop. A small improvement from deleting code
is better than a small improvement that adds complexity.

**First run**: Establish baseline with the default config as-is.

---

## Output Format

```
---
val_loss:         0.009394
val_bpd:          0.013552
training_seconds: 300.0
total_seconds:    342.0
peak_vram_mb:     227.8
total_tokens_M:   81.2
num_steps:        9909
num_params_M:     4.6
seq_len:          128
emb_dim:          32
unet_dim:         64
dim_mults:        (1, 2, 4)
batch_size:       64
grad_accum:       1
lr:               0.0001
timesteps:        1000
objective:        pred_v
beta_schedule:    cosine
self_condition:   False
dropout:          0.0
precision:        BF16
```

Extract the key metric:
```bash
grep "^val_loss:" run_diff.log
```

---

## Logging Results

Log to `results_diffusion.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	val_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss (e.g. 0.123456) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

Example:
```
commit	val_loss	memory_gb	status	description
a1b2c3d	0.123456	4.0	keep	baseline (dim=64 emb=32 seq=128 pred_v)
b2c3d4e	0.118000	4.1	keep	unet_dim=128 batch=32
c3d4e5f	0.130000	4.0	discard	pred_noise worse than pred_v
d4e5f6g	0.000000	0.0	crash	emb_dim=128 OOM
```

---

## The Experiment Loop

LOOP FOREVER on the dedicated branch:

1. Check git state (branch/commit).
2. Modify `train_diffusion.py` with an experimental idea.
3. `git commit`
4. Run: `python train_diffusion.py > run_diff.log 2>&1`
5. Check results: `grep "^val_loss:\|^peak_vram_mb:" run_diff.log`
6. If empty → crash. Check `tail -n 50 run_diff.log` for stack trace.
7. Log to `results_diffusion.tsv`.
8. If val_loss improved (lower) → keep commit, advance branch.
9. If not improved → `git reset --hard HEAD~1` (discard commit).

**Timeout**: If a run exceeds 10 minutes, kill it. Treat as failure.

**NEVER STOP**: Run indefinitely until the human interrupts you.
The human might be away. Keep experimenting.

---

## Experiment Ideas

See `docs/adjustable_params.md` for full details. Priority order:

1. Baseline (as-is)
2. Tune `LEARNING_RATE`: try `3e-4`, `5e-5`
3. `BATCH_SIZE=128` (more throughput per step)
4. `UNET_DIM=128` with `BATCH_SIZE=32`
5. `self_condition=True` in Unet1D
6. `UNET_DIM_MULTS=(1,2,4,8)` with `SEQ_LEN=128`
7. `EMB_DIM=64`
8. `SEQ_LEN=256` with `BATCH_SIZE=32`
9. `OBJECTIVE='pred_noise'` vs `pred_v`
10. `TIMESTEPS=500` (more steps in budget)
11. `learned_sinusoidal_cond=True`
12. `dropout=0.1` in ResNet blocks
