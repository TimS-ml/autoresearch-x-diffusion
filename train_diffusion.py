"""
Autoresearch training script using x-DDPM (denoising diffusion, 1D).
Single-GPU, single-file, time-budgeted training on enwik8.

Hardware: Any NVIDIA GPU with >= 8 GB VRAM (tested on RTX 4090).
Precision: BF16 (default)

Design:
    Bytes (0-255) are embedded into continuous 32-dim vectors via a learned embedding.
    The Unet1D operates on (batch, EMB_DIM, SEQ_LEN) shaped tensors.
    GaussianDiffusion1D handles the forward/reverse process.
    Metric: val_loss (MSE denoising loss, lower is better).
    Secondary: val_bpd = val_loss / ln(2)  (bits per dim, for rough comparison).
    Note: bpd here is NOT the same as autoregressive BPC — it is the denoising loss
    converted to bits. Lower is better in both cases.

Usage:
    python train_diffusion.py           # BF16 (default)
    USE_FP16=1 python train_diffusion.py

References:
    - x-DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch
    - See docs/adjustable_params.md for full parameter reference
    - See docs/design.md for design decisions
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import math
import time
import gzip
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# x-DDPM (from local ./x-DDPM submodule)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "x-DDPM"))

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

# ---------------------------------------------------------------------------
# Precision selection: FP16 or BF16 (default)
# ---------------------------------------------------------------------------

USE_FP16 = os.environ.get("USE_FP16", "0") == "1"

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # training time budget in seconds (5 minutes)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Data
SEQ_LEN = 128  # bytes per sample (sequence length for diffusion)
VOCAB_SIZE = 256  # byte-level vocabulary

# Embedding: continuous representation of discrete bytes
EMB_DIM = 32  # embedding dimension for each byte token

# Unet1D architecture
UNET_DIM = 64  # base channel dimension
UNET_DIM_MULTS = (1, 2, 4)  # channel multipliers (3 levels)

# Diffusion process
TIMESTEPS = 1000  # diffusion timesteps (T)
SAMPLING_TIMESTEPS = 50  # DDIM sampling steps (for fast generation)
OBJECTIVE = "pred_v"  # 'pred_noise', 'pred_x0', 'pred_v'
BETA_SCHEDULE = "cosine"  # 'linear' or 'cosine'

# Optimization
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Logging
VALIDATE_EVERY = 100  # validation frequency (in steps)
GENERATE_EVERY = 500  # generation frequency (in steps)
NUM_EVAL_BATCHES = 50  # batches for validation

# GPU performance reference for MFU estimation
GPU_BF16_PEAK_FLOPS = 330e12  # RTX 4090

# ---------------------------------------------------------------------------
# Data: enwik8 (character-level)
# ---------------------------------------------------------------------------


def _find_enwik8():
    """Search for enwik8.gz in several candidate locations."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "x-transformers", "data", "enwik8.gz"),
        os.path.join(base, "..", "x-transformers", "data", "enwik8.gz"),
        os.path.join(base, "..", "..", "x-transformers", "data", "enwik8.gz"),
        os.path.expanduser("~/offline-git/RAG/x-transformers/data/enwik8.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    # Fallback: original path (will raise a clear error)
    return candidates[0]


DATA_PATH = _find_enwik8()


def load_enwik8(data_path=DATA_PATH):
    """Load enwik8 dataset: 90M train, 5M validation."""
    with gzip.open(data_path) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        return torch.from_numpy(train_x), torch.from_numpy(valid_x)


class ByteSequenceDataset(Dataset):
    """Random subsequence sampler from a byte tensor — returns byte token IDs."""

    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        seq = self.data[rand_start : rand_start + self.seq_len].long()
        return seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


def cycle(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class ByteEmbeddingDiffusion(nn.Module):
    """
    Wrapper that embeds discrete bytes into continuous space for diffusion.

    Forward:
        bytes (B, L) -> embed -> (B, EMB_DIM, L) -> diffuse -> denoising loss

    The embedding projects discrete byte tokens into a continuous EMB_DIM-dim space
    where the Unet1D can operate. We normalize embeddings to unit sphere to keep
    the scale compatible with the [-1,1] range expected by GaussianDiffusion1D
    (auto_normalize=False since we handle normalization ourselves).
    """

    def __init__(self, vocab_size, emb_dim, unet, diffusion):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        # Initialize embeddings to unit sphere (helps diffusion stay in-distribution)
        nn.init.normal_(self.emb.weight, 0.0, 0.1)
        self.unet = unet
        self.diffusion = diffusion

    def embed(self, byte_seq):
        """
        Embed byte tokens to continuous space.
        byte_seq: (B, L) long
        returns: (B, EMB_DIM, L) float  (channel-first for Unet1D)
        """
        x = self.emb(byte_seq)  # (B, L, EMB_DIM)
        x = x.transpose(1, 2)  # (B, EMB_DIM, L)
        return x

    def forward(self, byte_seq):
        """
        Compute diffusion denoising loss on a batch of byte sequences.
        byte_seq: (B, L) long
        returns: scalar loss
        """
        x = self.embed(byte_seq)  # (B, EMB_DIM, L) in [-~1, ~1]
        loss = self.diffusion(x)
        return loss

    @torch.no_grad()
    def sample(self, batch_size=4):
        """Sample sequences: runs reverse diffusion then finds nearest embedding."""
        # Sample in embedding space
        x = self.diffusion.sample(batch_size=batch_size)  # (B, EMB_DIM, L)
        x = x.transpose(1, 2)  # (B, L, EMB_DIM)

        # Nearest-neighbor decode: find closest embedding vector for each position
        W = self.emb.weight  # (vocab_size, emb_dim)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)  # (B*L, D)
        # Cosine similarity for nearest neighbor
        x_norm = F.normalize(x_flat, dim=-1)
        W_norm = F.normalize(W, dim=-1)
        sim = x_norm @ W_norm.T  # (B*L, vocab_size)
        token_ids = sim.argmax(dim=-1).reshape(B, L)  # (B, L)
        return token_ids


def build_model():
    """Build the byte-embedding diffusion model."""
    unet = Unet1D(
        dim=UNET_DIM,
        dim_mults=UNET_DIM_MULTS,
        channels=EMB_DIM,  # input channels = embedding dim
        self_condition=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
    )

    diffusion = GaussianDiffusion1D(
        model=unet,
        seq_length=SEQ_LEN,
        timesteps=TIMESTEPS,
        sampling_timesteps=SAMPLING_TIMESTEPS,
        objective=OBJECTIVE,
        beta_schedule=BETA_SCHEDULE,
        auto_normalize=True,  # normalize input to [-1, 1]
        channel_first=True,
    )

    model = ByteEmbeddingDiffusion(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        unet=unet,
        diffusion=diffusion,
    )
    return model


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def build_optimizer(model):
    """Build AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )


# ---------------------------------------------------------------------------
# LR Schedule (time-based, matching Karpathy's autoresearch pattern)
# ---------------------------------------------------------------------------

WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.40
FINAL_LR_FRAC = 0.1


def get_lr_multiplier(progress):
    """Time-based LR schedule: warmup -> constant -> cosine cooldown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Precision context manager
# ---------------------------------------------------------------------------


def get_precision_context():
    """Return the appropriate AMP context manager."""
    if USE_FP16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Evaluation: Denoising loss (lower is better)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, precision_ctx, num_eval_batches=NUM_EVAL_BATCHES):
    """
    Compute mean denoising loss on validation set.
    val_loss: MSE-based denoising loss (lower = better model)
    val_bpd:  val_loss / ln(2)  (rough bits-per-dim conversion)
    """
    model.eval()
    total_loss = 0.0

    for _ in range(num_eval_batches):
        data = next(val_loader)
        with precision_ctx:
            loss = model(data)
        total_loss += loss.item()

    avg_loss = total_loss / num_eval_batches
    bpd = avg_loss / math.log(2)
    return avg_loss, bpd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    precision_tag = "FP16" if USE_FP16 else "BF16"
    print(f"Precision: {precision_tag}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    print("Loading enwik8...")
    data_train, data_val = load_enwik8()
    train_dataset = ByteSequenceDataset(data_train, SEQ_LEN)
    val_dataset = ByteSequenceDataset(data_val, SEQ_LEN)
    train_loader = cycle(
        DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    )
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True))

    # Model
    print(
        f"Building model: unet_dim={UNET_DIM}, dim_mults={UNET_DIM_MULTS}, "
        f"emb_dim={EMB_DIM}, seq_len={SEQ_LEN}, T={TIMESTEPS}, obj={OBJECTIVE}"
    )
    model = build_model()
    model.cuda()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = build_optimizer(model)
    initial_lr = LEARNING_RATE

    # Compile model
    try:
        model = torch.compile(model, dynamic=False)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: failed ({e}), running eager mode")

    effective_batch_tokens = BATCH_SIZE * SEQ_LEN * GRADIENT_ACCUMULATE_EVERY
    print(
        f"Effective batch: {BATCH_SIZE} x {SEQ_LEN} x {GRADIENT_ACCUMULATE_EVERY} = {effective_batch_tokens:,} tokens"
    )
    print(f"Time budget: {TIME_BUDGET}s")

    # Training loop
    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    precision_ctx = get_precision_context()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        model.train()

        # Gradient accumulation
        accumulated_loss = 0.0
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            with precision_ctx:
                loss = model(next(train_loader))
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()
            accumulated_loss += loss.item()

        train_loss = accumulated_loss / GRADIENT_ACCUMULATE_EVERY

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR schedule
        progress = (
            min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
        )
        lrm = get_lr_multiplier(progress)
        current_lr = initial_lr * lrm
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Fast fail
        if train_loss > 100 or math.isnan(train_loss):
            print("\nFAIL: loss exploded or NaN")
            sys.exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Don't count first 5 warmup steps (compilation overhead)
        if step > 5:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(effective_batch_tokens / dt) if dt > 0 else 0
        remaining = max(0, TIME_BUDGET - total_training_time)
        bpd = debiased_loss / math.log(2)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.6f} | bpd: {bpd:.4f} | lr: {current_lr:.2e} | dt: {dt * 1000:.0f}ms | tok/s: {tok_per_sec:,} | remain: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        # Validation
        if step % VALIDATE_EVERY == 0 and step > 0:
            with precision_ctx:
                val_loss, val_bpd = evaluate(model, val_loader, precision_ctx)
            print(f"\n  [val] loss: {val_loss:.6f}  bpd: {val_bpd:.4f}")

        # Generation (nearest-neighbor decode to bytes)
        if step % GENERATE_EVERY == 0 and step > 0:
            model.eval()
            try:
                with precision_ctx:
                    # Use the underlying module if compiled
                    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                    token_ids = raw.sample(batch_size=1)
                generated = decode_tokens(token_ids[0].tolist())
                print(f"\n  [gen] {generated[:120]!r}")
            except Exception as e:
                print(f"\n  [gen] failed: {e}")

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r

    total_tokens = step * effective_batch_tokens

    # Final eval
    model.eval()
    with precision_ctx:
        val_loss, val_bpd = evaluate(
            model, val_loader, precision_ctx, num_eval_batches=100
        )

    # Final summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_bpd:          {val_bpd:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"seq_len:          {SEQ_LEN}")
    print(f"unet_dim:         {UNET_DIM}")
    print(f"emb_dim:          {EMB_DIM}")
    print(f"timesteps:        {TIMESTEPS}")
    print(f"objective:        {OBJECTIVE}")
    print(f"precision:        {precision_tag}")


if __name__ == "__main__":
    main()
