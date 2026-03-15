Autonomous LLM-driven research on denoising diffusion models for enwik8 byte sequences, using my fork of [x-DDPM (denoising-diffusion-pytorch)](https://github.com/TimS-ml/x-DDPM). Runs on any NVIDIA GPU with >= 8 GB VRAM.

![teaser](progress.png)

![memory_usage](memory.png)

# autoresearch (x-DDPM edition)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real diffusion model training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. This edition uses Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) library, applying 1D diffusion to learn continuous embeddings of enwik8 byte sequences. The core idea is the same as [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — you program the `program.md` Markdown file that provides context to the AI agent.

## How it works

The repo has a few key files:

- **`train.py`** — the single file the agent edits. Contains the byte embedding, Unet1D model, GaussianDiffusion1D, optimizer (AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`AGENTS.md`** — detailed experiment protocol including hardware specs, parameter space guide, and experiment ideas.
- **`docs/adjustable_params.md`** — comprehensive reference of all adjustable x-DDPM parameters with descriptions.
- **`docs/design.md`** — design decisions and rationale.
- **`x-DDPM/`** — the denoising-diffusion-pytorch library (git submodule, read-only reference).

Dataset: **enwik8** (character-level, 256 vocab, 90M train / 5M val). Metric: **val_loss** (denoising MSE) — lower is better.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of your hardware. This makes experiments directly comparable.

## Quick start

**Requirements:** A single NVIDIA GPU (>= 8 GB VRAM), Python 3.10+. Any package manager works (uv / conda / mamba / pip).

```bash
# 1. Clone the repo
git clone --recursive https://github.com/TimS-ml/x-DDPM
cd x-DDPM

# 2. Install dependencies (pick one)
uv sync                  # uv (recommended)
# or: pip install -e .   # pip / conda / mamba

# 3. Verify data exists
ls x-transformers/data/enwik8.gz

# 4. Verify imports work
python -c "from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D; print('OK')"

# 5. Run a single training experiment (~5 min)
python train.py                  # BF16 (default)
USE_FP16=1 python train.py      # FP16
```

## Running the agent

Spin up your Claude/Codex or whatever you want in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the "skill" that drives the autonomous agent. `AGENTS.md` provides the full protocol.

## Project structure

```
train.py                — model, optimizer, training loop (agent modifies this)
program.md              — agent instructions
docs/
  adjustable_params.md  — x-DDPM parameter reference
  design.md             — design decisions and rationale
x-DDPM/                 — denoising-diffusion-pytorch library (git submodule, read-only)
x-transformers/         — data source (enwik8.gz lives here)
```

## Design choices

- **x-DDPM as the model backbone.** Phil Wang's denoising-diffusion-pytorch library provides a clean Unet1D + GaussianDiffusion1D implementation. The agent only touches `train.py` to configure model and diffusion parameters.
- **Learned byte embeddings.** Discrete bytes (0-255) are embedded into continuous 32-dim vectors. The Unet1D diffuses in this continuous space. Decoding uses nearest-neighbor cosine similarity back to the embedding table.
- **Character-level enwik8.** No tokenizer needed. 256-vocab byte-level. Simple and fast.
- **Fixed time budget.** Training always runs for exactly 5 minutes. This makes experiments directly comparable regardless of what the agent changes. ~12 experiments/hour, ~100 overnight.
- **Self-contained.** One GPU, one file, one metric. The x-DDPM submodule provides the model; everything else is standard PyTorch.

## Credits

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the original concept
- [Phil Wang's denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) here is [my fork](https://github.com/TimS-ml/x-DDPM) — the model library

## License

MIT
