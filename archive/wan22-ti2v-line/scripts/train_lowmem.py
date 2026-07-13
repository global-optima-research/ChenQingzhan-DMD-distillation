#!/usr/bin/env python
"""
train_lowmem.py — Low-memory wrapper for FastGen train.py

Replaces torch.optim.AdamW with bitsandbytes AdamW8bit BEFORE FastGen
initializes optimizers. This saves ~5 GB VRAM by halving optimizer states
(fp32 m+v → int8 quantized).

Usage (drop-in replacement for train.py):
    python train_lowmem.py --config=... - trainer.ddp=False ...

The wrapper is transparent — all arguments are passed through to train.py.
"""

import sys
import os

# === Step 1: Set CUDA allocator BEFORE any torch import ===
# expandable_segments reduces memory fragmentation (can save ~1-4 GB)
# MUST be set before torch initializes CUDA
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    new_conf = "expandable_segments:True"
    if alloc_conf:
        new_conf = f"{alloc_conf},{new_conf}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
    print(f"[lowmem] PYTORCH_CUDA_ALLOC_CONF={new_conf}")

# === Step 2: Monkey-patch torch.optim.AdamW with 8-bit version ===
try:
    import bitsandbytes as bnb
    import torch.optim

    # Wrapper to handle FastGen passing 'fused=True' which bnb doesn't support
    class AdamW8bitCompat(bnb.optim.AdamW8bit):
        """Drop-in replacement that accepts and ignores unsupported kwargs."""
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=1e-2, amsgrad=False, **kwargs):
            # Filter out kwargs not supported by bitsandbytes (e.g. 'fused')
            ignored = {k: v for k, v in kwargs.items()
                       if k in ('fused', 'foreach', 'maximize', 'capturable',
                                'differentiable')}
            if ignored:
                print(f"[lowmem] AdamW8bit: ignoring unsupported kwargs: {ignored}")
            super().__init__(params, lr=lr, betas=betas, eps=eps,
                             weight_decay=weight_decay, amsgrad=amsgrad)

    # Replace with 8-bit version
    torch.optim.AdamW = AdamW8bitCompat

    # Also patch the adamw module in case FastGen imports from there
    try:
        import torch.optim.adamw
        torch.optim.adamw.AdamW = AdamW8bitCompat
    except (ImportError, AttributeError):
        pass

    print(f"[lowmem] torch.optim.AdamW -> bitsandbytes.AdamW8bit (v{bnb.__version__})")
    print(f"[lowmem] Expected VRAM savings: ~5 GB (optimizer states halved)")

except ImportError:
    print("[lowmem] WARNING: bitsandbytes not installed, using standard AdamW")
    print("[lowmem] Install with: pip install bitsandbytes")

# === Step 3: Run original train.py ===
print("[lowmem] Launching FastGen train.py...")
print()

import runpy
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
