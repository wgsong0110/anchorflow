"""Lossless, crash-safe checkpointing for resumable training.

Design goals (instance can be preempted / OOM-killed at any moment):

- **Atomic writes**: never leave a half-written checkpoint. Write to a temp file
  on the same filesystem, fsync, then ``os.replace`` (atomic rename) onto the
  target. A crash mid-write leaves the *previous* good checkpoint intact.
- **Lossless resume**: persist everything needed to continue bit-for-bit — model,
  optimizer, LR scheduler, all free parameters (actuation latents / node params),
  the global step, best metric, config, AND the RNG states (torch/cuda/numpy/py)
  so noise injection and any sampling continue deterministically.
- **Signal-safe**: register SIGTERM/SIGINT handlers so a graceful preemption
  flushes a final checkpoint before exit.

Usage:
    ckpt = CheckpointManager(outdir)
    start_step = 0
    state = ckpt.load()                       # None on a fresh run
    if state is not None:
        start_step = restore(state)           # your restore fn
    ckpt.install_signal_handler(lambda: ckpt.save(step, collect(), metric))
    ...
    ckpt.save(step, collect_state(), metric=mse, is_best=(mse < best))
"""

from __future__ import annotations

import glob
import os
import signal
import random

import numpy as np
import torch


LAST = "ckpt_last.pt"
BEST = "ckpt_best.pt"


def rng_state():
    """Snapshot every RNG so resume is deterministic."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def load_rng_state(s):
    if s is None:
        return
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    if s.get("numpy") is not None:
        np.random.set_state(s["numpy"])
    if s.get("python") is not None:
        random.setstate(s["python"])


def _atomic_save(obj, path):
    """Write obj to path atomically (temp + fsync + os.replace)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)                      # atomic on POSIX same-fs


class CheckpointManager:
    def __init__(self, outdir, keep_last=3):
        self.outdir = outdir
        self.keep_last = keep_last
        os.makedirs(outdir, exist_ok=True)
        self._installed = False

    # --- paths ------------------------------------------------------------ #
    def _p(self, name):
        return os.path.join(self.outdir, name)

    def latest_path(self):
        p = self._p(LAST)
        if os.path.exists(p):
            return p
        # fall back to the highest-step rolling checkpoint if LAST is missing
        rolls = sorted(glob.glob(self._p("ckpt_step*.pt")),
                       key=lambda x: int(x.split("ckpt_step")[-1].split(".pt")[0]))
        return rolls[-1] if rolls else None

    # --- save / load ------------------------------------------------------ #
    def save(self, step, state: dict, metric=None, is_best=False, rolling=True):
        """state: dict of everything to restore. RNG + step are added here."""
        payload = dict(state)
        payload["step"] = step
        payload["metric"] = metric
        payload["rng"] = rng_state()
        _atomic_save(payload, self._p(LAST))            # always update LAST
        if rolling:
            _atomic_save(payload, self._p(f"ckpt_step{step}.pt"))
            self._prune_rolling()
        if is_best:
            _atomic_save(payload, self._p(BEST))
        return self._p(LAST)

    def _prune_rolling(self):
        rolls = sorted(glob.glob(self._p("ckpt_step*.pt")),
                       key=lambda x: int(x.split("ckpt_step")[-1].split(".pt")[0]))
        for old in rolls[: -self.keep_last] if self.keep_last > 0 else []:
            try:
                os.remove(old)
            except OSError:
                pass

    def load(self, map_location="cpu"):
        p = self.latest_path()
        if p is None:
            return None
        return torch.load(p, map_location=map_location, weights_only=False)

    # --- graceful preemption --------------------------------------------- #
    def install_signal_handler(self, flush_fn):
        """Call flush_fn() on SIGTERM/SIGINT, then re-raise default behaviour."""
        if self._installed:
            return

        def handler(signum, frame):
            try:
                flush_fn()
                print(f"[checkpoint] flushed on signal {signum}")
            finally:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handler)
        self._installed = True
