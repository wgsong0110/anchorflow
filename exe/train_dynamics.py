#!/usr/bin/env python
"""Train + validate GNN autoregressive anchor dynamics (Priority 1).

Learns a GNS-style graph network to predict per-anchor acceleration, then
free-runs an autoregressive rollout and compares it to ground truth.  The key
success signal is that the *rollout* position error stays low over many steps
and beats a constant-velocity baseline — i.e. the GNN has learned the spatial
coupling between anchors, not just a per-node extrapolation.

Usage
    python exe/train_dynamics.py --cfg cfg/pendulum.yaml
    python exe/train_dynamics.py --seq traveling_wave --steps 3000
Run from the repo root with lib on PYTHONPATH (the script adds it automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import torch                                           # noqa: E402
import torch.nn as nn                                  # noqa: E402

from anchorflow import synth                           # noqa: E402
from anchorflow.dynamics import GNSDynamics, build_graph, rollout   # noqa: E402


# --------------------------------------------------------------------------- #
def load_cfg(args):
    cfg = dict(
        seq="pendulum_chain", seq_kwargs={},
        hidden=128, message_passing_steps=6,
        graph="knn", k=6, radius=0.6, max_neighbors=16, rebuild_graph=True,
        steps=3000, batch=8, lr=1e-3, lr_final=1e-4, noise=0.0003,
        train_frac=0.7, seed=0, eval_every=250, compile=False,
        outdir="/home/wgsong/workspace/result/anchorflow",
    )
    if args.cfg:
        import yaml
        with open(args.cfg) as f:
            cfg.update(yaml.safe_load(f) or {})
    for key in ("seq", "steps", "hidden", "message_passing_steps", "k",
                "graph", "lr", "seed", "noise", "outdir"):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    return cfg


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- #
def make_dataset(cfg, device):
    data = synth.make(cfg["seq"], seed=cfg["seed"], **cfg.get("seq_kwargs", {}))
    pos = torch.as_tensor(data["positions"], device=device)        # [T, N, 3]
    fixed = torch.as_tensor(data["fixed"], device=device)          # [N]
    T = pos.shape[0]
    n_train = int(cfg["train_frac"] * T)
    # valid centre frames t need t-1, t, t+1 -> [1, n_train-2]
    train_t = torch.arange(1, n_train - 1, device=device)
    return pos, fixed, train_t, n_train, data


def prime_normalizers(model, pos, fixed, train_t):
    """One pass to populate input/target normaliser statistics."""
    model.train()
    with torch.no_grad():
        for t in train_t.tolist():
            vel = pos[t] - pos[t - 1]
            acc = pos[t + 1] - 2 * pos[t] + pos[t - 1]
            node_feat = torch.cat([vel, fixed.float().unsqueeze(-1)], dim=-1)
            model.in_norm(node_feat)                   # accumulate
            model.out_norm(acc[~fixed])                # accumulate on movers


def train_step(model, pos, fixed, t_batch, cfg, opt):
    model.train()
    loss_sum = 0.0
    opt.zero_grad()
    for t in t_batch.tolist():
        p_prev, p_cur, p_next = pos[t - 1], pos[t], pos[t + 1]
        # random-walk noise on the current state (GNS robustness trick)
        if cfg["noise"] > 0:
            noise = torch.randn_like(p_cur) * cfg["noise"]
            noise[fixed] = 0.0
            p_cur = p_cur + noise
            p_prev = p_prev + noise                    # keep velocity, shift both
        vel = p_cur - p_prev
        target_acc = p_next - 2 * p_cur + p_prev
        edge_index = build_graph(p_cur, cfg)
        _, acc_norm = model.predict_accel(p_cur, vel, fixed, edge_index)
        target_norm = model.out_norm(target_acc, accumulate=False)
        mask = ~fixed
        loss = ((acc_norm[mask] - target_norm[mask]) ** 2).mean()
        (loss / len(t_batch)).backward()
        loss_sum += loss.item()
    opt.step()
    return loss_sum / len(t_batch)


@torch.no_grad()
def evaluate(model, pos, fixed, n_train, cfg):
    """Full-sequence rollout from frames 0,1; report train/extrapolation MSE."""
    T = pos.shape[0]
    pred = rollout(model, pos[0], pos[1], fixed, steps=T - 2, cfg=cfg,
                   rebuild_graph=cfg["rebuild_graph"])
    err = ((pred - pos) ** 2).mean(dim=(1, 2))         # per-frame MSE [T]
    # constant-velocity baseline rollout
    base = [pos[0], pos[1]]
    for _ in range(T - 2):
        nb = 2 * base[-1] - base[-2]
        nb = nb.clone(); nb[fixed] = pos[0][fixed]
        base.append(nb)
    base = torch.stack(base)
    base_err = ((base - pos) ** 2).mean(dim=(1, 2))
    return dict(
        pred=pred.cpu().numpy(),
        mse_full=float(err.mean()),
        mse_train=float(err[:n_train].mean()),
        mse_extrap=float(err[n_train:].mean()),
        mse_final=float(err[-1]),
        base_full=float(base_err.mean()),
        base_extrap=float(base_err[n_train:].mean()),
        err_curve=err.cpu().numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--seq")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--message_passing_steps", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--graph", choices=["knn", "radius"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--noise", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    torch.manual_seed(cfg["seed"])
    gh = git_hash()

    pos, fixed, train_t, n_train, data = make_dataset(cfg, device)
    print(f"[data] seq={data['name']} pos={tuple(pos.shape)} "
          f"fixed={int(fixed.sum())} train_frames={n_train} commit={gh}")

    model = GNSDynamics(hidden=cfg["hidden"],
                        message_passing_steps=cfg["message_passing_steps"]).to(device)
    prime_normalizers(model, pos, fixed, train_t)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,} "
          f"| accel std={model.out_norm.std().mean().item():.4f}")

    if cfg["compile"]:
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:                          # noqa: BLE001
            print(f"[warn] torch.compile disabled: {e}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gamma = (cfg["lr_final"] / cfg["lr"]) ** (1.0 / max(1, cfg["steps"]))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)

    os.makedirs(cfg["outdir"], exist_ok=True)
    tag = f"{data['name']}_{gh}"
    history = []
    t0 = time.time()
    best = float("inf")
    for step in range(1, cfg["steps"] + 1):
        idx = train_t[torch.randint(len(train_t), (cfg["batch"],))]
        loss = train_step(model, pos, fixed, idx, cfg, opt)
        sched.step()
        if step % cfg["eval_every"] == 0 or step == 1:
            ev = evaluate(model, pos, fixed, n_train, cfg)
            skill = ev["base_extrap"] / max(ev["mse_extrap"], 1e-12)
            print(f"[{step:5d}] loss={loss:.4e} "
                  f"rollout mse full={ev['mse_full']:.3e} "
                  f"extrap={ev['mse_extrap']:.3e} final={ev['mse_final']:.3e} "
                  f"| baseline extrap={ev['base_extrap']:.3e} "
                  f"(x{skill:.1f} better) | {time.time()-t0:.0f}s")
            history.append(dict(step=step, loss=loss,
                                mse_full=ev["mse_full"],
                                mse_extrap=ev["mse_extrap"],
                                base_extrap=ev["base_extrap"]))
            if ev["mse_full"] < best:
                best = ev["mse_full"]
                np.savez(os.path.join(cfg["outdir"], f"rollout_{tag}.npz"),
                         gt=pos.cpu().numpy(), pred=ev["pred"],
                         fixed=fixed.cpu().numpy(), err_curve=ev["err_curve"],
                         n_train=n_train)
                torch.save({"model": (model._orig_mod if hasattr(model, "_orig_mod")
                                      else model).state_dict(), "cfg": cfg},
                           os.path.join(cfg["outdir"], f"model_{tag}.pt"))

    with open(os.path.join(cfg["outdir"], f"history_{tag}.json"), "w") as f:
        json.dump(dict(cfg=cfg, commit=gh, best_mse=best, history=history), f, indent=2)
    print(f"[done] best rollout mse={best:.3e} -> {cfg['outdir']}/*_{tag}.*")


if __name__ == "__main__":
    main()
