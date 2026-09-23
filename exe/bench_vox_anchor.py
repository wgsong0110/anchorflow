"""복셀중심 앵커(탐색 없는 kNN) 와 FPS 앵커(kNN 탐색) 의 속도·집계 결과를 잰다."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import torch
from anchorflow import vox_anchor
from anchorflow.deform import aggregate, anchor_knn, fps

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200000, help="가우시안 수")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--rep", type=int, default=20)
a = ap.parse_args()

dev = "cuda:0"
torch.manual_seed(0)
x = torch.rand(a.n, 3, device=dev)
v = torch.randn(a.n, 3, device=dev) * 0.01
m = torch.rand(a.n, device=dev) * 1e-6
X = x.clone()


def timeit(fn, rep):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3


# --- FPS 앵커: 앵커 선정 + kNN 탐색
def fps_path():
    ai = fps(x, a.n_anchors, 0)
    p = x[ai]
    idx, _ = anchor_knn(x, p, a.k)
    return p, idx


# --- 복셀중심 앵커: 격자 구성 + 계산식 kNN
def vox_path():
    lo, h, n = vox_anchor.grid_for(x, a.n_anchors)
    p = vox_anchor.centers(lo, h, n)
    idx = vox_anchor.knn(x, lo, h, n, a.k)
    return p, idx


p_f, idx_f = fps_path()
p_v, idx_v = vox_path()
H_f = float(torch.cdist(p_f, p_f).topk(2, largest=False).values[:, 1].mean())
H_v = float(torch.cdist(p_v, p_v).topk(2, largest=False).values[:, 1].mean())
print(f"[앵커] FPS {p_f.shape[0]} 개 간격 {H_f:.4f} | 복셀중심 {p_v.shape[0]} 개 "
      f"간격 {H_v:.4f}", flush=True)

t_f = timeit(lambda: fps_path(), a.rep)
t_v = timeit(lambda: vox_path(), a.rep)
print(f"[앵커+kNN] FPS {t_f:.2f} ms | 복셀중심 {t_v:.2f} ms  ({t_f/t_v:.1f} 배)",
      flush=True)

t_af = timeit(lambda: aggregate(x, v, X, m, idx_f, p_f.shape[0], H_f, pa=p_f), a.rep)
t_av = timeit(lambda: aggregate(x, v, X, m, idx_v, p_v.shape[0], H_v, pa=p_v), a.rep)
print(f"[집계] FPS {t_af:.2f} ms | 복셀중심 {t_av:.2f} ms", flush=True)
print(f"[합계] FPS {t_f+t_af:.2f} ms | 복셀중심 {t_v+t_av:.2f} ms "
      f"({(t_f+t_af)/(t_v+t_av):.1f} 배)", flush=True)

fe, _ = aggregate(x, v, X, m, idx_f, p_f.shape[0], H_f, pa=p_f)[:2] if isinstance(
    aggregate(x, v, X, m, idx_f, p_f.shape[0], H_f, pa=p_f), tuple) else (
    aggregate(x, v, X, m, idx_f, p_f.shape[0], H_f, pa=p_f), None)
print("BENCH_OK", flush=True)
