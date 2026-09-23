"""조밀 3D conv 와 희소(spconv) 의 스텝 시간을 같은 조건에서 잰다."""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import vox_anchor
from anchorflow.conv_stepper import ConvStepper, SparseConvStepper

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250000)
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--feat", type=int, default=75)
ap.add_argument("--rep", type=int, default=20)
a = ap.parse_args()
dev = "cuda:0"
torch.manual_seed(0)
# 물체가 bbox 일부만 채우도록 구 안에 점을 둔다 (실제 씬과 비슷하게)
q = torch.randn(a.n, 3, device=dev)
q = q / q.norm(dim=1, keepdim=True) * torch.rand(a.n, 1, device=dev) ** (1/3)
x = q * 0.4 + 0.5

lo, h, n3 = vox_anchor.grid_for(x, a.res ** 3)
dense_M = int(n3[0] * n3[1] * n3[2])
coords, cen, uniq = vox_anchor.occupied(x, lo, h, n3)
print(f"[격자] {tuple(int(t) for t in n3)} 전체 {dense_M} 칸, 점유 {coords.shape[0]} 칸 "
      f"({100*coords.shape[0]/dense_M:.1f}%)", flush=True)

def timeit(fn, rep):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/rep*1e3

fd = torch.randn(dense_M, a.feat, device=dev)
fs = torch.randn(coords.shape[0], a.feat, device=dev)
grid = tuple(int(t) for t in n3)
for name, arch in (("조밀 conv", "plain"), ("조밀 unet", "unet")):
    net = ConvStepper(a.feat, a.hidden, a.depth, arch=arch).to(dev)
    t = timeit(lambda: net(None, fd, 1/60., grid), a.rep)
    print(f"[{name}] {t:.2f} ms  (칸 {dense_M})", flush=True)
net = SparseConvStepper(a.feat, a.hidden, a.depth).to(dev)
t = timeit(lambda: net(None, fs, 1/60., (coords, grid)), a.rep)
print(f"[희소 점유만] {t:.2f} ms  (칸 {coords.shape[0]})", flush=True)

iu, cu, ceu = vox_anchor.knn_union(x, lo, h, n3, 16)
fu = torch.randn(cu.shape[0], a.feat, device=dev)
t = timeit(lambda: net(None, fu, 1/60., (cu, grid)), a.rep)
print(f"[희소 kNN합집합] {t:.2f} ms  (칸 {cu.shape[0]}, 점유 대비 "
      f"{cu.shape[0]/coords.shape[0]:.2f} 배, 조밀 대비 {100*cu.shape[0]/dense_M:.1f}%)",
      flush=True)
print("CONVBENCH_OK", flush=True)
