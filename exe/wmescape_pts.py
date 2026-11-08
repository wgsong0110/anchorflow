"""수박 입자 날아감 비교 영상 (exe/wmescape_vid.sh) 의 점 그림용 위치 (읽기만, 풀이와 무관).

프레임 0~100 (판은 프레임 0 = 정답 처음 상태, 그 뒤 pred) 마다:
  이탈 = exe/escape1_wrap.py 진단 D1 과 같은 정의 -- 몸 상자 (유한 입자 7 개마다 표본의 0.1~99.9 백분위) 에서 4 칸 넘게 떨어진 입자.
  저장: 고정 표본 (입자 STRIDE 개마다) 위치 float16, 이탈 입자 전부의 위치 float32 (프레임별 이어 붙임 + 시작 위치), 프레임별 이탈 수·거리 최대·몸 상자.
사용: python exe/wmescape_pts.py --cell 0.02 --gt wm.pt --outdir DIR [--run 이름=pred.pt ...]   (정답은 DIR/gt.npz, 판은 DIR/<이름>.npz)
"""
import argparse
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=float, required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--run", action="append", default=[])
ap.add_argument("--stride", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(p, map_location="cpu", weights_only=False)


X = load(a.gt)["x"]
QQ = torch.tensor([0.001, 0.999], dtype=torch.float64)


def frames_of(get, L, out):
    sub, far_x, far_o, nfar, dmax, box, nonfin = [], [], [0], [], [], [], []
    for f in range(L):
        x = get(f).float()
        ok = torch.isfinite(x).all(-1)
        xs = x[ok][::7].double()
        q = torch.quantile(xs, QQ, dim=0).float()
        d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        far = (d > 4 * a.cell) & ok
        sub.append(x[::a.stride].numpy().astype(np.float16))
        far_x.append(x[far].numpy().astype(np.float32)); far_o.append(far_o[-1] + int(far.sum()))
        nfar.append(int((d > 4 * a.cell).sum())); dmax.append(float(d[ok].max())); box.append(q.numpy()); nonfin.append(int((~ok).sum()))
    np.savez_compressed(out, sub=np.stack(sub), far_x=np.concatenate(far_x), far_o=np.array(far_o), nfar=np.array(nfar), dmax=np.array(dmax),
                        box=np.stack(box), nonfin=np.array(nonfin), cell=a.cell, stride=a.stride)
    print(out, "프레임", L, "이탈 최대", max(nfar), "거리 최대", round(max(dmax), 3), "비유한 최대", max(nonfin), "z 최소", float(np.concatenate([s[:, 2] for s in sub]).min()), flush=True)


frames_of(lambda f: X[f], X.shape[0], os.path.join(a.outdir, "gt.npz"))
for r in a.run:
    n, p = r.split("=", 1)
    P = load(p)["pred"]
    L = P.shape[0]
    frames_of(lambda f: X[0] if f == 0 else P[f - 1], L + 1, os.path.join(a.outdir, n + ".npz"))
