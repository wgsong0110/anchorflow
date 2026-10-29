"""절 Z 준비: 정답 입자 F 의 특이값 비율 σ1/σ3 분포 (프레임별 50·99·99.9 % 분위·최대).
  전체 F_t (정답 변형구배) 와 프레임 증분 F_{t+1} F_t⁻¹ (= 한 프레임 동안의 사상, 메시 셀 사상 A_T 와 같은 뜻) 둘 다.
사용: python exe/fem_z_gtsr.py --data <traj.pt> --frames L --out <json>"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--frames", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
h = float(d["cfg"]["frame_dt"])
F = DM.traj_F(d, h, dev)
L = min(a.frames, int(F.shape[0]) - 1)
Q = torch.tensor([0.5, 0.99, 0.999], device=dev, dtype=torch.float64)


def stats(Fm):
    sv = torch.linalg.svdvals(Fm.double())
    det = torch.linalg.det(Fm.double())
    r = sv[:, 0] / sv[:, -1].clamp_min(1e-30)
    q = torch.quantile(r[torch.randperm(r.numel(), device=dev)[:min(r.numel(), 16_000_000)]], Q).tolist()
    return dict(q50=q[0], q99=q[1], q999=q[2], max=float(r.max()), n_inv=int((det <= 0).sum()))


rows = []
for t in range(L + 1):
    Ft = F[t].to(dev).reshape(-1, 3, 3)
    rec = dict(t=t, F=stats(Ft))
    if t < L:
        Fn = F[t + 1].to(dev).reshape(-1, 3, 3)
        inc = Fn @ torch.linalg.inv(Ft.double()).float()
        rec["inc"] = stats(inc)
    rows.append(rec)
    if t % 10 == 0 or t == L:
        print(f"프레임 {t}: F σ1/σ3 중앙 {rec['F']['q50']:.4f} 99% {rec['F']['q99']:.4f} 99.9% {rec['F']['q999']:.4f} 최대 {rec['F']['max']:.4f}"
              + (f" | 증분 중앙 {rec['inc']['q50']:.4f} 99% {rec['inc']['q99']:.4f} 99.9% {rec['inc']['q999']:.4f} 최대 {rec['inc']['max']:.4f}" if "inc" in rec else ""), flush=True)
summ = dict(data=a.data, frames=L, Np=int(F.shape[1]),
            F_q999_max=max(r["F"]["q999"] for r in rows), F_q999_med=sorted(r["F"]["q999"] for r in rows)[len(rows) // 2], F_max=max(r["F"]["max"] for r in rows),
            inc_q999_max=max(r["inc"]["q999"] for r in rows if "inc" in r), inc_q999_med=sorted(r["inc"]["q999"] for r in rows if "inc" in r)[L // 2],
            inc_max=max(r["inc"]["max"] for r in rows if "inc" in r), rows=rows)
json.dump(summ, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"[요약] {os.path.basename(a.data)} {L} 프레임: F 99.9% 분위 최대 {summ['F_q999_max']:.4f} (프레임 중앙 {summ['F_q999_med']:.4f}, 전체 최대 {summ['F_max']:.4f}); "
      f"증분 99.9% 최대 {summ['inc_q999_max']:.4f} (중앙 {summ['inc_q999_med']:.4f}, 전체 최대 {summ['inc_max']:.4f})", flush=True)
