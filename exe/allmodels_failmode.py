"""바닥 충돌 이후 실패 양상 지표 (한 궤적, 여러 롤아웃 덤프) -> json.

충돌 프레임 c = 정답에서 입자가 바닥 법선 거리 --thr(0.01) 안에 처음 들어온 프레임 (L000 은 처음부터 닿아 있어 0; 계산값도 기록).
바닥 = 씬 cfg 의 surface_collider (point, normal) -- stick_diag.py 와 같은 정의.
프레임 t = 1..L (덤프 pred[t-1], 프레임 0 = 정답 처음 상태 x0). 프레임별:
  mv      : 입자 평균 이동량 mean_i |x_t − x_{t−1}| (모델, 정답 각각) -> 비 mv_ratio
  com     : 입자 평균 위치(무게중심; 입자 질량 같다고 봄) 이동 |c_t − c_{t−1}| -> 비 com_ratio
  smax50/99 : 입자별 max_i|σ_i−1| 의 중앙·99 분위 (덤프 F float32, 특이값은 float64)
  cond99  : σ1/σ3 의 99 분위
  ndet    : det F ≤ 0 입자 수
  psi     : 입자 ψ 평균 (phys_resid.psi_of, cfg = lbpt .pt 의 cfg; psi_color_lb 와 같은 식), 정답은 lbdata h5 의 float32 F 로 같은 식
  err     : 위치 오차 mean_i |x_t − gt_t| / EXT (train_deform 롤아웃 평균과 같은 정의 -- 평균을 로그 값과 대조)
  nonfin  : 위치·F 의 비유한 값 개수
정답 자체의 smax/cond/ndet 도 같은 식으로 (참고용).

  python exe/allmodels_failmode.py --traj 0000 --gt_dir lbdata/test/0000 --pt lbpt/test/0000.pt --floor_cfg lbpool/cfg/lego_jelly.json \
      --dump 이름=roll.pt ... --out fm_0000.json
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import phys_resid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--gt_dir", required=True)
ap.add_argument("--pt", required=True)
ap.add_argument("--floor_cfg", required=True)
ap.add_argument("--dump", action="append", required=True, help="이름=경로 (AF_ROLL_DUMP_F=32)")
ap.add_argument("--thr", type=float, default=0.01)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
torch.set_grad_enabled(False)

Pt = torch.load(a.pt, map_location="cpu", weights_only=False)
cfg = Pt["cfg"]; sel = np.asarray(Pt["sel"]).astype(np.int64)
del Pt
assert phys_resid.mat_name(cfg) == "jelly" and not cfg.get("additional_material_params")
DT = float(cfg.get("frame_dt", 1 / 60))
bcs = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"]
fc = json.load(open(a.floor_cfg))
bcf = [b for b in fc.get("boundary_conditions", []) if b.get("type") == "surface_collider"][0]
bc = bcs[0] if bcs else bcf
P0 = torch.tensor(bc["point"], dtype=torch.float64); NR = torch.tensor(bc["normal"], dtype=torch.float64); NR = NR / NR.norm()
print(f"[바닥] point {P0.tolist()} normal {NR.tolist()} (출처 {'lbpt cfg' if bcs else a.floor_cfg}; 풀 cfg {bcf['point']} {bcf['normal']}) "
      f"접촉 d < {a.thr}", flush=True)


def fstats(F):
    """F [N,3,3] -> smax [N], cond [N], det [N], psi [N] (모두 GPU), nonfinite 수."""
    F = torch.as_tensor(F).to(dev).float().reshape(-1, 3, 3)
    n = F.shape[0]
    sm, cd, dt, ps = (torch.empty(n, device=dev) for _ in range(4))
    nf = int((~torch.isfinite(F)).any(-1).any(-1).sum())
    Fz = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    for s in range(0, n, 65536):
        f = Fz[s:s + 65536]
        sv = torch.linalg.svdvals(f.double())
        sm[s:s + 65536] = (sv - 1.0).abs().amax(-1).float()
        cd[s:s + 65536] = (sv[:, 0] / sv[:, 2].clamp_min(1e-30)).float()
        dt[s:s + 65536] = torch.linalg.det(f.double()).float()
        ps[s:s + 65536] = phys_resid.psi_of(f, cfg, DT)[0]
    return sm, cd, dt, ps, nf


def q(x, p):
    x = x[torch.isfinite(x)]
    return float(torch.quantile(x[torch.randperm(x.numel(), device=x.device)[:2_000_000]].double(), p / 100)) if x.numel() else float("nan")


# ---- 정답 ----
fs = sorted(glob.glob(f"{a.gt_dir}/sim_*.h5"))
D0 = torch.load(a.dump[0].split("=", 1)[1], map_location="cpu", weights_only=False)
L = D0["gt"].shape[0]; N = D0["gt"].shape[1]; EXT = float(D0["EXT"])
x0 = D0["x0"].double(); GTX = torch.cat([x0[None], D0["gt"].double()], 0)        # [L+1,N,3]
del D0
assert len(fs) >= L + 1, (len(fs), L)
assert np.array_equal(sel, np.arange(N)), "sel 이 항등이 아니다"
dgt = ((GTX - P0) @ NR)                                                          # [L+1,N]
dmin = dgt.min(1).values
hit = (dmin < a.thr).nonzero()
c_calc = int(hit[0]) if hit.numel() else None
c = 0 if a.traj == "L000" else c_calc
print(f"[충돌 프레임] 계산 {c_calc} -> 사용 c={c} (L000 은 0 으로 둔다); 프레임별 정답 최소 바닥 거리 처음 몇 개 "
      f"{[round(float(v), 4) for v in dmin[:8]]}", flush=True)


def motion(X):
    mv = (X[1:] - X[:-1]).norm(dim=-1).mean(-1)                                  # [L]
    cm = X.mean(1); com = (cm[1:] - cm[:-1]).norm(dim=-1)
    return mv, com


mv_g, com_g = motion(GTX)
gt = dict(mv=mv_g.tolist(), com=com_g.tolist(), dmin=dmin.tolist(), smax50=[], smax99=[], cond99=[], ndet=[], psi=[])
for i, f in enumerate(tqdm(fs[1:L + 1], desc="정답 F")):
    with h5py.File(f) as h:
        Fm = np.array(h["F"], dtype=np.float32)[sel]
    sm, cd, dt, ps, nf = fstats(Fm)
    gt["smax50"].append(q(sm, 50)); gt["smax99"].append(q(sm, 99)); gt["cond99"].append(q(cd, 99))
    gt["ndet"].append(int((dt <= 0).sum())); gt["psi"].append(float(ps.mean()))
print(f"[정답] 이동량 평균 {float(mv_g.mean()):.3e}, smax99 최대 {max(gt['smax99']):.3f}, cond99 최대 {max(gt['cond99']):.3f}, "
      f"det≤0 최대 {max(gt['ndet'])}, ψ 평균 {np.mean(gt['psi']):.3e}", flush=True)

runs = {}
for d in a.dump:
    nm, p = d.split("=", 1)
    D = torch.load(p, map_location="cpu", weights_only=False)
    assert D.get("F") is not None and D["F"].dtype == torch.float32, f"{p}: float32 F 없음"
    assert D["F"].shape[:2] == (L, N) and D["pred"].shape[:2] == (L, N), (D["F"].shape, L, N)
    gtd = float((D["gt"].double() - GTX[1:]).abs().max())
    X = torch.cat([D["x0"].double()[None], D["pred"].double()], 0)
    nfx = (~torch.isfinite(X)).any(-1).sum(-1)                                   # [L+1]
    Xz = torch.nan_to_num(X)
    mv, com = motion(Xz)
    err = ((Xz[1:] - GTX[1:]).norm(dim=-1).mean(-1) / EXT)
    r = dict(path=p, gt_maxdiff=gtd, x0_maxdiff=float((D["x0"].double() - GTX[0]).abs().max()),
             mv=mv.tolist(), com=com.tolist(), err=err.tolist(), smax50=[], smax99=[], cond99=[], ndet=[], psi=[], nonfin=[],
             ndet_fscal=[])
    fsc = D.get("fscal")
    for i in tqdm(range(L), desc=nm):
        sm, cd, dt, ps, nf = fstats(D["F"][i])
        r["smax50"].append(q(sm, 50)); r["smax99"].append(q(sm, 99)); r["cond99"].append(q(cd, 99))
        r["ndet"].append(int((dt <= 0).sum())); r["psi"].append(float(ps[torch.isfinite(ps)].mean()) if torch.isfinite(ps).any() else float("nan"))
        r["nonfin"].append(nf + int(nfx[i + 1]) + int((~torch.isfinite(ps)).sum()))
        if fsc is not None:
            r["ndet_fscal"].append(int((fsc[i][:, 0].float() <= 0).sum()))
    del D
    runs[nm] = r
    print(f"[{nm}] 오차 평균 {100 * float(err.mean()):.3f}%  정답 동일성 최대차 {gtd:.1e}  smax99 최대 {max(r['smax99']):.3f}  "
          f"cond99 최대 {max(r['cond99']):.3f}  det≤0 최대 {max(r['ndet'])} (덤프 fscal {max(r['ndet_fscal']) if r['ndet_fscal'] else '-'})  "
          f"ψ/정답 최대 {max(x / max(y, 1e-12) for x, y in zip(r['psi'], gt['psi'])):.3e}  비유한 {sum(r['nonfin'])}", flush=True)
json.dump(dict(traj=a.traj, gt_dir=a.gt_dir, pt=a.pt, N=N, L=L, EXT=EXT, thr=a.thr, c=c, c_calc=c_calc,
               floor=dict(point=P0.tolist(), normal=NR.tolist()), gt=gt, runs=runs), open(a.out, "w"), ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
