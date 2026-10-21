"""lb 궤적(lbdata h5, float32 F) + 롤아웃 덤프 여러 개의 입자별 탄성 에너지 밀도 psi 를 **같은 함수**로 재고,
전부 합친 고정 범위로 색을 매겨 lb_render --colors 용 .npy 를 만든다. 분포 패널용 히스토그램·변형 통계도 저장.

psi = phys_resid.psi_of(F, cfg) (cfg = lbpt .pt 의 cfg, train_deform 이 쓰는 그대로). psi 는 회전 불변이라
lbdata F (초기 회전 R 이 실린 F_sim·R) 와 덤프 F (F_sim) 를 그대로 비교할 수 있다.
0 프레임은 모든 실행이 같은 상태(정답 0 프레임)에서 출발하므로 정답 0 프레임 값을 쓴다.

색: log10(psi + eps) 를 [lo, hi] (전 실행 1~T-1 프레임 전 입자 합친 분위수, 고정 상수) 에서 잘라 컬러맵 하나로.
히스토그램: 같은 구간, **가중 없음(입자 수 비율)**, 시뮬 입자 전부(채움 포함). 렌더 색은 가우시안(앞 NG 개)만 보인다.

  python exe/psi_color_lb.py --gt_dir W/lbdata/test_low/L000 --pt W/lbpt/test_low/L000.pt \
      --dump nog=W/lbours/pool_gen/roll_out_L000.pt --dump greg=W/lbours/ipcmp/roll_out_L000.pt --out W/lbours/lego_psi
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
ap.add_argument("--gt_dir", required=True, help="lbdata 궤적 디렉토리 (sim_*.h5 의 F (N,9) float32)")
ap.add_argument("--pt", required=True, help="lbpt 변환본 (sel·cfg 만 쓴다; 그 안의 half F 는 쓰지 않는다)")
ap.add_argument("--dump", action="append", required=True, help="이름=경로 (AF_ROLL_DUMP_F=32 덤프)")
ap.add_argument("--out", required=True)
ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--q", type=float, nargs=2, default=[1.0, 99.0])
ap.add_argument("--cmap", default="turbo")
ap.add_argument("--bins", type=int, default=120)
ap.add_argument("--pad", type=float, default=1.0)
ap.add_argument("--range", default="", help="고정 범위 range.json (주면 lo/hi 를 그대로 쓰고 분위수로 다시 정하지 않는다; eps 도 같아야 함)")
ap.add_argument("--stats_name", default="lego_psi_defstats.txt")
ap.add_argument("--stats_frames", type=int, nargs="+", default=[1, 40, 80])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
P = torch.load(a.pt, map_location="cpu", weights_only=False)
cfg = P["cfg"]; sel = np.asarray(P["sel"]).astype(np.int64)
del P
mu, lam = phys_resid.lame(cfg["E"], cfg["nu"])
print(f"[재질] {phys_resid.mat_name(cfg)} E {cfg['E']} nu {cfg['nu']} -> mu {mu:.4g} lam {lam:.4g}; "
      f"영역 물성 {cfg.get('additional_material_params', [])}", flush=True)
assert phys_resid.mat_name(cfg) == "jelly"
assert not cfg.get("additional_material_params"), "영역별 물성이 있다 -- 입자별 mu/lam 필요"


@torch.no_grad()
def stats(F):
    """psi, max_i|σ_i-1|, det F."""
    F = torch.as_tensor(F).to(dev).float().reshape(-1, 3, 3)
    ps, sm, dt = (torch.empty(F.shape[0], device=dev) for _ in range(3))
    for s in range(0, F.shape[0], 65536):
        f = F[s:s + 65536]
        ps[s:s + 65536] = phys_resid.psi_of(f, cfg, float(cfg.get("frame_dt", 1 / 60)))[0]
        sm[s:s + 65536] = (torch.linalg.svdvals(f.double()).float() - 1.0).abs().amax(-1)
        dt[s:s + 65536] = torch.linalg.det(f.double()).float()
    return ps.cpu().numpy(), sm.cpu().numpy(), dt.cpu().numpy()


fs = sorted(glob.glob(f"{a.gt_dir}/sim_*.h5"))
T = len(fs)
with h5py.File(fs[0]) as h:
    NF = h["F"].shape[0]
assert np.array_equal(sel, np.arange(NF)), "sel 이 항등이 아니다 -- 덤프 h5 행 순서와 정답 h5 행 순서가 다르다"
N = len(sel)
RUNS = {"gt": [], **{d.split("=", 1)[0]: [] for d in a.dump}}
SM = {k: {} for k in RUNS}; DT = {k: {} for k in RUNS}
for i, f in enumerate(tqdm(fs, desc="정답 psi")):
    with h5py.File(f) as h:
        Fm = np.array(h["F"], dtype=np.float32)[sel]
    ps, sm, dt = stats(Fm)
    RUNS["gt"].append(ps)
    if i in a.stats_frames:
        SM["gt"][i], DT["gt"][i] = sm, dt
info = {}
for d in a.dump:
    nm, p = d.split("=", 1)
    D = torch.load(p, map_location="cpu", weights_only=False)
    assert D.get("F") is not None and D["F"].dtype == torch.float32, f"{p}: float32 F 없음"
    L = D["F"].shape[0]
    assert D["F"].shape[1] == N and L == T - 1, (D["F"].shape, N, T)
    info[nm] = dict(path=p, frames=L, tag=D.get("tag"), t0=D.get("t0"))
    RUNS[nm].append(RUNS["gt"][0].copy())
    for i in tqdm(range(L), desc=f"{nm} psi"):
        ps, sm, dt = stats(D["F"][i])
        RUNS[nm].append(ps)
        if i + 1 in a.stats_frames:
            SM[nm][i + 1], DT[nm][i + 1] = sm, dt
    if D.get("fscal") is not None:
        fsc = np.asarray(D["fscal"][..., 4].float())
        print(f"[검산] {nm} psi vs 덤프 fscal psi: 최대 차이/최대값 "
              f"{np.abs(fsc - np.stack(RUNS[nm][1:])).max() / max(np.abs(fsc).max(), 1e-12):.2e}", flush=True)
    del D
for k in RUNS:
    RUNS[k] = np.stack(RUNS[k]).astype(np.float32)
    np.save(f"{a.out}/psi_{k}.npy", RUNS[k])
print(f"[psi] {', '.join(f'{k} {v.shape}' for k, v in RUNS.items())}; 0 프레임 정답 최대 {RUNS['gt'][0].max():.3g}", flush=True)

# 변형 통계 표
with open(f"{a.out}/{a.stats_name}", "w") as o:
    o.write(f"# max_i|σ_i-1| (입자별) 중앙값·상위 1% (99 분위수), det F < 0 입자 수 / 전체 {N}; psi 평균·중앙값 [Pa]\n")
    o.write(f"# psi_of: {phys_resid.mat_name(cfg)} FCR, E {cfg['E']} nu {cfg['nu']}\n")
    for k, v in info.items():
        o.write(f"# {k} = {v['path']} ({v['frames']} 프레임)\n")
    o.write(f"{'실행':<6} {'프레임':>5} {'|σ-1|max 중앙':>14} {'|σ-1|max 상위1%':>16} {'det<0':>7} {'ψ 평균':>11} {'ψ 중앙':>11}\n")
    for k in RUNS:
        for t in a.stats_frames:
            if t not in SM[k]:
                continue
            o.write(f"{k:<6} {t:>5} {np.median(SM[k][t]):>14.4e} {np.percentile(SM[k][t], 99):>16.4e} "
                    f"{int((DT[k][t] < 0).sum()):>7d} {RUNS[k][t].mean():>11.4e} {np.median(RUNS[k][t]):>11.4e}\n")
print(open(f"{a.out}/{a.stats_name}").read(), flush=True)

LG = {k: np.log10(v + a.eps) for k, v in RUNS.items()}
both = np.concatenate([v[1:].ravel() for v in LG.values()])
lo, hi = (float(v) for v in np.percentile(both, a.q))
q_lo, q_hi = lo, hi
if a.range:
    _R = json.load(open(a.range))
    assert abs(float(_R["eps"]) - a.eps) < 1e-15, ("eps 가 다르다", _R["eps"], a.eps)
    lo, hi = float(_R["lo"]), float(_R["hi"])
    print(f"[색 범위] {a.range} 고정 범위 그대로 [{lo:.4f}, {hi:.4f}] (이 실행들만의 {a.q} 분위수는 [{q_lo:.4f}, {q_hi:.4f}])", flush=True)
del both
json.dump(dict(lo=lo, hi=hi, eps=a.eps, q=a.q, cmap=a.cmap, unit="Pa (J/m^3)", runs=list(RUNS), dumps=info,
               psi="fixed corotated mu*sum(sig-1)^2 + lam/2*(J-1)^2", mu=mu, lam=lam, E=cfg["E"], nu=cfg["nu"],
               T=T, N=N, frames_for_range=(f"고정: {a.range}" if a.range else f"1..{T - 1} (실행 {len(RUNS)} 개 합침)"),
               own_quantile_range=[q_lo, q_hi]),
          open(f"{a.out}/range.json", "w"), indent=1)
print(f"[색 범위] log10(psi+{a.eps}) [{lo:.4f}, {hi:.4f}] (분위수 {a.q}, {a.cmap}, 실행 {list(RUNS)})", flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
cmap = matplotlib.colormaps[a.cmap]
for k, v in LG.items():
    C = np.empty((T, N, 3), np.float16)
    for t in range(T):
        C[t] = cmap(np.clip((v[t] - lo) / (hi - lo), 0.0, 1.0))[:, :3].astype(np.float16)
    np.save(f"{a.out}/col_{k}.npy", C)
    print(f"[색] {a.out}/col_{k}.npy {C.shape}", flush=True)
edges = np.linspace(lo - a.pad, hi + a.pad, a.bins + 1)
Z = dict(edges=edges, lo=lo, hi=hi, eps=a.eps, runs=np.array(list(RUNS)))
for k, v in LG.items():
    Z[f"h_{k}"] = np.stack([np.histogram(np.clip(v[t], edges[0], edges[-1]), bins=edges)[0] / N for t in range(T)])
    Z[f"mean_{k}"] = RUNS[k].mean(1); Z[f"med_{k}"] = np.median(RUNS[k], 1)
np.savez(f"{a.out}/psi_hist.npz", **Z)
# psi_dist_vid 형식 (--hist: h_gt·h_ov = 첫 덤프, --hist2: 실행마다 h_ov = 그 덤프)
for k in list(RUNS)[1:]:
    np.savez(f"{a.out}/psi_hist_{k}.npz", edges=edges, lo=lo, hi=hi, eps=a.eps, h_gt=Z["h_gt"], mean_gt=Z["mean_gt"], med_gt=Z["med_gt"],
             h_ov=Z[f"h_{k}"], mean_ov=Z[f"mean_{k}"], med_ov=Z[f"med_{k}"])
print(f"[히스토그램] {a.out}/psi_hist.npz 구간 [{edges[0]:.3f}, {edges[-1]:.3f}] {a.bins} 칸, 가중 없음", flush=True)
print("PSI_COLOR_DONE", flush=True)
