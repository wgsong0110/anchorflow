"""정답(PG h5)·우리(롤아웃 덤프) 입자별 탄성 에너지 밀도 psi 를 **같은 함수**로 프레임마다 재고,
두 궤적 공통 고정 범위로 색을 매겨 lb_render --colors 용 .npy 를 만든다. 분포 패널용 히스토그램도 같이 저장.

psi = phys_resid.psi_of(F, cfg) -- jelly 는 fixed corotated  mu Σ(σ_i-1)^2 + lam/2 (J-1)^2 (σ 하한 0.01, J=Πσ).
PG 의 jelly(material 0) 응력 kirchoff_stress_FCR = 2mu(F-R)Fᵀ + lam J(J-1) I 가 이 에너지의 미분이고,
우리 증분 포텐셜(pts_ip_energy)의 탄성항 Σ V_p psi_of(F_p) 도 같은 함수다.
ficus 공식 config 는 두 영역 모두 E 2e6, nu 0.4 (밀도만 200/70) 라 mu·lam 은 전 입자 공통.

색: c = log10(psi + eps) 를 [lo, hi] (두 궤적 1~T 프레임 전 입자 합친 분위수, 고정 상수) 에서 잘라 컬러맵 하나로.
히스토그램: 같은 구간, **가중 없음(입자 수 비율)**.

  python exe/psi_color.py --gt_h5 W/topo/pg_ficus_elastic --dump W/lbours/ficus_ov/roll_ov.pt \
      --config W/lbours/ficus_ov/gt/config.json --out W/lbours/ficus_psi
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import phys_resid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gt_h5", required=True, help="PG gs_simulation h5 디렉토리 (f_tensor (9,N) float32)")
ap.add_argument("--dump", required=True, help="train_deform 롤아웃 덤프 (AF_ROLL_DUMP_F=32)")
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--eps", type=float, default=1e-3, help="log10(psi+eps) 의 작은 상수 [Pa]")
ap.add_argument("--q", type=float, nargs=2, default=[1.0, 99.0], help="색 범위 분위수 (두 궤적 합쳐서, 0 프레임 제외)")
ap.add_argument("--range", type=float, nargs=2, default=None,
                help="색 범위 log10(psi+eps) [lo, hi] 를 이 고정값으로 (다른 실행 영상과 같은 색 척도; 없으면 --q 분위수)")
ap.add_argument("--stats_frames", type=int, nargs="*", default=[],
                help="이 프레임들에서 변형 통계(입자별 max|σ-1| 의 중앙·상위 1%·최대, det F<0 입자 수)를 정답·우리 같이 "
                     "def_stats.json 에 (σ = F 특이값, 같은 float32 F)")
ap.add_argument("--gt_trunc", action="store_true",
                help="우리 롤아웃이 정답보다 짧으면 (예: N32s 15 프레임) 정답을 우리 길이(0..L)로 잘라 같은 프레임만 비교한다. 없으면 길이가 다를 때 멈춘다")
ap.add_argument("--cmap", default="turbo")
ap.add_argument("--bins", type=int, default=120)
ap.add_argument("--pad", type=float, default=1.0, help="히스토그램 구간 = [lo-pad, hi+pad] (바깥 값은 끝 칸에)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
cfg = json.load(open(a.config))
mu, lam = phys_resid.lame(cfg["E"], cfg["nu"])
print(f"[재질] {phys_resid.mat_name(cfg)} E {cfg['E']} nu {cfg['nu']} -> mu {mu:.4g} lam {lam:.4g}; "
      f"영역 물성 {[(p.get('E'), p.get('nu'), p.get('density')) for p in cfg.get('additional_material_params', [])]}", flush=True)
assert phys_resid.mat_name(cfg) == "jelly"
for p in cfg.get("additional_material_params", []):
    assert float(p.get("E", cfg["E"])) == float(cfg["E"]) and float(p.get("nu", cfg["nu"])) == float(cfg["nu"]), \
        "영역별 E/nu 가 다르다 -- 입자별 mu/lam 이 필요"


@torch.no_grad()
def psi(F):
    F = torch.as_tensor(F).to(dev).float().reshape(-1, 3, 3)
    out = torch.empty(F.shape[0], device=dev)
    for s in range(0, F.shape[0], 65536):
        out[s:s + 65536] = phys_resid.psi_of(F[s:s + 65536], cfg, float(cfg.get("frame_dt", 0.04)))[0]
    return out.cpu().numpy().astype(np.float32)


# 정답: PG 원본 h5 (float32 F)
fs = sorted(glob.glob(os.path.join(a.gt_h5, "**", "sim_*.h5"), recursive=True),
            key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
@torch.no_grad()
def defstat(F):
    F = torch.as_tensor(F).to(dev).float().reshape(-1, 3, 3)
    ms, dt = [], []
    for s in range(0, F.shape[0], 65536):
        ms.append((torch.linalg.svdvals(F[s:s + 65536]) - 1.0).abs().amax(-1)); dt.append(torch.linalg.det(F[s:s + 65536]))
    ms, dt = torch.cat(ms), torch.cat(dt)
    return dict(med=float(ms.median()), p99=float(torch.quantile(ms, 0.99)), max=float(ms.max()), det_neg=int((dt < 0).sum()),
                N=int(F.shape[0]))


STATS = {"gt": {}, "ov": {}}
PG = []
for i, f in enumerate(tqdm(fs, desc="정답 psi")):
    with h5py.File(f, "r") as h:
        Fm = np.array(h["f_tensor"], dtype=np.float32)
    Fm = Fm.T.reshape(-1, 3, 3)
    if i == 0 and np.abs(Fm).max() == 0.0:
        Fm = np.broadcast_to(np.eye(3, dtype=np.float32), Fm.shape).copy()   # PG 0 프레임 미초기화 -> 항등 (b670068)
    PG.append(psi(Fm))
    if i in a.stats_frames:
        STATS["gt"][i] = defstat(Fm)
PG = np.stack(PG)
# 우리: 덤프 F (프레임 1..L), 0 프레임은 항등
D = torch.load(a.dump, map_location="cpu", weights_only=False)
assert D.get("F") is not None and D["F"].dtype == torch.float32, "덤프에 float32 F 가 없다 -- AF_ROLL_DUMP_F=32 로 다시"
L = D["F"].shape[0]
OV = [np.zeros(D["F"].shape[1], np.float32)]
for i in tqdm(range(L), desc="우리 psi"):
    OV.append(psi(D["F"][i]))
    if i + 1 in a.stats_frames:
        STATS["ov"][i + 1] = defstat(D["F"][i])     # 덤프 F[i] = 프레임 i+1
OV = np.stack(OV)
if a.gt_trunc and OV.shape[0] < PG.shape[0] and OV.shape[1] == PG.shape[1]:
    print(f"[정답 자름] --gt_trunc: 정답 {PG.shape[0]} 프레임 -> 우리 길이 {OV.shape[0]} (0..{OV.shape[0] - 1})", flush=True)
    PG = PG[:OV.shape[0]]
    STATS["gt"] = {k: v for k, v in STATS["gt"].items() if k < OV.shape[0]}
assert OV.shape == PG.shape, (OV.shape, PG.shape)
T, N = PG.shape
print(f"[psi] 정답 {PG.shape} 우리 {OV.shape}; 0 프레임 최대 정답 {PG[0].max():.3g} 우리 {OV[0].max():.3g}", flush=True)
# 검산: 덤프에 같이 저장된 psi (fscal[...,4] = 같은 psi_of, 우리 F_ov) 와 비교
if D.get("fscal") is not None:
    fsc = np.asarray(D["fscal"][..., 4].float())
    d_ = np.abs(fsc - OV[1:]).max() / max(np.abs(fsc).max(), 1e-12)
    print(f"[검산] 우리 psi vs 덤프 fscal psi: 최대 차이/최대값 {d_:.2e}", flush=True)
if D.get("gt_fscal") is not None:
    g_ = np.asarray(D["gt_fscal"][..., 4].float())
    print(f"[검산] 정답 psi(원본 h5 f32) vs 덤프 gt_fscal psi(half F): 합 비 "
          f"{PG[1:].sum() / max(g_.sum(), 1e-12):.4f}, 중앙값 {np.median(PG[1:]):.4g} vs {np.median(g_):.4g}", flush=True)
if a.stats_frames:
    json.dump(STATS, open(f"{a.out}/def_stats.json", "w"), indent=1)
    for t in a.stats_frames:
        for nm in ("gt", "ov"):
            r = STATS[nm].get(t)
            if r:
                print(f"[변형] 프레임 {t} {'정답' if nm == 'gt' else '우리'}: max|σ-1| 중앙 {r['med']:.3e} 상위1% {r['p99']:.3e} "
                      f"최대 {r['max']:.3e}, det<0 {r['det_neg']}/{r['N']}", flush=True)
np.save(f"{a.out}/psi_gt.npy", PG)
np.save(f"{a.out}/psi_ov.npy", OV)
del D

lg_gt, lg_ov = np.log10(PG + a.eps), np.log10(OV + a.eps)
both = np.concatenate([lg_gt[1:].ravel(), lg_ov[1:].ravel()])
lo, hi = (float(v) for v in np.percentile(both, a.q))
print(f"[색 범위] 이 두 궤적의 {a.q} 분위수 [{lo:.4f}, {hi:.4f}]", flush=True)
if a.range is not None:
    lo, hi = float(a.range[0]), float(a.range[1])
    print(f"[색 범위] --range 고정값 [{lo:.4f}, {hi:.4f}] 를 쓴다; 범위 밖 비율 아래 {float((both < lo).mean()):.4f} "
          f"위 {float((both > hi).mean()):.4f}", flush=True)
del both
rng = dict(lo=lo, hi=hi, eps=a.eps, q=a.q, fixed_range=a.range is not None, cmap=a.cmap, unit="Pa (J/m^3)",
           psi="fixed corotated mu*sum(sig-1)^2 + lam/2*(J-1)^2", mu=mu, lam=lam, E=cfg["E"], nu=cfg["nu"],
           T=T, N=N, frames_for_range="1..T-1 (두 궤적 합침)")
json.dump(rng, open(f"{a.out}/range.json", "w"), indent=1)
print(f"[색 범위] log10(psi+{a.eps}) [{lo:.4f}, {hi:.4f}] (분위수 {a.q}, {a.cmap})", flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
cmap = matplotlib.colormaps[a.cmap] if hasattr(matplotlib, "colormaps") else cm.get_cmap(a.cmap)
for nm, LG in (("gt", lg_gt), ("ov", lg_ov)):
    C = np.empty((T, N, 3), np.float16)
    for t in range(T):
        u = np.clip((LG[t] - lo) / (hi - lo), 0.0, 1.0)
        C[t] = cmap(u)[:, :3].astype(np.float16)
    np.save(f"{a.out}/col_{nm}.npy", C)
    print(f"[색] {a.out}/col_{nm}.npy {C.shape}", flush=True)

edges = np.linspace(lo - a.pad, hi + a.pad, a.bins + 1)
def hist(LG):
    H = np.empty((T, a.bins), np.float64)
    for t in range(T):
        H[t] = np.histogram(np.clip(LG[t], edges[0], edges[-1]), bins=edges)[0] / LG.shape[1]
    return H
np.savez(f"{a.out}/psi_hist.npz", edges=edges, h_gt=hist(lg_gt), h_ov=hist(lg_ov),
         mean_gt=PG.mean(1), mean_ov=OV.mean(1), med_gt=np.median(PG, 1), med_ov=np.median(OV, 1),
         mlog_gt=lg_gt.mean(1), mlog_ov=lg_ov.mean(1), lo=lo, hi=hi, eps=a.eps)
print(f"[히스토그램] {a.out}/psi_hist.npz  구간 [{edges[0]:.3f}, {edges[-1]:.3f}] {a.bins} 칸, 가중 없음", flush=True)
print("PSI_COLOR_DONE", flush=True)
