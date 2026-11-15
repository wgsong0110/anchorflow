#!/usr/bin/env python3
"""절 CM186 점소성 진단: 정답 (PG h5: x, f_tensor = 반환 사상 뒤 탄성 F) 과 우리 판 (pred.pt: pred, F = 프레임 끝 ag_rm 뒤 입자 탄성 F) 을 프레임별로 같은 잣대로.
  - com_z: 입자 평균 높이 (입자 질량 = 부피 x 밀도, 같은 밀도라 정답 입자 부피 가중 대신 단순 평균 -- 두 쪽 같은 입자 집합이라 비교에 충분)
  - ke: 프레임 사이 변위로 낸 운동 에너지 대리 0.5 <|dx/dt|^2> (출렁임)
  - psi: 입자 평균 탄성 에너지 밀도 (PG 응력 모형: plasticine 0/5 = FCR, metal 1·foam 3 = 헨키 StVK)
  - yfrac: 항복면 위 (또는 밖) 입자 비율 -- 헨키 편차 ‖2μ ε̂‖ >= 0.99 x (plasticine·metal σ_y, foam √(2/3) σ_y; 정답 식 그대로) (plasticine 은 손상으로 σ_y 가 줄어들 수 있어 초기 σ_y 기준 = 하한)
  - edev: 평균 ‖ε̂‖ (탄성 편차 변형률)
  python nshape_plast_an.py --h5 DIR --pred pred.pt --config cfg.json --out out.json"""
import argparse, glob, json, math, os
import h5py, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--h5", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = json.load(open(a.config)); E, nu = float(cfg["E"]), float(cfg["nu"]); mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
ys = float(cfg.get("yield_stress", 0.0)); mat = cfg.get("material"); fdt = float(cfg["frame_dt"])
dev = "cuda" if torch.cuda.is_available() else "cpu"


def stats(x, F, xprev):
    F = F.to(dev).double(); x = x.to(dev).double()
    ok = torch.isfinite(F.reshape(F.shape[0], -1)).all(1) & torch.isfinite(x).all(1)
    F, x = F[ok], x[ok]
    U, S, Vh = torch.linalg.svd(F)
    S = S.clamp_min(1e-6); eps = torch.log(S); tr = eps.sum(1, keepdim=True); ed = eps - tr / 3
    if mat in ("jelly", "plasticine"):
        R = U @ Vh; J = S.prod(1)
        psi = mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2
    else:
        psi = mu * (eps ** 2).sum(1) + 0.5 * lam * tr[:, 0] ** 2
    sn = (2 * mu * ed).norm(dim=1)
    thr = ys if mat in ("plasticine", "metal") else math.sqrt(2 / 3) * ys   # 절 CM189: PG von Mises (재질 1·5) 는 ‖dev τ‖ > ys, 점소성 (재질 3) 은 ‖s‖ > √(2/3) ys
    yf = float((sn >= 0.99 * thr).double().mean()) if ys > 0 else float("nan")
    ke = float((0.5 * ((x - xprev.to(dev).double()[ok]) / fdt).pow(2).sum(1)).mean()) if xprev is not None else 0.0
    return dict(com_z=float(x[:, 2].mean()), psi=float(psi.mean()), yfrac=yf, edev=float(ed.norm(dim=1).mean()), ke=ke, n=int(ok.sum()))


fs = sorted(glob.glob(os.path.join(a.h5, "**", "sim_*.h5"), recursive=True))
d = torch.load(a.pred, map_location="cpu", weights_only=False); P, PF = d["pred"], d["F"]
T = min(len(fs), P.shape[0]); G, O = [], []
xg_prev = xo_prev = None
for t in range(T):
    with h5py.File(fs[t]) as h:
        x = np.array(h["x"]); x = x.T if x.shape[0] == 3 else x
        F = np.array(h["f_tensor"]); F = (F.T if F.shape[0] == 9 else F).reshape(-1, 3, 3)
    xg = torch.as_tensor(x).float(); G.append(stats(xg, torch.as_tensor(F).float(), xg_prev)); xg_prev = xg
    O.append(stats(P[t].float(), PF[t].float(), xo_prev)); xo_prev = P[t].float()
json.dump(dict(material=mat, mu=mu, lam=lam, ys=ys, gt=G, ours=O), open(a.out, "w"))
print("프레임  com_z 정답/우리   psi 정답/우리   항복비 정답/우리   ‖ε̂‖ 정답/우리   KE 정답/우리")
for t in range(0, T, max(1, T // 17)):
    g, o = G[t], O[t]
    print(f"{t:3d}  {g['com_z']:.3f}/{o['com_z']:.3f}  {g['psi']:.3g}/{o['psi']:.3g}  {g['yfrac']:.3f}/{o['yfrac']:.3f}  {g['edev']:.4f}/{o['edev']:.4f}  {g['ke']:.3g}/{o['ke']:.3g}", flush=True)
