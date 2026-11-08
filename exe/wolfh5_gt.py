"""절 CM70 방출 뒤 낙하 매달림 -- 입자 수준 대조 (읽기만, 결과 불변): 정답 (PG h5 원본 x·v·f_tensor f32) 과 우리 판 덤프 (pred.pt 의 pred·F, --dump_F) 를
같은 식으로 방출 앞면 거리 칸 x 높이 칸 별로 모음. 정답 MPM 은 서브스텝마다 구속 입자 v = 0 (cfg release_particles_sequentially).

칸: d = (t − t_방출) v_앞면 (cm, 방출된 쪽 +; 구속 입자 −), 높이 = 바닥 (z 0.48) 위 (< 10 cm, ≥ 10 cm). 프레임 f 의 값 (f = 1..FMAX):
  ᾱ = Σ_g w_pg α_g (정답 격자 dx 1 cm, α_g = 구속 질량 / 전체 질량, f → f+1 구간 S 4 표본 평균),
  k̄ = Σ_g w_pg k_g, k_g = exp(expo Σ_s log(1 − α_g(t_s))) (expo = Δt/(S dt_gt) = 500; 정답 노드가 그 프레임 동안 유지하는 속도 몫), 1 − k̄ = 구속 노드에 걸린 무게,
  v = (x_{f+1} − x_{f−1})·ẑ / 2Δt (중심 차분), 낙하 몫 = v / (−g τ), τ = 방출 뒤 시간 (프레임 가운데),
  p · |s| = 입자 F (정답 = PG 의 투영 뒤 탄성 F, 우리 = 덤프 F) 의 Hencky 키르히호프 압력 (압축 +)·편차 노름,
  F_el = 재질 응력 힘의 입자 몫 = Σ_g w_pg (m_p/m_g) f_g·ẑ, f_g = −Σ_q V0_q τ_q ∇w_qg (정답 격자 P2G, PG 와 같은 꼴), F_net = m ((v_{f+½} − v_{f−½})·ẑ/Δt − g·ẑ).
사용: python exe/wolfh5_gt.py <wolf.pt> <정답 h5 디렉토리 | -> <출력 json> [판이름=pred.pt ...]   (환경 W5_FMAX = 마지막 프레임, 기본 30)
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch

EDGES = [-6, -3, 0, 3, 6, 11, 21, 40]
FMAX = int(os.environ.get("W5_FMAX", "30"))
FIELDS = ["n", "M", "Mk", "Ma", "Mpin", "Mv", "Mtau", "Mfree", "Fnet", "Fel", "Mp", "Ms"]


def h5_frames(d):
    import h5py
    fs = sorted(glob.glob(os.path.join(d, "**", "sim_*.h5"), recursive=True), key=lambda p: int(re.findall(r"(\d+)", os.path.splitext(os.path.basename(p))[0])[-1]))
    X, F = [], []
    for f in fs[:FMAX + 2]:
        with h5py.File(f, "r") as h:
            X.append(torch.from_numpy(np.array(h["x"]).T.copy()).double()); F.append(torch.from_numpy(np.array(h["f_tensor"]).T.copy()).double().reshape(-1, 3, 3))
    return torch.stack(X), torch.stack(F)


def main():
    wp, hd, outp = sys.argv[1:4]
    d = torch.load(wp, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    x0 = d["x"][0].double()
    N = x0.shape[0]
    ng = int(cfg["n_grid"]); dx = float(cfg["grid_lim"]) / ng
    rho = float(cfg["density"]); E, nu = float(cfg["E"]), float(cfg["nu"])
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    K3 = lam + 2 * mu / 3
    vi = (x0 / dx).long().clamp(0, ng - 1); fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
    cn = torch.zeros(ng ** 3, dtype=torch.float64).index_add_(0, fl, torch.ones(N, dtype=torch.float64))
    m = (dx ** 3) / cn[fl] * rho; V0 = m / rho
    bc = [b for b in cfg["boundary_conditions"] if b.get("type") == "release_particles_sequentially"][0]
    ax = [k for k in range(3) if bc["normal"][k] != 0][0]; c0 = float(bc["end_position"]); H0 = abs(float(bc["start_position"]) - c0); T = float(bc["end_time"]); nl = int(bc.get("num_layers", 50))
    vf = H0 / T; hl = H0 / nl
    kmax = torch.ceil(nl - (x0[:, ax] - c0).abs() / hl) - 1
    trel = torch.where(kmax >= 0, T / nl * (kmax + 1), torch.zeros_like(kmax))
    FL = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]["point"][2]
    fdt = float(cfg["frame_dt"]); sdt = float(cfg["substep_dt"]); S = 4; expo = fdt / S / sdt
    g = -9.8
    G = ng + 4
    off = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3)

    def wts(x):
        y = x / dx; b = torch.floor(y - 0.5); f = y - b
        w = torch.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2], -1)          # N,3(축),3
        dw = torch.stack([-(1.5 - f), -2 * (f - 1), f - 0.5], -1) / dx
        bl = b.long() + 1
        ids = ((bl[:, None, 0] + off[None, :, 0]) * G + (bl[:, None, 1] + off[None, :, 1])) * G + (bl[:, None, 2] + off[None, :, 2])   # N,27
        W = w[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * w[:, 2, off[:, 2]]                                    # N,27
        dW = torch.stack([dw[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * w[:, 2, off[:, 2]], w[:, 0, off[:, 0]] * dw[:, 1, off[:, 1]] * w[:, 2, off[:, 2]],
                          w[:, 0, off[:, 0]] * w[:, 1, off[:, 1]] * dw[:, 2, off[:, 2]]], -1)            # N,27,3
        return ids, W, dW

    def p2g(ids, val):
        o = torch.zeros(G ** 3, dtype=torch.float64)
        o.index_add_(0, ids.reshape(-1), val.reshape(-1))
        return o

    def frame_stats(xm, x, xp, Fp, f):
        t = fdt * f
        ids, W, dW = wts(x)
        mg = p2g(ids, W * m[:, None])
        lk = torch.zeros(G ** 3, dtype=torch.float64); asum = torch.zeros(G ** 3, dtype=torch.float64)
        for s in range(S):
            ts = t + (s + 0.5) * fdt / S
            c = (trel > ts).double()
            mc = p2g(ids, W * (m * c)[:, None])
            a = torch.where(mg > 0, (mc / mg.clamp_min(1e-300)).clamp(max=1.0), torch.zeros_like(mg))
            lk += torch.log1p(-a); asum += a / S
        kg = torch.exp(lk * expo)
        kp = (W * kg[ids]).sum(1); ap = (W * asum[ids]).sum(1)
        # 응력 (Hencky)
        U, sg, Vh = torch.linalg.svd(Fp)
        ep = torch.log(sg.abs().clamp_min(1e-12)); tr = ep.sum(-1)
        dev = ep - tr[:, None] / 3
        tau_p = 2 * mu * ep + lam * tr[:, None]
        tau = U @ torch.diag_embed(tau_p) @ U.transpose(1, 2)
        pr = -K3 * tr; sd = 2 * mu * dev.norm(dim=-1)
        fz = -(V0[:, None] * (tau[:, 2, :][:, None, :] * dW).sum(-1))                                   # N,27 (z 성분)
        fgz = p2g(ids, fz)
        Fel = (W * m[:, None] / mg[ids].clamp_min(1e-300) * fgz[ids]).sum(1)
        vz = (xp[:, 2] - xm[:, 2]) / (2 * fdt)
        vm = (x[:, 2] - xm[:, 2]) / fdt; vpp = (xp[:, 2] - x[:, 2]) / fdt
        Fnet = m * ((vpp - vm) / fdt - g)
        dcm = torch.where(trel > t, -(trel - t), t - trel) * vf * 100
        tau_r = torch.where(trel > t, torch.zeros_like(trel), t - trel)
        hg = x[:, 2] - FL
        b = torch.bucketize(dcm, torch.tensor(EDGES, dtype=torch.float64), right=True) * 2 + (hg >= 0.1).long()
        fr = (trel <= t).double()
        cols = [torch.ones_like(m), m, m * kp, m * ap, m * (1 - kp), m * vz, m * tau_r, m * fr, Fnet, Fel, V0 * pr, V0 * sd]
        out = torch.zeros(len(EDGES) + 1, 2, len(cols), dtype=torch.float64)
        for j, cv in enumerate(cols):
            out[:, :, j] = torch.zeros(2 * (len(EDGES) + 1), dtype=torch.float64).index_add_(0, b, cv).reshape(-1, 2)
        return out

    res = {}
    if hd != "-":
        X, Fh = h5_frames(hd)
        if float(Fh[0].abs().max()) == 0.0:
            Fh[0] = torch.eye(3, dtype=torch.float64)
        res["정답"] = [frame_stats(X[f - 1], X[f], X[f + 1], Fh[f], f).tolist() for f in range(1, min(FMAX, X.shape[0] - 2) + 1)]
        print(f"[정답] 프레임 {len(res['정답'])} h5 {X.shape} F 0 프레임 뺀 |F−I| 최대 {float((Fh[1:] - torch.eye(3, dtype=torch.float64)).abs().max()):.3e}", flush=True)
        del X, Fh
    for a in sys.argv[4:]:
        n, p = a.split("=", 1)
        q = torch.load(p, map_location="cpu", weights_only=False)
        Xr = torch.cat([x0[None], q["pred"].double()], 0)
        if "F" not in q:
            print(f"[{n}] F 없음 -- 건너뜀", flush=True)
            continue
        Fr = torch.cat([torch.eye(3, dtype=torch.float64).expand(1, N, 3, 3), q["F"].double().reshape(-1, N, 3, 3)], 0)
        nf = min(FMAX, Xr.shape[0] - 2)
        res[n] = [frame_stats(Xr[f - 1], Xr[f], Xr[f + 1], Fr[f], f).tolist() for f in range(1, nf + 1)]
        print(f"[{n}] 프레임 {nf}", flush=True)
        del q, Xr, Fr
    json.dump(dict(edges_cm=EDGES, fields=FIELDS, fmax=FMAX, g=g, note="res[판][f−1][거리 칸][높이 칸][칸] = 합", res=res), open(outp, "w"))


if __name__ == "__main__":
    main()
