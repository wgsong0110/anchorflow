"""절 AN.17 접촉 분석 (lego 미달 궤적): 정답과 예측 덤프의 강체 운동·접촉 충격량을 프레임별로 나눈다 (진단, 결과 정의 무관).
프레임 t (기록 t = 1..L, 0 = 처음 상태) 마다:
  e %          = mean |x − x_gt| / EXT (판정 지표)
  zc mm        = 질량 중심 바닥 법선 높이 (평면 기준)
  tilt 도      = 처음 상태에서 지금까지의 최적 강체 회전 (Kabsch) 중 수평축 몫 (기울기), yaw = 법선축 몫
  dR 도        = 예측 강체 회전과 정답 강체 회전의 차 (예측 행만)
  w rad/s      = 질량 중심 둘레 각속도 ω = I⁻¹ L (L 은 중앙 차분 속도), 수평·법선 성분
  KEr / KEt J  = 회전·병진 운동 에너지
  pen %        = 평면 아래 입자 몫, band % = 평면 위 1 cm 안 입자 몫, arm mm = 그 접촉 입자들 중심과 질량 중심의 수평 거리
  Jn / Jt N·s  = 그 프레임 접촉 충격량 (질량 중심 운동량 변화 − 중력) 의 법선·접선 크기, Jt/Jn
  Tq N·m·s     = 각운동량 변화 (접촉 토크 충격량) 크기
덧붙임 (--pic): 정답 상태에 우리 격자 (능면체 정육면체, 간격 = 정답 dx) 로 P2G -> G2P 한 번 왕복했을 때 잃는 운동 에너지·각운동량 몫,
  그리고 노드 집중 질량의 관성 텐서 대 입자 관성 텐서 (고유값 비).
사용: python exe/fem_an_contact_ana.py <궤적.pt> <이름:pred.pt> [...] [--pic] [--frames a,b,..]"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from anchorflow import deform_model as DM  # noqa: E402
from anchorflow import simplex as SX  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load(args[0], map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
X = d["x"].to(dev).double()
g = torch.tensor([float(q) for q in cfg["g"]], dtype=torch.float64, device=dev)
pl = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"]
NR = torch.tensor(pl[0]["normal"], dtype=torch.float64, device=dev); NR = NR / NR.norm()
PT = torch.tensor(pl[0]["point"], dtype=torch.float64, device=dev)
M = float(m.sum())
X0 = X[0]
c0 = (m[:, None] * X0).sum(0) / M


def kabsch(P):
    """처음 상태 X0 -> P 최적 강체 회전 (질량 가중)."""
    cp_ = (m[:, None] * P).sum(0) / M
    H = ((X0 - c0) * m[:, None]).T @ (P - cp_)
    U, S, Vt = torch.linalg.svd(H)
    dd = float(torch.sign(torch.det(Vt.T @ U.T)))
    D = torch.diag(torch.tensor([1.0, 1.0, dd], dtype=torch.float64, device=dev))
    return Vt.T @ D @ U.T


def rotvec(R):
    ang = math.acos(max(-1.0, min(1.0, (float(torch.trace(R)) - 1) / 2)))
    if ang < 1e-9:
        return torch.zeros(3, dtype=torch.float64, device=dev)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return w * ang


def series(P):
    """P [T+1, N, 3] (0 = 처음 상태) -> 프레임별 dict 목록."""
    T = P.shape[0] - 1
    out = []
    C = (m[None, :, None] * P).sum(1) / M                                       # [T+1, 3]
    for t in range(T + 1):
        x = P[t]
        c = C[t]
        r = x - c
        if 0 < t < T:
            v = (P[t + 1] - P[t - 1]) / (2 * h)
        elif t == 0:
            v = (P[1] - P[0]) / h
        else:
            v = (P[t] - P[t - 1]) / h
        vc = (m[:, None] * v).sum(0) / M
        vr = v - vc
        Lm = (m[:, None] * torch.cross(r, vr, dim=1)).sum(0)
        I = (m[:, None, None] * ((r * r).sum(1)[:, None, None] * torch.eye(3, dtype=torch.float64, device=dev) - r[:, :, None] * r[:, None, :])).sum(0)
        w = torch.linalg.solve(I, Lm)
        wn = float(w @ NR); wh = float((w - wn * NR).norm())
        R = kabsch(x)
        rv = rotvec(R)
        rvn = float(rv @ NR); rvh = float((rv - rvn * NR).norm())
        s = (x - PT) @ NR
        cont = s < 0.01
        if bool(cont.any()):
            dcc = (m[cont, None] * x[cont]).sum(0) / m[cont].sum() - c
            arm = float((dcc - (dcc @ NR) * NR).norm())
        else:
            arm = float("nan")
        out.append(dict(t=t, c=c, vc=vc, L=Lm, R=R, zc=float((c - PT) @ NR), tilt=math.degrees(rvh), yaw=math.degrees(rvn), wh=wh, wn=wn,
                        KEr=0.5 * float(w @ Lm), KEt=0.5 * M * float(vc @ vc), pen=float(m[s < 0].sum()) / M, band=float(m[(s >= 0) & (s < 0.01)].sum()) / M, arm=arm, smin=float(s.min())))
    for t in range(T):                                                              # 프레임 t -> t+1 접촉 충격량 (기록 간격 h 의 운동량 변화 − 중력)
        J = M * (out[t + 1]["vc"] - out[t]["vc"]) - M * g * h
        Jn = float(J @ NR); Jt = float((J - Jn * NR).norm())
        out[t + 1]["Jn"], out[t + 1]["Jt"] = Jn, Jt
        out[t + 1]["Tq"] = float((out[t + 1]["L"] - out[t]["L"]).norm())
        out[t + 1]["Jvec"] = J
    out[0]["Jn"] = out[0]["Jt"] = out[0]["Tq"] = 0.0
    out[0]["Jvec"] = torch.zeros(3, dtype=torch.float64, device=dev)
    return out


L_ = X.shape[0] - 1
runs = [("gt", X)]
for spec in args[1:]:
    nm, pp = spec.split(":", 1)
    if not os.path.exists(pp):
        print(f"(없음) {nm}: {pp}")
        continue
    Pd = torch.load(pp, map_location="cpu", weights_only=False)["pred"].to(dev).double()
    runs.append((nm, torch.cat([X0[None], Pd], 0)))
T = min(r[1].shape[0] for r in runs) - 1
S = {nm: series(P[:T + 1]) for nm, P in runs}
G = S["gt"]
hdr = "t | e% | zc mm | tilt 도 | yaw 도 | dR 도 | w수평 / w법선 rad/s | KEr / KEt J | pen % / band % | arm mm | Jn / Jt N·s | Tq"
for nm, P in runs:
    A = S[nm]
    print(f"\n## {nm} ({T} 프레임) -- {hdr}")
    for t in range(T + 1):
        q = A[t]
        e = float((P[t] - X[t]).norm(dim=-1).mean()) / EXT * 100 if nm != "gt" else 0.0
        dR = math.degrees(float(rotvec(q["R"] @ G[t]["R"].T).norm())) if nm != "gt" else 0.0
        print(f"  {t:2d} | {e:5.2f} | {1e3 * q['zc']:7.1f} | {q['tilt']:6.2f} | {q['yaw']:6.2f} | {dR:6.2f} | {q['wh']:6.3f} / {q['wn']:6.3f} | {q['KEr']:7.3f} / {q['KEt']:7.3f} | "
              f"{100 * q['pen']:5.2f} / {100 * q['band']:5.2f} | {1e3 * q['arm']:6.1f} | {q['Jn']:7.3f} / {q['Jt']:6.3f} | {q['Tq']:6.3f}")
    # 요약: 첫 접촉, 창별 충격량 합
    tc = next((t for t in range(T + 1) if A[t]["smin"] < 0.005), None)
    print(f"  첫 접촉 (최저점 < 5 mm) 프레임 {tc}; 질량 {M:.3f} kg")
    for a_, b_ in ((1, 14), (15, 25), (26, 40), (41, 60), (61, T)):
        if a_ > T:
            continue
        b_ = min(b_, T)
        Jv = sum((A[t]["Jvec"] for t in range(a_, b_ + 1)), torch.zeros(3, dtype=torch.float64, device=dev))
        Jn = float(Jv @ NR); Jt = float((Jv - Jn * NR).norm())
        dL = A[b_]["L"] - A[a_ - 1]["L"]
        print(f"  창 {a_}-{b_}: 접촉 충격량 법선 {Jn:.3f} 접선 {Jt:.3f} (비 {Jt / max(abs(Jn), 1e-9):.3f}), 각운동량 변화 {float(dL.norm()):.3f} (수평 {float((dL - (dL @ NR) * NR).norm()):.3f}), "
              f"KEr {A[a_ - 1]['KEr']:.3f} -> {A[b_]['KEr']:.3f}, KEt {A[a_ - 1]['KEt']:.3f} -> {A[b_]['KEt']:.3f}, tilt {A[b_]['tilt']:.2f} 도")

if "--pic" in sys.argv:
    DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
    NODE_H = DXG * math.sqrt(2.0)
    lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
    A, Ai = lat.A, lat.Ai
    V = d.get("v")
    fr = [5, 10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60]
    for a in sys.argv:
        if a.startswith("--frames="):
            fr = [int(q) for q in a.split("=")[1].split(",")]
    print("\n## 우리 격자 P2G -> G2P 한 번 왕복 (정답 상태, PIC) -- 프레임 | KE 잃음 % | 회전 KE 잃음 % | |L| 잃음 % | 질량 중심 둘레 L 방향 바뀜 도 | 노드 관성 / 입자 관성 고유값 비")
    for t in fr:
        if t > L_:
            continue
        x = X[t]
        v = (V[t].to(dev).double() if V is not None else ((X[min(t + 1, L_)] - X[t - 1]) / (2 * h)))
        y = x @ Ai.T
        b = torch.floor(y - 0.5)
        f = y - b
        b = b.long()
        w = torch.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2], -1)   # [N,3 축,3]
        bmin = b.min(0).values
        bb = b - bmin
        ext = bb.max(0).values + 3
        nn_ = int(ext[0] * ext[1] * ext[2])
        mn = torch.zeros(nn_, dtype=torch.float64, device=dev); mv = torch.zeros(nn_, 3, dtype=torch.float64, device=dev)
        ids, ws = [], []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = ((bb[:, 0] + i) * ext[1] + (bb[:, 1] + j)) * ext[2] + (bb[:, 2] + k)
                    ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                    mn.index_add_(0, idx, ww * m); mv.index_add_(0, idx, (ww * m)[:, None] * v)
                    ids.append(idx); ws.append(ww)
        vn = torch.where(mn[:, None] > 0, mv / mn.clamp_min(1e-300)[:, None], torch.zeros_like(mv))
        v2 = sum(ws[o][:, None] * vn[ids[o]] for o in range(27))
        c = (m[:, None] * x).sum(0) / M
        r = x - c

        def ke_l(vv):
            vc = (m[:, None] * vv).sum(0) / M
            Lm = (m[:, None] * torch.cross(r, vv - vc, dim=1)).sum(0)
            I = (m[:, None, None] * ((r * r).sum(1)[:, None, None] * torch.eye(3, dtype=torch.float64, device=dev) - r[:, :, None] * r[:, None, :])).sum(0)
            wv = torch.linalg.solve(I, Lm)
            return 0.5 * float((m * (vv * vv).sum(1)).sum()), 0.5 * float(wv @ Lm), Lm, I
        k1, kr1, L1, I1 = ke_l(v)
        k2, kr2, L2, _ = ke_l(v2)
        # 노드 집중 질량 관성 (노드 위치 = A (b + 정수))
        nz = (mn > 0).nonzero().squeeze(1)
        iz = nz % ext[2]; iy = (nz // ext[2]) % ext[1]; ix = nz // (ext[1] * ext[2])
        xn = (torch.stack([ix, iy, iz], 1) + bmin[None, :]).double() @ A.T
        rn = xn - c
        In = (mn[nz, None, None] * ((rn * rn).sum(1)[:, None, None] * torch.eye(3, dtype=torch.float64, device=dev) - rn[:, :, None] * rn[:, None, :])).sum(0)
        ev1 = torch.linalg.eigvalsh(I1); evn = torch.linalg.eigvalsh(In)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(L1 @ L2) / max(float(L1.norm() * L2.norm()), 1e-30)))))
        print(f"  {t:2d} | {100 * (1 - k2 / k1):6.3f} | {100 * (1 - kr2 / max(kr1, 1e-30)):6.3f} | {100 * (1 - float(L2.norm()) / max(float(L1.norm()), 1e-30)):6.3f} | {ang:5.2f} | "
              + " ".join(f"{float(evn[i] / ev1[i]):.4f}" for i in range(3)) + f" (|L| {float(L1.norm()):.3f}, KEr {kr1:.3f} J)")
