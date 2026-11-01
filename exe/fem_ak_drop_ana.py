"""절 AK.29 분석: 떨어지는 블록 궤적 (정답 x240 또는 roll_fem.pt pred) 에서 질량 중심 높이·연직 속도, 전체 아핀 맞춤 (전체 입자 최소제곱, 회전 뺀 신장) 의 연직 신장 ε_zz.
- 반발 계수 e = (착지 뒤 첫 상승 최대 중심 속도) / (착지 직전 중심 속력);  튀어 오름 = 착지 뒤 중심 최고 높이 − 쉼 중심 높이 (바닥 + 처음 블록 중심 아래 깊이);
- 최대 압축 = 착지 뒤 ε_zz 최소;  복원 시간 = 최대 압축 시각부터 ε_zz 가 처음 0 이상이 될 때까지 (영점 교차 보간).
사용: python exe/fem_ak_drop_ana.py <init.pt> <이름:경로:프레임dt> ..."""
import sys

import numpy as np
import torch

d = torch.load(sys.argv[1], weights_only=False)
fz = [b for b in d["cfg"]["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2]
X0 = d["x"][0].double()
Xr = X0 - X0.mean(0)
z_rest = fz + float(X0[:, 2].mean() - X0[:, 2].min())
print(f"쉼 중심 높이 {z_rest:.4f}, 처음 중심 속도 {float(d['v'][0][0, 2]):.3f} m/s", flush=True)
for arg in sys.argv[2:]:
    nm, p, fdt = arg.rsplit(":", 2)
    fdt = float(fdt)
    o = torch.load(p, map_location="cpu", weights_only=False)
    X = o["pred"] if isinstance(o, dict) else o
    if X.shape[0] and float((X[0].double() - X0).abs().max()) > 1e-6:
        X = torch.cat([X0.float()[None], X.float()], 0)
    T = X.shape[0]
    cz = np.array([float(X[t][:, 2].double().mean()) for t in range(T)])
    if not np.isfinite(cz).all():
        T = int(np.argmax(~np.isfinite(cz))); cz = cz[:T]
    ez = []
    for t in range(T):
        x = X[t].double()
        B = torch.linalg.lstsq(Xr, x - x.mean(0)).solution.T
        U_, S_, Vh_ = torch.linalg.svd(B)
        ez.append(float((Vh_.T @ torch.diag(S_) @ Vh_)[2, 2] - 1))
    ez = np.array(ez)
    vz = np.diff(cz) / fdt
    tt = np.arange(T) * fdt
    i_c = int(np.argmin(vz))                                                          # 가장 빠른 하강 = 착지 직전
    v_in = -vz[i_c]
    up = vz[i_c:]
    v_out = float(up.max()) if up.size else float("nan")
    i_pk = i_c + int(np.argmin(ez[i_c:]))
    eps_min = float(ez[i_pk])
    rec = float("nan")
    for i in range(i_pk + 1, T):
        if ez[i] >= 0:
            rec = (tt[i - 1] + (0 - ez[i - 1]) / (ez[i] - ez[i - 1]) * fdt) - tt[i_pk]
            break
    hb = float(cz[i_pk:].max() - z_rest)                                            # 최대 압축 뒤 중심 최고 높이
    s60 = max(1, int(round((1 / 60) / fdt)))
    print(f"{nm}: 프레임 {T - 1}, 착지 속력 {v_in:.3f} m/s, 상승 최대 {v_out:.3f} m/s, 반발 계수 {v_out / v_in:.3f}, 튀어 오름 {1e3 * hb:.1f} mm, "
          f"최대 압축 ε_zz {eps_min:.4f} (t {tt[i_pk]:.4f} s), 복원 시간 {1e3 * rec:.1f} ms, ε_zz 1/60 마다 {np.round(ez[::s60][:24], 4).tolist()}", flush=True)
