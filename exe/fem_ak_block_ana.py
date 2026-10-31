"""절 AK.24 분석: 위치 궤적 [T,N,3] 에서 프레임마다 전체 아핀 맞춤 (전체 입자 최소제곱 -- 이웃 추정 아님) 의 변형 성분 (stretch: ε_xx, shear: γ_xy = A01 + A10 대칭 몫) 시계열,
주기 (영점 교차 평균 x 2, FFT), 진폭 감쇠 (첫 극값 대비 마지막 극값). 사용: python exe/fem_ak_block_ana.py <block.pt> <이름:경로 (x.pt 또는 roll_fem.pt)> ..."""
import sys

import numpy as np
import torch

d = torch.load(sys.argv[1], weights_only=False)
X0 = d["x"][0].double()
A0 = d["A"].double()
Xr = (X0 - X0.mean(0)) @ torch.linalg.inv(A0).T                                     # 쉼 모양 좌표
kind = "stretch" if abs(float(A0[0, 0]) - 1) > 1e-6 else "shear"
dt = float(d["cfg"]["frame_dt"])
for arg in sys.argv[2:]:
    nm, p = arg.split(":", 1)
    o = torch.load(p, map_location="cpu", weights_only=False)
    X = o["pred"] if isinstance(o, dict) else o
    s = []
    for t in range(X.shape[0]):
        x = X[t].double()
        B = torch.linalg.lstsq(Xr, x - x.mean(0)).solution.T                              # x − c ≈ B Xr
        U_, S_, Vh_ = torch.linalg.svd(B)
        Sym = Vh_.T @ torch.diag(S_) @ Vh_                                               # 회전 뺀 신장 (오른쪽 극분해)
        s.append(float(Sym[0, 0] - 1) if kind == "stretch" else float(Sym[0, 1] + Sym[1, 0]))
    s = np.array([float(A0[0, 0] - 1) if kind == "stretch" else float(A0[0, 1])] + s)
    tt = np.arange(len(s)) * dt
    m = s - s[len(s) // 2:].mean()
    zc = [tt[i - 1] + m[i - 1] / (m[i - 1] - m[i]) * dt for i in range(1, len(m)) if m[i - 1] * m[i] < 0]
    T_zc = 2 * np.mean(np.diff(zc)) if len(zc) >= 2 else float("nan")
    Fq = np.abs(np.fft.rfft(m * np.hanning(len(m)), 8192))
    fq = np.fft.rfftfreq(8192, dt)
    T_fft = 1 / fq[1:][np.argmax(Fq[1:])]
    ext = [abs(m[i]) for i in range(1, len(m) - 1) if (m[i] - m[i - 1]) * (m[i + 1] - m[i]) < 0]
    dec = (ext[-1] / ext[0]) if len(ext) >= 2 else float("nan")
    print(f"{kind} {nm}: 주기 영점 {T_zc * 1e3:.1f} ms FFT {T_fft * 1e3:.1f} ms, 극값 진폭 끝/처음 {dec:.3f} (극값 {len(ext)} 개), 평균 변형 {s[len(s) // 2:].mean():.4f}, 처음 10 프레임 {np.round(s[:10], 4).tolist()}", flush=True)
