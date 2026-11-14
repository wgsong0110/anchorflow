"""절 CM145 (CPU, 읽기만): escwhy 기록으로 앞면 부분 구속 노드 (0.01 < KB < 0.99, 고정 아님) 의 걸음 속도 변화와 운동량 수지.
(1) 구속 몫 KB 구간별: 질량 가중 걸음 수직 속도 변화 비 = (V1_z - V0_z) / (-g h) (자유낙하 = 1, 정답 식이면 구속 노드는 몫 (1-α) 꼴로 줄어든 값), 수평 x 변화 (m/s)
(2) 노드 합 운동량: 부분 구속 노드 Σ m ΔV_x · Σ m ΔV_z, 그 노드들이 받은 탄성 충격량 Σ m·(-Fn/m)·h = -Σ Fn h 과 중력 충격량 -g h Σ m 와의 차 (수지 = Σ m ΔV - (중력 + 힘 충격량 γ 가중 못 함: 걸음 끝 힘 Fn 하나로 근사)).
사용: python exe/cm145_part.py <esc 디렉토리> F0 F1"""
import os, sys
import numpy as np
esc = sys.argv[1]; F0 = int(sys.argv[2]); F1 = int(sys.argv[3]); h = 0.04; g = 9.8; gam = 0.611
for f in range(F0, F1 + 1):
    p = os.path.join(esc, f"f{f:03d}.npz")
    if not os.path.exists(p): continue
    z = np.load(p); IP = z["IP"]; A = z["A"].astype(np.float64); nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8])
    i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
    X = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1).astype(np.float64) @ A.T
    mn = z["mn"].astype(np.float64); V0 = z["V0"][0].astype(np.float64); V1 = z["V1"][0].astype(np.float64); A0 = z["A0"][0]; A1 = z["A1"][0]; Fn = z["Fn"][0].astype(np.float64); KB = z["KB"][0]; PM = z["PM"][0]
    front = 1.5 - f * h / 1.5
    base = (mn > 0) & (PM != 2) & (X[:, 2] > 0.6) & (np.abs(X[:, 0] - front) < 0.12)
    print(f"f{f} 앞면 x={front:.3f} ±0.12, 바닥 위 > 0.12 m 노드 {base.sum()}")
    for lo, hi in ((0.01, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.99)):
        m = base & (KB > lo) & (KB <= hi)
        if m.sum() < 5: continue
        w = mn[m]; dv = V1[m] - V0[m]
        rz = np.average(dv[:, 2] / (-g * h), weights=w); dx = np.average(dv[:, 0], weights=w)
        mom = (w[:, None] * dv).sum(0); fi = -(Fn[m]).sum(0) * h; gi = np.array([0, 0, -g * h * w.sum()])
        print(f"   KB ({lo},{hi}] n{m.sum():5d} 평균 KB {np.average(KB[m], weights=w):.2f} | 수직 변화 비 {rz:5.2f} 수평 x {dx:6.3f} m/s | Σ m ΔV x {mom[0]:+.3e} z {mom[2]:+.3e} | 중력 충격량 z {gi[2]:+.3e} 힘 충격량(걸음 끝 Fn) x {fi[0]:+.2e} z {fi[2]:+.2e}", flush=True)
