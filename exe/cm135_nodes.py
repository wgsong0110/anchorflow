"""절 CM135 (CPU, 읽기만): escwhy 기록 (esc/fNNN.npz) 으로 방출된 높은 자유 노드의 걸음 가속도를 항별로 가르고, 구속·부분구속 노드와 칸 공유 여부(격자 거리)별로 비교.
사용: python exe/cm135_nodes.py <esc 디렉토리> <wolf.pt 를 cm130_ptload 로 읽은 정답 x 가 든 경로> [F0 F1]
노드 세계 좌표 = A · (i+IP0, j+IP1, k+IP2), nid = (i nb1 + j) nb2 + k (IP[6..8] = nb).
항: a_dv = (V1 - V0)/h 의 z (걸음 평균 가속도), A1 (갱신 뒤 가속도), (Y - Ut)/h² 근사 대신 힘 항 Fn/m (탄성+ghost 등 f), BG/m (장벽 기울기), G0/m (마지막 반복 시작 잔차).
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cm130_ptload as L

esc = sys.argv[1]; gtp = sys.argv[2]
F0 = int(sys.argv[3]) if len(sys.argv) > 3 else 5; F1 = int(sys.argv[4]) if len(sys.argv) > 4 else 14
GT = np.array(L.load(gtp)["x"][:51], dtype=np.float32)
G = 9.8
print("frame | 방출된 높은 자유 노드(z>0.8, 앞면에서 6 cm 넘게 방출된 쪽(x 큼), KB>0.99, 고정 노드(PM 2) 아님): n, a_dv_z, A1_z, Fn_z/m, BG_z/m, G0_z/m | 입자 가속 z: 판 / 정답 (같은 입자)")
for f in range(F0, F1 + 1):
    p = os.path.join(esc, f"f{f:03d}.npz")
    if not os.path.exists(p): continue
    z = np.load(p)
    IP = z["IP"]; A = z["A"].astype(np.float64); h = float(z["h"]); K = int(z["K"])
    nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8])
    i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
    idx = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1)
    X = idx.astype(np.float64) @ A.T
    mn = z["mn"].astype(np.float64)
    U0 = z["U0"][0]; V0 = z["V0"][0]; V1 = z["V1"][0]; A1 = z["A1"][0]; Fn = z["Fn"][0]; BG = z["BG"][0]; G0 = z["G0"][0]
    SK = z["SK"][0]; AN = z["AN"][0]; KB = z["KB"][0]
    t = f * 0.04
    front = 1.5 - (t / 1.5)                       # 방출 앞면 x (cfg: 1.5 -> 0.5, 1.5 s)
    xn = X[:, 0]; zn = X[:, 2]
    PM = z["PM"][0]
    rel = (xn > front + 0.06) & (zn > 0.8) & (mn > 0) & (KB > 0.99) & (PM != 2) & (AN == 0)      # 방출된 쪽 = x 큰 쪽
    # 칸 공유: 구속·부분구속 노드 (KB<0.99 | SK<0.999 | AN>0) 까지 격자 체비셰프 거리
    cons = (KB < 0.99) | (AN > 0)                                   # KB = 구속 인자 (1 자유, 0 완전 구속)
    ci = idx[cons]
    d = np.full(len(idx), 99.)
    nb0 = int(IP[6]); vol = np.zeros((nb0, nb1, nb2), bool); loc = np.stack([i, j, k], 1)
    vol[loc[cons, 0], loc[cons, 1], loc[cons, 2]] = True
    dist = np.full(vol.shape, 99.); cur = vol.copy(); dist[cur] = 0
    for r in range(1, 9):                                       # 체비셰프 거리 = 3x3x3 팽창 반복
        nxt = cur.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nxt |= np.roll(cur, (dx, dy, dz), (0, 1, 2))
        dist[nxt & ~cur] = np.minimum(dist[nxt & ~cur], r); cur = nxt
    d = dist[loc[:, 0], loc[:, 1], loc[:, 2]]
    def mm(a, m): return float(np.average(a[m, 2], weights=mn[m])) if m.sum() else float("nan")
    a_dv = (V1 - V0) / h
    line = f"f{f:02d} n{rel.sum():5d} 구속 노드 {cons.sum():5d} | "
    for lo, hi in [(0, 1.5), (1.5, 3.5), (3.5, 6.5), (6.5, 99)]:
        m = rel & (d >= lo) & (d < hi)
        line += f"거리[{lo},{hi}) n{m.sum():5d} a_dv {mm(a_dv, m):6.2f} A1 {mm(A1, m):6.2f} Fn/m {np.average(Fn[m,2]/mn[m], weights=mn[m]) if m.sum() else float('nan'):7.2f} BG/m {np.average(BG[m]/mn[m], weights=mn[m]) if m.sum() else float('nan'):7.2f} | "
    # 입자: 방출된 높은 입자 (z>0.9, x<front-0.06) 가속 z
    p0v = z["p0_v"][:, 2]; p1v = z["p1_v"][:, 2]; p0x = z["p0_x"]
    sel = (p0x[:, 2] > 0.9) & (p0x[:, 0] > front + 0.06)
    a_p = (p1v - p0v)[sel].mean() / h if sel.sum() else float("nan")
    vg0 = (GT[f - 1][sel, 2] - GT[f - 2][sel, 2]) / 0.04; vg1 = (GT[f][sel, 2] - GT[f - 1][sel, 2]) / 0.04
    a_g = (vg1 - vg0).mean() / 0.04 if sel.sum() else float("nan")
    print(line + f"입자 n{sel.sum()} 판 {a_p:6.2f} 정답 {a_g:6.2f}", flush=True)

# ---- 항별 분해 (수렴 판용; 노드 방정식 (m/tau^2)(Y - Ut) + T f = 0, tau^2/h^2 = 0.268 = 중력 계수 적합값 (k5 f12 free 노드 Ut - U0 - h V0 = c1 A0 + tau^2 g)) ----
if os.environ.get("CM135_TERMS"):
    T2 = 0.268 * 0.04 ** 2
    print("\n항별 분해 (질량 가중 평균, z 성분 m/s^2): 관성 (Y-Ut)/tau^2 | 탄성 -Fn/m | 장벽 -BG/m | 잔차 G0/m | 중력 -9.8 포함은 Ut 안")
    for f in range(F0, F1 + 1):
        z = np.load(os.path.join(esc, f"f{f:03d}.npz")); IP = z["IP"]; A = z["A"].astype(np.float64)
        nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8]); i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
        X = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1).astype(np.float64) @ A.T
        front = 1.5 - f * 0.04 / 1.5; mn = z["mn"].astype(np.float64); KB = z["KB"][0]; PM = z["PM"][0]; AN = z["AN"][0]
        for lab, cond in [("자유 높은 노드 (KB>0.99)", (KB > 0.99)), ("부분구속 (0<KB<0.99)", (KB > 0.01) & (KB <= 0.99))]:
            m = (X[:, 0] > front - 0.06) & (X[:, 2] > 0.8) & (mn > 0) & cond & (PM != 2) & (AN == 0)
            if m.sum() < 5: continue
            w = mn[m]; av = lambda a: float(np.average(a[m], weights=w))
            Y = z["Y"][0][:, 2]; Ut = z["Ut"][0][:, 2]; Fz = z["Fn"][0][:, 2]; G0 = z["G0"][0][:, 2]; BG = z["BG"][0]
            print(f"f{f:02d} {lab} n{m.sum():5d} 질량 {w.sum():.3f}: 관성 {av((Y-Ut)/T2):6.2f} | 탄성 {av(-Fz/mn):6.2f} | 장벽 {av(-BG/mn):6.2f} | G0/m {av(G0/mn):7.2f} | A0 {av(z['A0'][0][:,2]):6.2f} A1 {av(z['A1'][0][:,2]):6.2f} | KB 평균 {av(KB):.2f}")
