"""절 CM144 (CPU, 읽기만): escwhy 기록 (esc/fNNN.npz) 으로 앞면 바닥 막 노드 (바닥 위 < 0.13 m, x > 1.38) 의 걸음 수평 (x) 가속 a_dv = (V1-V0)/h 와 노드 방정식 항 -- 관성 (Y-Ut)/tau^2, 탄성+ghost Fn/m, 잔차 G0/m (질량 가중 평균) -- 과
접선 고착 (SK<1 · AN>0 노드 몫), 구속 kb 를 낸다. 사용: python exe/cm144_film.py <esc 디렉토리> F0 F1 (장벽 BG 는 바닥 수직이라 수평 0)."""
import os, sys
import numpy as np
esc = sys.argv[1]; F0 = int(sys.argv[2]); F1 = int(sys.argv[3]); h = 0.04; T2 = 0.268 * h * h
for f in range(F0, F1 + 1):
    p = os.path.join(esc, f"f{f:03d}.npz")
    if not os.path.exists(p): continue
    z = np.load(p); IP = z["IP"]; A = z["A"].astype(np.float64); nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8])
    i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
    X = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1).astype(np.float64) @ A.T
    mn = z["mn"].astype(np.float64); V0 = z["V0"][0]; V1 = z["V1"][0]; Y = z["Y"][0]; Ut = z["Ut"][0]; Fn = z["Fn"][0]; G0 = z["G0"][0]
    SK = z["SK"][0]; AN = z["AN"][0]; KB = z["KB"][0]; PM = z["PM"][0]
    film = (X[:, 0] > 1.38) & (X[:, 2] - 0.48 < 0.13) & (mn > 0) & (PM != 2)
    w = mn[film]; av = lambda a: float(np.average(a[film], weights=w)) if film.sum() else float("nan")
    a = (V1 - V0) / h
    print(f"f{f} 막 노드 {film.sum():5d} 질량합 {w.sum():.3e} | x 방향 a_dv {av(a[:,0]):7.2f}  관성 {av((Y-Ut)[:,0]/T2):8.2f}  탄성+ghost(-Fn/m) {av(-Fn[:,0]/mn):8.2f}  잔차 G0/m {av(G0[:,0]/mn):8.2f} | 접선 고착: SK<0.999 몫 {float((SK[film]<0.999).mean()):.2f} AN>0 몫 {float((AN[film]>0).mean()):.2f} 질량가중 SK {av(SK):.2f} | kb {av(KB):.2f} | z 방향 a_dv {av(a[:,2]):7.2f} 탄성 {av(-Fn[:,2]/mn):8.1f}", flush=True)

# ---- 상위 노드: 막 노드 중 x 방향 가속 a_dv 가 큰 순 (질량 > 중앙값 1e-3 배 이상만) ----
if len(sys.argv) > 4:
    for f in range(F0, F1 + 1):
        p = os.path.join(esc, f"f{f:03d}.npz")
        if not os.path.exists(p): continue
        z = np.load(p); IP = z["IP"]; A = z["A"].astype(np.float64); nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8])
        i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
        X = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1).astype(np.float64) @ A.T
        mn = z["mn"].astype(np.float64); V0 = z["V0"][0]; V1 = z["V1"][0]; Y = z["Y"][0]; Ut = z["Ut"][0]; Fn = z["Fn"][0]; G0 = z["G0"][0]
        SK = z["SK"][0]; AN = z["AN"][0]; PM = z["PM"][0]; med = np.median(mn[mn > 0])
        film = (X[:, 0] > 1.38) & (X[:, 2] - 0.48 < 0.13) & (mn > 1e-2 * med) & (PM != 2)
        a = (V1 - V0) / h; idx = np.where(film)[0]; top = idx[np.argsort(-a[idx, 0])[:int(sys.argv[4])]]
        mt = float(np.sum(mn[film] * (V1 - V0)[film, 0])); print(f"f{f} 막 (질량>1e-2 중앙값) 노드 {film.sum()} 총 x 운동량 변화 {mt:.3e} 양의 몫 {float(np.sum(mn[film]*np.maximum((V1-V0)[film,0],0))):.3e}")
        for n in top:
            print(f"   노드 x {X[n].round(3).tolist()} 질량비 {mn[n]/med:.2e} SK {SK[n]:.2f} AN {AN[n]:.1e} PM {PM[n]} | a_dv_x {a[n,0]:7.1f} 관성 {(Y-Ut)[n,0]/T2:8.1f} -Fn_x/m {-Fn[n,0]/mn[n]:9.1f} 잔차 {G0[n,0]/mn[n]:8.1f} | z: a_dv {a[n,2]:7.1f} -Fn_z/m {-Fn[n,2]/mn[n]:9.1f}")
