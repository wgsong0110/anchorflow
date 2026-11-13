"""절 CM137 (CPU, 읽기만): 걸음 사이 전달 (P2G/G2P) 에서 속도·운동량을 잃는지 -- escwhy 기록으로 방출된 높은 영역의 질량 가중 평균 v_z 를 세 지점에서 비교.
 (a) 걸음 끝 노드 속도 V1 (rt_ga_end 뒤)  (b) G2P 뒤 입자 속도 p1_v  (c) 다음 프레임 시작 P2G 노드 속도 mv/mn  (d) 입자 속도 p0_v (이 프레임 시작, 앞 프레임 G2P 값)
사용: python exe/cm137_transfer.py <esc 디렉토리> F0 F1"""
import os, sys
import numpy as np
esc = sys.argv[1]; F0 = int(sys.argv[2]); F1 = int(sys.argv[3]); h = 0.04
def load(f):
    z = np.load(os.path.join(esc, f"f{f:03d}.npz")); IP = z["IP"]; A = z["A"].astype(np.float64)
    nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8]); i = nid // (nb1 * nb2); j = (nid // nb2) % nb1; k = nid % nb2
    X = np.stack([i + IP[0], j + IP[1], k + IP[2]], 1).astype(np.float64) @ A.T
    L = (((i + IP[0]) + 1000) * 4096 + (j + IP[1]) + 1000) * 4096 + (k + IP[2]) + 1000     # 상자가 프레임마다 옮겨지므로 격자 좌표로 대응
    return z, L, X
print("f | 방출된 높은 영역 (앞면 +6 cm 너머, z > 0.8): 노드 V1_z(걸음 끝) | 입자 p1_v_z (G2P 뒤) | 입자 p0_v_z (시작) | 다음 프레임 P2G 노드 속도 mv/mn z | 운동량 합 m v_z: 노드 V1 / 입자 p1 / 다음 P2G (단위 임의)")
for f in range(F0, F1 + 1):
    if not (os.path.exists(os.path.join(esc, f"f{f:03d}.npz")) and os.path.exists(os.path.join(esc, f"f{f+1:03d}.npz"))): continue
    z, nid, X = load(f); z2, nid2, X2 = load(f + 1)    # nid = 격자 좌표 키
    front = 1.5 - f * h / 1.5
    mn = z["mn"].astype(np.float64); KB = z["KB"][0]; PM = z["PM"][0]
    g = (X[:, 0] > front + 0.06) & (X[:, 2] > 0.8) & (mn > 0) & (PM != 2)
    w = mn[g]; V1 = z["V1"][0][g, 2]
    p0x = z["p0_x"]; sel = (p0x[:, 2] > 0.8) & (p0x[:, 0] > front + 0.06)
    p1 = z["p1_v"][sel, 2]; p0 = z["p0_v"][sel, 2]
    # 다음 프레임 같은 노드의 P2G 속도
    d2 = {int(n): t for t, n in enumerate(nid2)}
    idx = np.array([d2.get(int(n), -1) for n in nid[g]]); ok = idx >= 0
    mn2 = z2["mn"].astype(np.float64); mv2 = z2["mv"].astype(np.float64)
    v2 = mv2[idx[ok], 2] / np.where(mn2[idx[ok]] > 0, mn2[idx[ok]], 1)
    wn = mn[g][ok]
    mp = w.sum() / max(sel.sum(), 1)                     # 입자 질량 균일 가정 (노드 질량 합 / 입자 수)
    print(f"f{f:02d} 노드 V1 {np.average(V1, weights=w):7.3f} | 입자 p1 {p1.mean():7.3f} | 입자 p0 {p0.mean():7.3f} | 다음 P2G {np.average(v2, weights=wn):7.3f} (같은 노드 V1 {np.average(z['V1'][0][g][ok, 2], weights=wn):7.3f}) | 입자수 {sel.sum()} 노드질량 {w.sum():.3f}")
