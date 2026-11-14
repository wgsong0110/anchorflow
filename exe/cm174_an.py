"""절 CM174 분석 (로컬 numpy): <run>/cm174/fNNN.npz (반복별 노드 힘 탄성·꼬리 분해·점 상태) + <run>/esc/fNNN.npz (걸음별 노드) 로
걸음 끝 위로 가장 빨라진 노드 (esc V1_z 상위) 의 반복별 y_z−Ũ_z·탄성 몫·꼬리 몫·전체 f_z (물리 힘 = −f), 둘레 점 상태 수를 낸다.
사용: python3 -I exe/cm174_an.py <run> [--top 6] [--frames 12:16]"""
import argparse, glob, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--top", type=int, default=6); ap.add_argument("--frames", default="")
a = ap.parse_args()
np.set_printoptions(precision=3, suppress=True, linewidth=220)
fs = sorted(glob.glob(os.path.join(a.run, "cm174", "f*.npz")))
for fp in fs:
    fr = int(os.path.basename(fp)[1:4])
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        if not lo <= fr <= hi: continue
    c = np.load(fp); ep = os.path.join(a.run, "esc", os.path.basename(fp))
    if not os.path.exists(ep) or "nid" not in c.files: continue
    e = np.load(ep)
    if "nid" not in e.files or not np.array_equal(e["nid"], c["nid"]): print(f"f{fr} 노드 집합 다름"); continue
    nid, IP, A = c["nid"].astype(np.int64), c["IP"], c["A"]; nt = int(c["nt"])
    nb1, nb2 = IP[7], IP[8]
    pos = (np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]) @ A.T
    mn = c["mn"]; Ut = e["Ut"][0]; V1 = e["V1"][0]; U0 = e["U0"][0]; Y = e["Y"][0]
    NY, NE, NT, NF, NG = c["NY"], c["NE"], c["NT"], c["NF"], c["NG"]
    st, dq, cb = c["STq"], c["DQ"], c["cb"]
    idx = {int(n): i for i, n in enumerate(nid)}
    # 노드 -> 둘레 점
    nq = st.shape[1]; cell = np.arange(nq) >> 3
    # 걸음 끝 위 이탈: 걸음 끝 y_z − Ũ_z (관성 예측 대비 위로 간 양) 와 V1_z
    up = Y[:, 2] - Ut[:, 2]
    print(f"== f{fr}: 노드 {len(nid)}, 점 {nq}, 반복 {nt}; 위 이탈 y−Ũ 최대 {up.max():.4f} m, V1_z 최대 {V1[:,2].max():.3f}, 반복별 점 상태 수 (탄성/꼬리/꼭짓점) "
          + " ".join(f"[{(st[i]==0).sum()}/{(st[i]==1).sum()}/{(st[i]==2).sum()}]" for i in range(nt)))
    print("   rls (노름², 기준², a, 받음, 돈, it):", c["rlh"][:nt].tolist() if "rlh" in c.files else "-")
    # 전체 합: 위 이탈 상위 노드
    sel = np.argsort(-up)[:a.top]
    # 둘레 점 상태 수: 노드 n 이 받는 칸 = 셀 c 의 cb + (0..2)^3
    cbd = {tuple(v): i for i, v in enumerate(cb)}
    for n in sel:
        g = np.array([nid[n] // (nb1 * nb2), (nid[n] // nb2) % nb1, nid[n] % nb2])
        cells = [cbd.get(tuple(g - np.array([i, j, k])), -1) for i in range(3) for j in range(3) for k in range(3)]
        cells = [x for x in cells if x >= 0]
        qs = np.concatenate([np.arange(x * 8, x * 8 + 8) for x in cells]) if cells else np.zeros(0, int)
        print(f"  노드 {nid[n]} x {pos[n] + U0[n]} 질량 {mn[n]:.2e} 외부 {int(e['W8'][0][n] >= 0)} sk {e['SK'][0][n]:.2f} pm {e['PM'][0][n]} 장벽 bK {e['BK'][0][n]:.1e} an {e['AN'][0][n]:.1e} 갱신 {e['UL'][0][n]}  y−Ũ_z {up[n]:+.4f} V1_z {V1[n,2]:+.2f} 둘레 칸 {len(cells)}")
        for it in range(nt):
            s_ = st[it, qs]
            print(f"     it{it}: y−Ũ_z {NY[it,n,2]-Ut[n,2]:+.4f}  f_탄성_z {NE[it,n,2]:+.3e} f_꼬리_z {NT[it,n,2]:+.3e} f_전체_z {NF[it,n,2]:+.3e} g_z {NG[it,n,2]:+.3e}  점 (탄/꼬/꼭) {(s_==0).sum()}/{(s_==1).sum()}/{(s_==2).sum()} 최소 detX {dq[it,qs].min() if len(qs) else 0:.2f}")
    # 상위 이탈 노드 군 합: 마지막 반복 힘 구성 (물리 힘 = −f, 위 + )
    top = np.argsort(-up)[:200]
    for it in (0, nt - 1):
        print(f"   상위 200 노드 it{it}: 물리 위 힘 합 탄성 {-NE[it,top,2].sum():+.3e} 꼬리 {-NT[it,top,2].sum():+.3e} 전체(장벽 포함) {-NF[it,top,2].sum():+.3e}, 질량 합 {mn[top].sum():.3e} (중력 {mn[top].sum()*9.8:.3e})")
