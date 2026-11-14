"""절 CM175 분석 (인스턴스 안 numpy; 판 디렉토리 <run>/esc·<run>/cm174 의 fNNN.npz): 바깥 모서리 층 (x>1.3, 바닥 위 0~14 cm) 에서
 (1) 입자 det F (프레임 시작·끝) 5·50 % 와 0.9 미만 몫 -> 처음으로 내려가는 프레임
 (2) 고착 바닥 노드 (sk<0.01) 와 그 바로 위 노드 (격자 +1) 의 한 걸음 (U0 -> 뉴턴 해 Y) 상대 수직 변위 (셀 폭 단위)
 (3) 장벽 힘 몫 (esc BG 합 대 전체 f_z 합, 장벽 노드 수)
 (4) 꼭짓점 점 몫이 큰 노드의 반복 1 뉴턴 걸음 (|NY[1]−NY[0]|, 셀 단위) -- 헤시안 공백 가설
사용: python3 -I exe/cm175_an.py <run> [--pn 0.48]"""
import argparse, glob, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--pn", type=float, default=0.48); a = ap.parse_args()
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for ep in sorted(glob.glob(os.path.join(a.run, "esc", "f*.npz"))):
    fr = int(os.path.basename(ep)[1:4]); e = np.load(ep)
    F0 = e["p0_F"].reshape(-1, 3, 3); F1 = e["p1_F"].reshape(-1, 3, 3); x0 = e["p0_x"]
    m = (x0[:, 0] > 1.3) & (x0[:, 2] < a.pn + 0.14) & (x0[:, 2] > a.pn)
    d0, d1 = np.linalg.det(F0[m]), np.linalg.det(F1[m])
    print(f"== f{fr}: 모서리 입자 {m.sum()} det F 시작 5/50% {np.percentile(d0,[5,50]).round(3)} (<0.9 몫 {(d0<0.9).mean():.3f}) 끝 5/50% {np.percentile(d1,[5,50]).round(3)} (<0.9 몫 {(d1<0.9).mean():.3f})")
    if "nid" not in e.files: continue
    nid = e["nid"].astype(np.int64); IP, A = e["IP"], e["A"]; nb1, nb2 = int(IP[7]), int(IP[8]); mn = e["mn"]
    cell = float(np.linalg.norm(A[:, 2]))
    pos = (np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]) @ A.T + e["U0"][0]
    sk, U0, Y = e["SK"][0], e["U0"][0], e["Y"][0]
    lay = (pos[:, 0] > 1.3) & (pos[:, 2] < a.pn + 0.14)
    idx = {int(n): i for i, n in enumerate(nid)}
    rel = []
    for i in np.where(lay & (sk < 0.01))[0]:
        j = idx.get(int(nid[i]) + 1)
        if j is not None and sk[j] > 0.99 and mn[j] > 0:
            rel.append(((Y[j] - U0[j]) - (Y[i] - U0[i]))[2] / cell)
    rel = np.array(rel)
    if len(rel):
        print(f"   (2) 고착 노드 위 노드 쌍 {len(rel)}: 한 걸음 상대 수직 변위 (셀 {cell*1000:.1f} mm 단위) 5/50/95% {np.percentile(rel,[5,50,95]).round(3)}, < −0.5 칸 {(rel<-0.5).sum()}, < −1 칸 {(rel<-1).sum()}")
    bg, bk = e["BG"][0], e["BK"][0]
    print(f"   (3) 층 노드 {lay.sum()} 중 장벽 노드 (bK>0) {(lay&(bk>0)).sum()}, Σ|bG| {np.abs(bg[lay]).sum():.3e}, Σ|f_전체_z| (esc F) {np.abs(e['Fn'][0][lay,2]).sum():.3e}, 고착 노드 {(lay&(sk<0.01)).sum()}")
    cp = os.path.join(a.run, "cm174", os.path.basename(ep))
    if os.path.exists(cp):
        c = np.load(cp)
        if "STq" in c.files and np.array_equal(c["nid"], e["nid"]):
            st, cb, NY = c["STq"], c["cb"], c["NY"]
            cbd = {tuple(v): i for i, v in enumerate(cb)}
            free = np.where(lay & (sk > 0.99) & (mn > 0))[0]
            ap_, stp = [], []
            for n in free:
                g = np.array([nid[n] // (nb1 * nb2), (nid[n] // nb2) % nb1, nid[n] % nb2])
                cs = [cbd.get(tuple(g - np.array([i, j, k])), -1) for i in range(3) for j in range(3) for k in range(3)]
                qs = np.concatenate([np.arange(x * 8, x * 8 + 8) for x in cs if x >= 0]) if any(x >= 0 for x in cs) else None
                if qs is None: ap_.append(np.nan); stp.append(0); continue
                ap_.append((st[0, qs] == 2).mean()); stp.append(abs(NY[1, n, 2] - NY[0, n, 2]) / cell)
            ap_, stp = np.array(ap_), np.array(stp)
            ok = ~np.isnan(ap_)
            for lo, hi in ((0, 0.01), (0.01, 0.5), (0.5, 1.01)):
                s = ok & (ap_ >= lo) & (ap_ < hi)
                if s.sum(): print(f"   (4) 자유 층 노드 꼭짓점 점 몫 [{lo:.2f},{hi:.2f}): {s.sum()} 개, 반복 1 걸음 |Δy_z|/셀 중앙 {np.median(stp[s]):.4f} 95% {np.percentile(stp[s],95):.4f} 최대 {stp[s].max():.3f}")
