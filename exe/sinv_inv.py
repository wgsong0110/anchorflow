"""절 CM128 뒤집힌 가우스 점 집계 (로컬, numpy 만): escwhy 기록 <run>/esc/fNNN.npz 의 걸음별 QS (특이값 3·DP 상태·det X) 로
프레임·걸음마다 det X < 0 점 수, 그중 꼭짓점 (상태 2) 몫, 전체 꼭짓점 부피 몫, 관성 예측에서 1.5 칸 (0.03 m) 넘게 벗어난 노드 수와
그 노드 둘레 가우스 점의 det X 최저 중앙 (exe/escwhy_apex.py 와 같은 이웃 정의).
사용: python3 exe/sinv_inv.py <run> [프레임 a:b]"""
import glob
import os
import sys

import numpy as np

run = sys.argv[1]
fa, fb = (int(t) for t in sys.argv[2].split(":")) if len(sys.argv) > 2 else (0, 999)
for p in sorted(glob.glob(os.path.join(run, "esc", "f*.npz"))):
    f = int(p[-7:-4])
    if not fa <= f <= fb:
        continue
    z = np.load(p)
    if "QS" not in z.files:
        continue
    Vq = z["Vq"]; on = Vq > 0
    IP = z["IP"]; nb1, nb2 = int(IP[7]), int(IP[8]); nid = z["nid"].astype(np.int64)
    nrel = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1)
    cb = z["cb"].astype(np.int64); nc = len(cb)
    key = lambda a: (a[:, 0] * 4096 + a[:, 1]) * 4096 + a[:, 2]
    ck = key(cb); order = np.argsort(ck); cks = ck[order]
    row = []
    for k in range(z["QS"].shape[0]):
        QS = z["QS"][k]; st = QS[:, 3].astype(int); dX = QS[:, 4]
        inv = on & (dX < 0)
        nap = int((inv & (st == 2)).sum())
        fap = float(Vq[on & (st == 2)].sum() / Vq[on].sum())
        dy = np.linalg.norm(z["Y"][k] - z["Ut"][k], axis=1); big = dy > 0.03
        minJ = np.full(len(nid), 9.0); Vap = np.zeros(len(nid)); Vall = np.zeros(len(nid))
        dXc = dX.reshape(-1, 8); Vc = Vq.reshape(-1, 8); stc = st.reshape(-1, 8)
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    c = nrel - np.array([di, dj, dk]); kk = key(c); j = np.clip(np.searchsorted(cks, kk), 0, nc - 1); ok = cks[j] == kk; ci = order[j]
                    minJ = np.minimum(minJ, np.where(ok, np.where(Vc[ci] > 0, dXc[ci], 9.0).min(1), 9.0))
                    v = np.where(ok[:, None], Vc[ci], 0); Vall += v.sum(1); Vap += (v * (stc[ci] == 2)).sum(1)
        fa_n = np.where(Vall > 0, Vap / np.maximum(Vall, 1e-30), np.nan)
        row.append(f"k{k}: det<0 {int(inv.sum())} (꼭짓점 {nap}) · 꼭짓점 부피 {fap:.2f} · |y−Ũ|>0.03 노드 {int(big.sum())}"
                   + (f" (둘레 꼭짓점 몫 중앙 {np.nanmedian(fa_n[big]):.2f}, 둘레 det 최저 중앙 {np.median(minJ[big]):.2f}, det<0 둘레 몫 {(minJ[big] < 0).mean():.2f})" if big.any() else ""))
    print(f"f{f}: " + " | ".join(row), flush=True)
