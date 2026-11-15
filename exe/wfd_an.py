"""wolffix 진단 집계 (11-15, CPU 읽기만): <판>/wfd/f*.npz (lib/anchorflow/fem_wfdiag.py) -> 뉴턴 반복별 잔차 분해.
사용: python exe/wfd_an.py <판 디렉토리> [프레임 ...]
 g = −g0buf (= Mi η + T f), 힘 f = 장벽 전 (F0: 탄성·고스트·구속) + 장벽 (F1 − F0), 관성 몫 = g − f (T 무시 근사; 고착 노드는 T 로 성분이 빠져 근사).
 노드 묶음: bar (장벽 힘 ≠ 0), light (질량 < 1e-2 x 질량 > 0 노드 중앙값), fix (pm 2 = 고정), 나머지 body. 몫 = 그 묶음 |g|² / 전체.
 상위 노드: 위치 x = A L + U (걸음 시작), 질량/중앙값, |g|, |f탄성|, |f장벽|, |관성|."""
import glob, os, sys
import numpy as np

D = sys.argv[1]
want = set(int(a) for a in sys.argv[2:])
for fn in sorted(glob.glob(os.path.join(D, "wfd", "f*.npz"))):
    f_ = int(os.path.basename(fn)[1:4])
    if want and f_ not in want:
        continue
    z = np.load(fn)
    mn, pm, sk = z["mn"], z["pm"], z["sk"]
    med = np.median(mn[mn > 0]) if (mn > 0).any() else 1.0
    pos = z["L"] @ z["A"].T + z["U"]
    nt, K = int(z["nt"]), int(z["K"])
    G, F0, F1, KK, DG = z["G"], z["F0"], z["F1"], z["KK"], z["DG"]
    print(f"== 프레임 {f_}  노드 {len(mn)} (질량>0 {int((mn>0).sum())}, 고정 pm2 {int((pm==2).sum())}, light<1e-2med {int(((mn>0)&(mn<1e-2*med)).sum())})  K {K}")
    for i in range(min(K, 4) * nt):
        g2tot = float(DG[2 * i, 1])
        if g2tot == 0:
            continue
        g = -G[i].astype(np.float64); fe = F0[i].astype(np.float64); fb = (F1[i] - F0[i]).astype(np.float64)
        gi = g - (fe + fb)
        gn2 = (g * g).sum(1)
        bar = np.abs(fb).sum(1) > 0; light = (mn > 0) & (mn < 1e-2 * med); fix = pm == 2
        body = ~(bar | light | fix)
        tot = gn2.sum() + 1e-300
        sh = dict(bar=gn2[bar].sum() / tot, light=gn2[light & ~bar].sum() / tot, fix=gn2[fix & ~bar & ~light].sum() / tot, body=gn2[body].sum() / tot)
        print(f" 걸음 {i // nt} 반복 {i % nt}: |g|² {tot:.3e} (DG {g2tot:.3e}) 장벽보폭 k {int(KK[i])} SC14 {z['SC'][i][14]:.4g} | 몫 " + " ".join(f"{k} {v:.2f}" for k, v in sh.items())
              + f" | Σ|f탄성|² {(fe*fe).sum():.2e} Σ|f장벽|² {(fb*fb).sum():.2e} Σ|관성|² {(gi*gi).sum():.2e} | 장벽 노드 {int(bar.sum())}")
        top = np.argsort(-gn2)[:8]
        for j in top:
            print(f"    노드 {int(z['idx'][j])} x ({pos[j,0]:.3f},{pos[j,1]:.3f},{pos[j,2]:.3f}) m/med {mn[j]/med:.2e} pm {int(pm[j])} sk {sk[j]:.2f} |g| {np.sqrt(gn2[j]):.3e} "
                  f"|f탄성| {np.linalg.norm(fe[j]):.3e} |f장벽| {np.linalg.norm(fb[j]):.3e} |관성| {np.linalg.norm(gi[j]):.3e} g {np.round(g[j], 1).tolist()}")
