"""절 CM151 (인스턴스): Kacdhnz 판 (exe/cm151_wrap.py 덤프 cm151_f{10,20,30}.npz) 가우스 점별 ln J 와 압력 대 정답 h5 입자.
(1) 층별 (바닥 위 높이) ln J 분포 분위수·히스토그램, 눌린 점 (ln J0 < -3e-3) 의 위치·칸 입자 수·바닥 거리,
(2) 눌린 점의 한 걸음 변화 ln(JX/J0) (노드 변위장이 칸을 늘린 몫), 평형 무게 압력 (rho g 깊이) 대 시작 압력 -K ln J0.
사용: python exe/cm151_an.py <판 디렉토리> <h5 디렉토리> <출력 디렉토리>"""
import sys, os
import numpy as np, h5py
run, h5d, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
E, NU, RHO, G = 5e7, 0.3, 2000.0, 9.8; K = E / (3 * (1 - 2 * NU))
def gt(f):
    with h5py.File(os.path.join(h5d, f"sim_{f:010d}.h5"), "r") as h: return np.array(h["x"]).T, np.array(h["f_tensor"]).T.reshape(-1, 3, 3).astype(np.float64)
def depth(x, ref):                                        # 같은 (x,y) 0.04 칸의 최고 z 와의 차 (ref 점들 기준)
    b = lambda p: (np.floor(p[:, 0] / 0.04).astype(int) * 1000 + np.floor(p[:, 1] / 0.04).astype(int))
    kr = b(ref); top = {}
    for k, z in zip(kr, ref[:, 2]): top[k] = max(top.get(k, -9), z)
    return np.array([top.get(k, z) - z for k, z in zip(b(x), x[:, 2])])
Q = (1, 10, 50, 90, 99)
def fmt(a): return " ".join(f"{v:+.4f}" for v in np.percentile(a, Q))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, f in enumerate((10, 20, 30)):
    d = np.load(os.path.join(run, f"cm151_f{f}.npz")); xq = d["xq"]; ok = (d["mq"] > 0) & (d["J0"] > 0) & (d["JX"] > 0) & (d["JE"] > 0)
    xq = xq[ok]; l0 = np.log(d["J0"][ok]); lX = np.log(d["JX"][ok]); lE = np.log(d["JE"][ok]); pc = d["pc"][ok]; m = d["mq"][ok]
    xg, Fg = gt(f); Jg = np.linalg.det(Fg); lg = np.log(np.abs(Jg) + 1e-12)
    zf = min(xg[:, 2].min(), xq[:, 2].min()); hq = xq[:, 2] - zf; hg = xg[:, 2] - zf
    print(f"==== f{f}: 우리 가우스 점 {ok.sum()}, 정답 입자 {len(xg)}, 바닥 z {zf:.3f}")
    print(f"질량 가중 평균 ln J: 우리 시작 {np.average(l0, weights=m):+.5f} 물리 뒤 {np.average(lX, weights=m):+.5f} 반환 뒤 {np.average(lE, weights=m):+.5f} | 정답 {lg.mean():+.5f}")
    print(f"분위수 {Q}: 우리 시작 {fmt(l0)} | 정답 {fmt(lg)}")
    ed = np.quantile(np.concatenate([hq, hg]), [0, .25, .5, .75, 1.0001])
    for a, b in zip(ed[:-1], ed[1:]):
        mq_, mg_ = (hq >= a) & (hq < b), (hg >= a) & (hg < b)
        print(f"  높이 {a:.2f}~{b:.2f} m: 우리 평균 {l0[mq_].mean():+.5f} 중앙 {np.median(l0[mq_]):+.5f} 점 {mq_.sum()} | 정답 평균 {lg[mg_].mean():+.5f} 중앙 {np.median(lg[mg_]):+.5f}")
    P = l0 < -3e-3
    print(f"눌린 점 (ln J0 < -3e-3): {P.sum()} ({P.mean():.3f}); 높이 평균 {hq[P].mean():.3f} (전체 {hq.mean():.3f}), 칸 입자 수 평균 {pc[P].mean():.2f} (전체 {pc.mean():.2f}), 입자 <3 인 몫 {np.mean(pc[P] < 3):.3f} (전체 {np.mean(pc < 3):.3f}), 바닥 0.05 m 안 몫 {np.mean(hq[P] < 0.05):.3f} (전체 {np.mean(hq < 0.05):.3f})")
    print(f"  눌린 점 걸음 변화 ln(JX/J0): 평균 {np.average((lX - l0)[P], weights=m[P]):+.5f} 분위수 {fmt((lX - l0)[P])}; 안 눌린 점 {np.average((lX - l0)[~P], weights=m[~P]):+.5f}")
    print(f"  눌린 점 반환 뒤 ln(JE/JX): {np.average((lE - lX)[P], weights=m[P]):+.5f}; JX>1 인 몫 {np.mean(lX[P] > 0):.3f}")
    dp = depth(xq, xq); pw = RHO * G * dp; p0 = -K * l0
    print(f"  평형 무게 압력 중앙 {np.median(pw)/1e3:.1f} kPa 대 시작 압력 -K ln J0 중앙 {np.median(p0)/1e3:.1f} kPa (눌린 점 {np.median(p0[P])/1e3:.1f} kPa, 무게 {np.median(pw[P])/1e3:.1f} kPa); 정답 무게 압력 K|lnJ| 중앙 {np.median(K*np.abs(lg))/1e3:.2f} kPa")
    for lo, hi in ((0, .1), (.1, .25), (.25, .5), (.5, 9)):
        s = (dp >= lo) & (dp < hi)
        if s.sum() > 10 and (s & P).sum() > 0: print(f"  깊이 {lo}~{hi} m: 점 {s.sum()} 눌림 몫 {(s & P).mean() / s.mean():.3f} 평균 ln J0 {l0[s].mean():+.5f} 무게 ln J {-(pw[s].mean())/K:+.5f} 걸음 변화 {np.mean((lX - l0)[s]):+.5f}")
    ax[0, i].hist(l0, bins=np.linspace(-0.03, 0.03, 120), alpha=.6, density=True, label="우리 가우스 점 (시작)"); ax[0, i].hist(np.clip(lg, -0.03, 0.03), bins=np.linspace(-0.03, 0.03, 120), alpha=.6, density=True, label="정답 입자")
    ax[0, i].set_title(f"f{f} ln J"); ax[0, i].set_yscale("log"); ax[0, i].legend()
    ax[1, i].hist((lX - l0)[P], bins=np.linspace(-0.03, 0.06, 100), alpha=.7); ax[1, i].set_title(f"f{f} 눌린 점의 한 걸음 변화 ln(JX/J0)")
plt.tight_layout(); plt.savefig(os.path.join(od, "cm151_hist.png"), dpi=80)
