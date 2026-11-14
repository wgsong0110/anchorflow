"""절 CM155: 칸을 크게 넓히는 점 (ln(JX/J0) > 0.05, 'big') 이 f10~12 에 처음 생기는 자리의 노드 (질량·변위·구속) 와 정답 같은 자리 변위장.
점 덤프 (cm151_wrap CM151_FR=9-13, 노드 U·mn·pm·kb 포함) + 정답 h5.  사용: python exe/cm155_an.py <판 디렉토리 (r2: 가능)> <h5 디렉토리 (r2: 가능)>
정답 변위장: 0.02 m 칸마다 정답 입자 걸음 변위 (x_f − x_{f−1}) 평균, 이웃 3x3x3 칸 평균 변위 z 의 최대−최소 (칸 안 상대 변위, 칸 늘림 척도) 대 우리 점의 칸 노드 변위 z 범위."""
import sys, os, io, subprocess
import numpy as np, h5py
run, h5d = sys.argv[1:3]
def rd(p): return io.BytesIO(subprocess.run(["rclone", "cat", p], capture_output=True, check=True).stdout) if p.startswith("r2:") else p
def gtx(f):
    with h5py.File(rd(h5d + f"/sim_{f:010d}.h5"), "r") as h: return np.array(h["x"]).T
V = 0.02; lo = np.array([0.4, 0.4, 0.4]); N = 70
off = np.array([(o // 9, (o // 3) % 3, o % 3) for o in range(27)])
def gtfield(f):
    a, b = gtx(f - 1), gtx(f); dz = b[:, 2] - a[:, 2]
    i = np.floor((a - lo) / V).astype(int); ok = ((i >= 0) & (i < N)).all(1)
    s = np.zeros((N, N, N)); c = np.zeros((N, N, N)); np.add.at(s, tuple(i[ok].T), dz[ok]); np.add.at(c, tuple(i[ok].T), 1)
    m = np.where(c > 0, s / np.maximum(c, 1), np.nan); mx = np.full((N, N, N), -np.inf); mn = np.full((N, N, N), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz_ in (-1, 0, 1):
                r = np.roll(m, (dx, dy, dz_), (0, 1, 2)); mx = np.fmax(mx, np.where(np.isnan(r), -np.inf, r)); mn = np.fmin(mn, np.where(np.isnan(r), np.inf, r))
    rng = np.where(np.isfinite(mx) & np.isfinite(mn), mx - mn, np.nan)
    return m, rng
print("f | 점 | big(>0.05) 점 | big 몫 | big 점: 노드 최소 질량 중앙 (평온 점) | 노드 질량 < 5e-6 가진 몫 (평온) | 칸 노드 변위 z 범위 mm 중앙 (평온) | 정답 같은 자리 이웃 칸 변위 범위 mm 중앙 (평온) | big 점 최소 kb 중앙 (평온) | 눌림 | big 점 높이 중앙 (전체)")
for f in range(9, 14):
    try: d = np.load(rd(run + f"/cm151_f{f}.npz"))
    except Exception: continue
    nb1, nb2 = [int(v) for v in d["nbs"]]; U = d["U"]; mn = d["mn"]; kb = d["kb"]; cb = d["cb"]
    ok = (d["mq"] > 0) & (d["J0"] > 0) & (d["JX"] > 0) & (d["JE"] > 0); q = np.nonzero(ok)[0]
    l0 = np.log(d["J0"][q]); lX = np.log(d["JX"][q]); x = d["xq"][q]; big = (lX - l0) > 0.05; calm = ~big
    base = cb[q >> 3]; ids = ((base[:, None, 0] + off[None, :, 0]) * nb1 + (base[:, None, 1] + off[None, :, 1])) * nb2 + (base[:, None, 2] + off[None, :, 2])
    mnn = mn[ids]; Uz = U[ids][:, :, 2]; Uzm = np.where(mnn > 0, Uz, np.nan); rngU = (np.nanmax(Uzm, 1) - np.nanmin(Uzm, 1)) * 1e3
    minm = np.where(mnn > 0, mnn, np.inf).min(1); lowfrac = (np.where(mnn > 0, mnn, np.inf).min(1) < 5e-6)
    kbm = kb[ids].min(1) if kb.size else np.full(len(q), np.nan)
    gm, gr = gtfield(f); i = np.clip(np.floor((x - lo) / V).astype(int), 0, N - 1); gR = gr[tuple(i.T)] * 1e3
    print(f"{f} | {len(q)} | {big.sum()} | {big.mean():.4f} | {np.median(minm[big]) if big.any() else 0:.2e} ({np.median(minm[calm]):.2e}) | {lowfrac[big].mean() if big.any() else 0:.3f} ({lowfrac[calm].mean():.3f}) | {np.nanmedian(rngU[big]) if big.any() else 0:.3f} ({np.nanmedian(rngU[calm]):.3f}) | {np.nanmedian(gR[big]) if big.any() else 0:.3f} ({np.nanmedian(gR[calm]):.3f}) | {np.nanmedian(kbm[big]) if big.any() else float('nan'):.3f} ({np.nanmedian(kbm[calm]):.3f}) | {(l0[big] < -3e-3).mean() if big.any() else 0:.3f} | {np.median(x[big, 2]) if big.any() else 0:.3f} ({np.median(x[:, 2]):.3f})")
