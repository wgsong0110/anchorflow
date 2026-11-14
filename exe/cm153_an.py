"""절 CM153: 눌린 점 (ln J0 < -3e-3) 이 처음 생기는 걸음·위치.  점 덤프 (cm151_wrap CM151_FR=1-9) + 정답 h5 입자의 방출 프레임 (처음 |dz| > 0.5 mm) 로
점마다 '정답에서 아직 안 풀린 (잡힌) 입자 몫' hv (0.02 m 칸), 잡힌 칸까지 거리, 칸 입자 수 pc 별로 눌림 몫·걸음 안 ln J 변화 (시작 -> 물리 -> 반환).
사용: python exe/cm153_an.py <판 디렉토리 (r2: 가능)> <h5 디렉토리 (r2: 가능)> <출력 디렉토리>  -- 로컬에서 메모리로만 읽음"""
import sys, os, io, subprocess
import numpy as np, h5py
run, h5d, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
def rd(p): return io.BytesIO(subprocess.run(["rclone", "cat", p], capture_output=True, check=True).stdout) if p.startswith("r2:") else p
def gtx(f):
    with h5py.File(rd(h5d + f"/sim_{f:010d}.h5"), "r") as h: return np.array(h["x"]).T
FRS = [f for f in range(1, 10) if subprocess.run(["rclone", "lsf", run + f"/cm151_f{f}.npz"], capture_output=True, text=True).stdout.strip()] if run.startswith("r2:") else [f for f in range(1, 10) if os.path.exists(run + f"/cm151_f{f}.npz")]
X = [gtx(f) for f in range(0, max(FRS) + 1)]
rel = np.full(len(X[0]), 99)
for f in range(1, len(X)):
    m = (rel == 99) & (np.abs(X[f][:, 2] - X[0][:, 2]) > 5e-4); rel[m] = f
print("정답 방출 프레임별 입자 수:", {f: int((rel == f).sum()) for f in range(1, len(X))}, "안 풀림", int((rel == 99).sum()))
V = 0.02; lo = np.array([0.4, 0.4, 0.4]); N = 70
def grid(x, w):
    i = np.floor((x - lo) / V).astype(int); ok = ((i >= 0) & (i < N)).all(1)
    g = np.zeros((N, N, N)); np.add.at(g, tuple(i[ok].T), w[ok]); return g
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
fig, ax = plt.subplots(2, 3, figsize=(15, 9)); shown = [f for f in (1, 3, 5, 9) if f in FRS][:3]
print("f | 점 | 눌림 몫 | 잡힌 칸 (hv>0.5) 눌림 몫 (점 몫) | 잡힌 칸 2칸 안 눌림 몫 (점 몫) | 자유 눌림 몫 | pc 1~2 눌림 몫 (점 몫) | 눌린 점 lnJ0 -> 물리 뒤 -> 반환 뒤 | 안 눌린 점 걸음 변화")
for f in FRS:
    d = np.load(rd(run + f"/cm151_f{f}.npz")); ok = (d["mq"] > 0) & (d["J0"] > 0) & (d["JX"] > 0) & (d["JE"] > 0)
    x = d["xq"][ok]; l0 = np.log(d["J0"][ok]); lX = np.log(d["JX"][ok]); lE = np.log(d["JE"][ok]); pc = d["pc"][ok]; P = l0 < -3e-3
    xp = X[f - 1]; held = rel >= f
    gh = grid(xp[held], np.ones(held.sum())); gt = grid(xp, np.ones(len(xp))); hvg = gh / np.maximum(gt, 1)
    hold = (hvg > 0.5) & (gt > 0)
    dist = np.full((N, N, N), 99.0)
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            for dz in range(-4, 5):
                s = np.roll(hold, (dx, dy, dz), (0, 1, 2)); dist = np.where(s, np.minimum(dist, np.sqrt(dx * dx + dy * dy + dz * dz)), dist)
    i = np.clip(np.floor((x - lo) / V).astype(int), 0, N - 1); hv = hvg[tuple(i.T)]; ds = dist[tuple(i.T)]
    cls = {"held": hv > 0.5, "near": (hv <= 0.5) & (ds <= 2.0), "free": (hv <= 0.5) & (ds > 2.0)}
    r = {k: (P[m].mean() if m.any() else float("nan"), m.mean()) for k, m in cls.items()}; sp = (pc <= 2)
    print(f"{f} | {ok.sum()} | {P.mean():.4f} | {r['held'][0]:.4f} ({r['held'][1]:.3f}) | {r['near'][0]:.4f} ({r['near'][1]:.3f}) | {r['free'][0]:.4f} | {P[sp].mean():.4f} ({sp.mean():.3f}) | {l0[P].mean() if P.any() else 0:+.4f} {lX[P].mean() if P.any() else 0:+.4f} {lE[P].mean() if P.any() else 0:+.4f} | {np.mean((lX - l0)[~P]):+.5f} {np.mean((lE - lX)[~P]):+.5f}")
    if f in shown:
        a = ax[0, shown.index(f)]; a.scatter(x[~P, 0][::20], x[~P, 2][::20], s=1, c="lightgray"); a.scatter(x[P, 0], x[P, 2], s=2, c=np.where(cls["held"][P], "red", np.where(cls["near"][P], "orange", "blue")))
        a.set_title(f"f{f} x-z pressed (red held, orange near, blue free)"); b = ax[1, shown.index(f)]; b.scatter(x[~P, 0][::20], x[~P, 1][::20], s=1, c="lightgray"); b.scatter(x[P, 0], x[P, 1], s=2, c="k"); b.set_title(f"f{f} x-y pressed")
plt.tight_layout(); plt.savefig(os.path.join(od, "cm153_map.png"), dpi=70)
