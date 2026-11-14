"""절 CM152 추적 (exe/cm151_wrap.py CM151_FR=5-22 덤프): 프레임별 칸이 늘어남 ln(JX/J0) 과 눌린 점 (ln J0 < -3e-3) 몫·위치.  사용: python exe/cm152_an.py <판 디렉토리 (r2: 가능)> <출력 로그>"""
import sys, os, io, subprocess, glob
import numpy as np
run = sys.argv[1]
def rd(p): return io.BytesIO(subprocess.run(["rclone", "cat", p], capture_output=True, check=True).stdout) if p.startswith("r2:") else p
def names():
    if run.startswith("r2:"):
        return sorted(int(l[7:-4]) for l in subprocess.run(["rclone", "lsf", run], capture_output=True, text=True).stdout.split() if l.startswith("cm151_f") and l.endswith(".npz"))
    return sorted(int(os.path.basename(f)[7:-4]) for f in glob.glob(run + "/cm151_f*.npz"))
print("f  | 점 | 눌림 몫 | 눌린 점 평균 lnJ0 | 전체 ln(JX/J0) 평균 · 90% · 99% | 늘어남>0.01 몫 | 그 점 평균 높이 | 눌린 점 평균 높이 · 바닥5cm 몫 (전체) | 늘어난 점 중 눌린 몫 | 시작 평균 lnJ0 · 물리 뒤 · 반환 뒤")
zf = None
for f in names():
    d = np.load(rd(run + f"/cm151_f{f}.npz")); ok = (d["mq"] > 0) & (d["J0"] > 0) & (d["JX"] > 0) & (d["JE"] > 0)
    xq = d["xq"][ok]; l0 = np.log(d["J0"][ok]); lX = np.log(d["JX"][ok]); lE = np.log(d["JE"][ok]); m = d["mq"][ok]
    if zf is None: zf = xq[:, 2].min()
    h = xq[:, 2] - zf; dl = lX - l0; P = l0 < -3e-3; Ex = dl > 0.01
    w = lambda a, mk=None: np.average(a if mk is None else a[mk], weights=m if mk is None else m[mk]) if (mk is None or mk.any()) else float("nan")
    print(f"{f:2d} | {ok.sum()} | {P.mean():.3f} | {w(l0, P):+.4f} | {w(dl):+.4f} {np.percentile(dl, 90):+.4f} {np.percentile(dl, 99):+.4f} | {Ex.mean():.3f} | {h[Ex].mean() if Ex.any() else float('nan'):.3f} | {h[P].mean() if P.any() else float('nan'):.3f} {np.mean(h[P] < 0.05) if P.any() else float('nan'):.3f} ({np.mean(h < 0.05):.3f}) | {(P & Ex).sum() / max(Ex.sum(), 1):.3f} | {w(l0):+.5f} {w(lX):+.5f} {w(lE):+.5f}")
