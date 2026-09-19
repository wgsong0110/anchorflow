"""성긴 격자가 빽빽한 격자와 **같은 답**을 내는지 확인한다.

빈 칸을 들고 다니지 않는 것은 계산을 줄이는 일이지 물리를 바꾸는 일이 아니다.
다만 "그럴 것이다" 로 두면 안 된다 -- 나중에 GF 와 갈렸을 때 원인이 이식인지
성김인지 못 가른다. 그래서 켜기 전에 한 번 못을 박는다.

**완전히 같기를 기대하면 안 된다.** p2g 는 원자적 덧셈이라 더하는 순서가 실행마다
다르고, 부동소수는 순서를 탄다. 그래서 먼저 **빽빽 격자를 두 번 돌려 재현 바닥**을
재고, 성김이 그 바닥 안에 드는지로 판정한다.
"""
import argparse, glob, os, shutil, subprocess, sys, time

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--frames", type=int, default=6)
ap.add_argument("--block", type=int, default=8)
a = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
secs = {}
for mode in ("dense", "dense2", "sparse"):
    out = os.path.join(a.work, f"eq_{mode}")
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "exe", "gf_mpm.py"),
                        "--config", a.config, "--h5", a.h5, "--out", out,
                        "--frames", str(a.frames),
                        "--grid", "dense" if mode.startswith("dense") else mode,
                        "--block", str(a.block), "--ckpt_every", "0"],
                       capture_output=True, text=True)
    secs[mode] = time.time() - t0
    if r.returncode != 0:
        print(f"[{mode}] 실패\n{(r.stdout + r.stderr)[-2000:]}")
        raise SystemExit(1)

def gap(m1, m2):
    fa = sorted(glob.glob(os.path.join(a.work, f"eq_{m1}", "sim_*.h5")))
    fb = sorted(glob.glob(os.path.join(a.work, f"eq_{m2}", "sim_*.h5")))
    w = 0.0
    for p, q in zip(fa, fb):
        with h5py.File(p, "r") as h:
            xa = np.array(h["x"])
        with h5py.File(q, "r") as h:
            xb = np.array(h["x"])
        w = max(w, float(np.abs(xa - xb).max()))
    return w, len(fa)


floor, nf = gap("dense", "dense2")       # 같은 코드 두 번 -- 재현 바닥
got, _ = gap("dense", "sparse")
sp = secs["dense"] / max(secs["sparse"], 1e-9)
print(f"[성김 검사] {nf} 프레임 | 재현 바닥(빽빽 x2) {floor:.3e} | "
      f"빽빽 대 성김 {got:.3e} | 빽빽 {secs['dense']:.1f}s, "
      f"성김 {secs['sparse']:.1f}s ({sp:.2f}배)")
if got <= max(floor, 1e-12) * 1.5:
    print("SPARSE_EQUAL")
else:
    print(f"SPARSE_DIFFERS 바닥 {floor:.3e} 인데 {got:.3e} -- 성긴 격자를 쓰지 말 것")
