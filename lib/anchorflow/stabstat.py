"""서브스텝 안정성 실험 공통 측정 (모든 방법이 같은 정의로).

프레임마다: 운동+위치 에너지(바닥 기준), NaN·상자 밖·det F ≤ 0 비율, 최대 속력, 고정 부분표본 위치.
탄성 에너지는 음이 아니므로 안정한(보존·소산) 계라면 운동+위치 에너지는 처음 값을 넘지 못한다.
"""
import numpy as np

NSUB = 32768


def subset(n, k=NSUB, seed=0):
    """모든 방법·모든 서브스텝 수에서 같은 번호 (같은 입력 입자 순서 기준)."""
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))


def frame_stats(x, v, F, mass, g, floor_z, box=(0.0, 2.0)):
    x = np.asarray(x, np.float64); v = np.asarray(v, np.float64)
    fin = np.isfinite(x).all(1) & np.isfinite(v).all(1)
    out = fin & ((x < box[0]) | (x > box[1])).any(1)
    xm, vm, mm = x[fin], v[fin], mass[fin]
    gz = -float(g[2])
    ke = 0.5 * float((mm * (vm * vm).sum(1)).sum())
    pe = gz * float((mm * (xm[:, 2] - floor_z)).sum())
    st = dict(ke=ke, pe=pe, nan=float(1 - fin.mean()), out=float(out.mean()),
              vmax=float(np.sqrt((vm * vm).sum(1)).max()) if len(vm) else float("nan"))
    if F is not None:
        F = np.asarray(F, np.float64).reshape(-1, 3, 3)
        d = np.linalg.det(F[fin]) if fin.any() else np.array([np.nan])
        st["detneg"] = float((d <= 0).mean()); st["detmin"] = float(np.nanmin(d))
    return st


def unstable(rows, e0, vcap, k_energy=1.5, x_det=0.01):
    """고정 문턱: NaN, 상자 밖 0.1%, 운동+위치 에너지 > k·처음, det F ≤ 0 > x, 최대 속력 > vcap. 처음 걸린 프레임 (없으면 -1)."""
    for t, s in enumerate(rows):
        if (s["nan"] > 0 or s["out"] > 1e-3 or s["ke"] + s["pe"] > k_energy * e0 or s.get("detneg", 0.0) > x_det
                or not np.isfinite(s["vmax"]) or s["vmax"] > vcap):
            return t
    return -1
