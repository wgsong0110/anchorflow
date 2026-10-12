"""서브스텝 안정성 실험 공통 측정 (모든 방법이 같은 정의로).

프레임마다: 운동+위치 에너지(바닥 기준), NaN·상자 밖·det F ≤ 0 비율, 최대 속력, 고정 부분표본 위치.
탄성 에너지는 음이 아니므로 안정한(보존·소산) 계라면 운동+위치 에너지는 처음 값을 넘지 못한다.
"""
import numpy as np

NSUB = 32768


def subset(n, k=NSUB, seed=0):
    """모든 방법·모든 서브스텝 수에서 같은 번호 (같은 입력 입자 순서 기준)."""
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))


# 결과 npz 의 stats 열 순서 (모든 방법 공통). 뒤의 8 개는 2026-10-12 추가 -- 그 전 결과에는 없다(NaN).
KEYS = ("ke", "pe", "nan", "out", "vmax", "detneg", "detmin",
        "ee", "jdev", "jdev10", "penr", "pend", "px", "py", "pz")


def elastic_fcr(F, vol, E, nu):
    """고정 공회전(fixed corotated) 탄성 에너지 합 Σ V (μ‖F−R‖² + λ/2 (J−1)²). 방법마다 구성모델이 조금씩 달라도
    같은 자로 비교하려고 하나로 고정한다 (PG jelly 와 같은 모델)."""
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    U, S, Vt = np.linalg.svd(F)
    R = U @ Vt
    J = np.prod(S, axis=1)
    psi = mu * ((F - R) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2
    return float((vol * psi).sum())


def frame_stats(x, v, F, mass, g, floor_z, box=(0.0, 2.0), vol=None, E=None, nu=None):
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
        st["jdev"] = float(np.nanmean(np.abs(d - 1))); st["jdev10"] = float((np.abs(d - 1) > 0.1).mean())
        if vol is not None and E is not None and fin.any():
            st["ee"] = elastic_fcr(F[fin], np.asarray(vol, np.float64)[fin], float(E), float(nu))
    # 바닥 관통: 바닥 아래 입자 비율과 최대 깊이. 운동량 Σ m v (자유낙하 구간 보존 확인용)
    if len(xm):
        dep = float(floor_z) - xm[:, 2]
        st["penr"] = float((dep > 0).mean()); st["pend"] = float(max(dep.max(), 0.0))
        p = (mm[:, None] * vm).sum(0)
        st["px"], st["py"], st["pz"] = (float(q) for q in p)
    return st


def unstable(rows, e0, vcap, k_energy=1.5, x_det=0.01):
    """고정 문턱: NaN, 상자 밖 0.1%, 운동+위치 에너지 > k·처음, det F ≤ 0 > x, 최대 속력 > vcap. 처음 걸린 프레임 (없으면 -1)."""
    for t, s in enumerate(rows):
        if (s["nan"] > 0 or s["out"] > 1e-3 or s["ke"] + s["pe"] > k_energy * e0 or s.get("detneg", 0.0) > x_det
                or not np.isfinite(s["vmax"]) or s["vmax"] > vcap):
            return t
    return -1
