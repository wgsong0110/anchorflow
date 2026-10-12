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
# 보정판 누적 개입 횟수 (클램프·되돌림 입자 수, 반으로 나눠 다시 푼 횟수, 관성 대체 횟수). 보정판이 아니면 0
SAFE_KEYS = ("safe_n", "safe_split", "safe_inert")


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


# ---- 발산 방지 보정판 (--safe, 2026-10-12 사용자 지시: 네 방법에 같은 처리) ----------------------------------------
# 발산해 죽는 작은 N 도 끝까지 돌려 지표를 재기 위한 **공통** 안전장치. 서브스텝마다:
#   ① 위치·속도·F 중 비유한이 생긴 입자는 직전 위치로 되돌리고 속도 0, F = I
#   ② 속력을 안정성 판정 상한 VCAP = 10·sqrt(2 g H) (자유낙하 최고 속력의 10 배) 로 자른다
#   ③ 위치를 영역 상자 [3dx, GL−3dx] 안으로 자른다
# 정상 실행에서는 한 번도 걸리지 않으므로(속력·위치가 그 범위 안) 결과가 그대로다. 걸린 입자 수를 센다.
def safe_params(x0, floor_z, g, grid_lim, n_grid):
    gz = abs(float(g[2])) if hasattr(g, "__len__") else abs(float(g))
    H = float(np.asarray(x0)[:, 2].max() - floor_z)
    dx = float(grid_lim) / int(n_grid)
    return dict(lo=3 * dx, hi=float(grid_lim) - 3 * dx, vcap=10.0 * np.sqrt(2 * gz * max(H, 1e-6)))


def safe_fix(x, v, F, x_prev, sp):
    """torch 텐서 제자리 수정. x,v [N,3], F [N,3,3] 또는 None, x_prev [N,3]. -> 손댄 입자 수 (int)"""
    import torch
    bad = ~(torch.isfinite(x).all(1) & torch.isfinite(v).all(1))
    if F is not None:
        bad = bad | ~torch.isfinite(F.reshape(F.shape[0], -1)).all(1)
    n = int(bad.sum())
    if n:
        x[bad] = x_prev[bad]; v[bad] = 0.0
        if F is not None:
            F[bad] = torch.eye(3, device=F.device, dtype=F.dtype)
    sp_ = v.norm(dim=1)
    over = sp_ > sp["vcap"]
    m = int(over.sum())
    if m:
        v[over] *= (sp["vcap"] / sp_[over]).unsqueeze(1)
    out = ((x < sp["lo"]) | (x > sp["hi"])).any(1)
    k = int(out.sum())
    if k:
        x.clamp_(sp["lo"], sp["hi"])
    return n + m + k
