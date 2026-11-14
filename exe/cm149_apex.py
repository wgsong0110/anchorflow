"""절 CM149 (인스턴스, numpy+torch): 정답 h5 (f_tensor 원본 f32) 와 우리 판 pred.pt 의 입자 F 에서 프레임별 '꼭짓점 (응력 0) 몫'.
정의: PhysGaussian 모래 반환 사상의 꼭짓점 갈래 (tr ε > 0 인 팽창 -> F_e = U Vᵀ, 특이값 전부 1) 와 같은 상태 = 특이값 전부 1 (max |σ_i - 1| < tol, tol 1e-5; f32 반올림 1e-7 보다 크고 정지 압축 변형 ~4e-4 보다 작음).
층 = 시작 높이 z0 구간. 같이: 평균 J = det F, J > 1 몫, 꼭짓점 아닌 입자의 평균 ln J.  사용: python exe/cm149_apex.py <h5 디렉토리> <표 이름=pred.pt 경로> ... """
import sys, glob, os
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch
except ImportError:
    torch = None
    import cm130_ptload as L                                   # 로컬 arm64 (torch 없음): numpy 로 pred.pt 읽기
h5d = sys.argv[1]; runs = [a.split("=", 1) for a in sys.argv[2:]]
FR = (10, 15, 20, 25, 30, 35, 40); TOL = 1e-5
def load_h5(f):
    with h5py.File(os.path.join(h5d, f"sim_{f:010d}.h5"), "r") as h: return np.array(h["x"]).T, np.array(h["f_tensor"]).T.reshape(-1, 3, 3)
x0, _ = load_h5(0); z0 = x0[:, 2]
layers = (("z0<0.7", z0 < 0.7), ("0.7-1.0", (z0 >= 0.7) & (z0 < 1.0)), ("z0>=1.0", z0 >= 1.0), ("전체", np.ones_like(z0, bool)))
def stats(F):
    F = F.astype(np.float64); s = np.linalg.svd(F, compute_uv=False); J = np.linalg.det(F)
    apex = np.abs(s - 1).max(1) < TOL
    return apex, J
def row(name, F):
    apex, J = stats(F); out = []
    for ln, m in layers:
        a = apex[m].mean(); jm = np.log(np.abs(J[m & ~apex]) + 1e-12).mean() if (m & ~apex).any() else float("nan")
        out.append(f"{ln} 꼭짓점 {a:.3f} J>1 {(J[m] > 1).mean():.3f} lnJ(비꼭짓점) {jm:+.4f}")
    print(f"{name:14s} " + " | ".join(out), flush=True)
R = {n: (torch.load(p, map_location="cpu", weights_only=False)["F"] if torch else L.load(p)["F"]) for n, p in runs}
for f in FR:
    print(f"== f{f}")
    _, Fg = load_h5(f); row("정답", Fg)
    for n, Fa in R.items(): row(n, np.asarray(Fa[f - 1]).reshape(-1, 3, 3))
