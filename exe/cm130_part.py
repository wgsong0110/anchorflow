"""절 CM130 (로컬 numpy): wolf 판 입자 덤프 (pred.pt F·x) 에서 프레임별 큰 탄성 에너지·뒤집힌 입자의 자리.
psi = Hencky mu*sum(eps^2)+lam/2*(sum eps)^2, eps = log|sigma| (exe/cvben_psi.py 와 같은 식), 입자 부피 = 균일 (총 부피/N 근사 -- 비교용 몫만 씀).
분류: 이탈 = 그 프레임 몸 상자 (0.1·99.9 % 분위 ± 0.08 m, sinv_esc 와 같은 정의) 밖, 바닥 근처 = z − z_floor < 0.03 m (z_floor = 프레임 0 최소 z).
사용: python exe/cm130_part.py <pred.pt> <이름>"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cm130_ptload as L
MU, LAM = 1.9230769e7, 2.8846154e7
pp, nm = sys.argv[1:3]
D = L.load(pp); X = D["pred"]; F = D["F"]
z0 = float(np.asarray(X[0])[:, 2].min())
print(f"[{nm}] 프레임 {X.shape[0]} 입자 {X.shape[1]} z_floor {z0:.4f}")
print(" f | det<=0 수 (바닥3cm·이탈) det최소 | psi>1e6 수 (바닥·이탈·det<=0) 총psi몫 (이탈·바닥·det<=0·top10) | 큰psi 입자 sigma 중앙 (max,min) det 중앙")
seen = np.zeros(X.shape[1], bool)
for f in range(X.shape[0]):
    x = np.asarray(X[f], np.float64); A = np.asarray(F[f], np.float64)
    s = np.linalg.svd(A, compute_uv=False); d = np.linalg.det(A)
    e = np.log(np.maximum(np.abs(s), 1e-14)); psi = MU * (e ** 2).sum(1) + 0.5 * LAM * e.sum(1) ** 2
    lo = np.percentile(x, 0.1, 0) - 0.08; hi = np.percentile(x, 99.9, 0) + 0.08
    esc = ((x < lo) | (x > hi)).any(1); fl = (x[:, 2] - z0) < 0.03; ng = d <= 0
    big = psi > 1e6; tot = psi.sum() + 1e-30; top = np.sort(psi)[::-1][:10].sum() / tot
    new = ng & ~seen; seen |= ng
    if f + 1 in (5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50) or (new.sum() and new.sum() == ng.sum()):
        sb = f"{np.median(s[big,0]):.3g} {np.median(s[big,2]):.3g} {np.median(d[big]):.3g}" if big.any() else "-"
        print(f"{f+1:2d} | {ng.sum():5d} ({(ng&fl).sum()}·{(ng&esc).sum()}) {d.min():.3f} | {big.sum():6d} ({(big&fl).sum()}·{(big&esc).sum()}·{(big&ng).sum()}) "
              f"{psi[esc].sum()/tot:.3f} {psi[fl].sum()/tot:.3f} {psi[ng].sum()/tot:.3f} {top:.3f} | {sb}")
