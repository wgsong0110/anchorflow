"""wolf H9 증분 포텐셜 항별 기록 (lib/anchorflow/fem_wolfip.py 덤프 ip/f*.npz, wolfip.jsonl) -> 렌더 색 .npy · 프레임별 합 json · 산점도 표본.
색 (lb_render --colors, [51, N, 3], 프레임 0 = 걸음 전 회색):
  합 = 관성 + 탄성 + 중력 + 장벽 + 정규화 (입자 몫, J): 발산형 RdBu_r (0 흰색), 대칭 로그 고정 범위 ±lim (lim = |합| 99.5 분위수, 선형 구간 lim·1e-4)
  관성 · 탄성 · 장벽+정규화 (입자 몫, J): YlOrRd, log10 고정 범위 (양수 값 1~99.5 분위수), 0 이하는 가장 낮은 색
  범위는 프레임 1~50 전체에서 한 번 정함 (프레임마다 정규화 안 함).
산점도 표본 (그리기용만): 입자 2만 개 (모든 프레임 같은 번호), 칸은 프레임마다 격자가 달라 프레임마다 무작위 2만 칸.
사용: python exe/wolfip_an.py <판 출력 디렉토리 (ip/ wolfip.jsonl)> <출력 디렉토리>"""
import glob, json, os, sys
import numpy as np
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
rd, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
fs = sorted(glob.glob(f"{rd}/ip/f*.npz"))
L = len(fs)
assert L >= 1 and fs[-1].endswith(f"f{L:03d}.npz"), (L, fs[-1:])
REC = {}
for ln in open(f"{rd}/wolfip.jsonl"):
    r = json.loads(ln); REC[r["f"]] = r                                     # 쪼갠 프레임은 마지막 시도 (받아들인 것) 가 덮어씀
Z0 = np.load(fs[0]); N = len(Z0["e_in"])
KEYS = ("e_in", "e_el", "e_gr", "e_bar", "e_reg")
A = {k: np.zeros((L + 1, N), np.float32) for k in KEYS + ("psi",)}
rng = np.random.default_rng(0)
IDX = np.sort(rng.choice(N, min(20000, N), replace=False))
CEL, CREG, CN = [np.zeros(0, np.float32)], [np.zeros(0, np.float32)], [0]
for i, f in enumerate(tqdm(fs, desc="프레임", ncols=70)):
    z = np.load(f)
    for k in KEYS + ("psi",):
        A[k][i + 1] = z[k]
    nc = len(z["el_c"]); sel = np.sort(rng.choice(nc, min(20000, nc), replace=False))
    CEL.append(z["el_c"][sel]); CREG.append(z["reg_c"][sel]); CN.append(nc)
S = sum(A[k] for k in KEYS)
RB = A["e_bar"] + A["e_reg"]
tot = lambda X: [0.0] + [float(X[t].astype(np.float64).sum()) for t in range(1, L + 1)]
J = dict(L=L, N=N, T=L + 1, n_sample=int(len(IDX)),
         E_sum=tot(S), E_in=tot(A["e_in"]), E_el=tot(A["e_el"]), E_gr=tot(A["e_gr"]), E_bar=tot(A["e_bar"]), E_reg=tot(A["e_reg"]), E_rb=tot(RB),
         cell_n=CN, rec=[REC.get(t) for t in range(L + 1)])
# ---- 색 범위 (프레임 1~L 전체) ----
def logrange(X):
    v = X[1:].ravel(); v = v[v > 0]
    if v.size == 0:
        return -12.0, 0.0
    lo, hi = (float(q) for q in np.percentile(np.log10(v), (1.0, 99.5)))
    return lo, hi
lim = float(np.percentile(np.abs(S[1:]).ravel(), 99.5)) or 1.0; lt = lim * 1e-4
R = dict(sum_lim=lim, sum_lt=lt)
for nm, X in (("in", A["e_in"]), ("el", A["e_el"]), ("rb", RB)):
    R[nm] = logrange(X)
J["range"] = R
print(f"[범위] 합 ±{lim:.3g} J (선형 {lt:.3g}) · 관성 {R['in']} · 탄성 {R['el']} · 장벽+정규화 {R['rb']} (log10 J)", flush=True)
GRAY = np.array([0.75, 0.75, 0.75], np.float16)
def paint_log(X, lo, hi, cm):
    C = np.empty((L + 1, N, 3), np.float16); C[0] = GRAY
    for t in range(1, L + 1):
        v = np.log10(np.maximum(X[t].astype(np.float64), 1e-300))
        C[t] = cm(np.clip((v - lo) / (hi - lo), 0, 1))[:, :3].astype(np.float16)
    return C
def symlog(v):
    return np.sign(v) * np.log10(1 + np.abs(v) / lt) / np.log10(1 + lim / lt)
cmS = matplotlib.colormaps["RdBu_r"]; cmL = matplotlib.colormaps["YlOrRd"]
C = np.empty((L + 1, N, 3), np.float16); C[0] = GRAY
for t in range(1, L + 1):
    C[t] = cmS(np.clip(0.5 + 0.5 * symlog(S[t].astype(np.float64)), 0, 1))[:, :3].astype(np.float16)
np.save(f"{out}/col_sum.npy", C)
np.save(f"{out}/col_in.npy", paint_log(A["e_in"], *R["in"], cmL))
np.save(f"{out}/col_el.npy", paint_log(A["e_el"], *R["el"], cmL))
np.save(f"{out}/col_rb.npy", paint_log(RB, *R["rb"], cmL))
np.savez(f"{out}/sample.npz", idx=IDX, psi=A["psi"][:, IDX], reg=A["e_reg"][:, IDX],
         cel=np.concatenate(CEL), creg=np.concatenate(CREG), coff=np.cumsum([0] + [len(c) for c in CEL]))
json.dump(J, open(f"{out}/ip.json", "w"), ensure_ascii=False)
r = REC.get(L, {})
print(f"[f{L}] 입자 합: 관성 {J['E_in'][-1]:.4g} 탄성 {J['E_el'][-1]:.4g} 중력 {J['E_gr'][-1]:.4g} 장벽 {J['E_bar'][-1]:.4g} 정규화 {J['E_reg'][-1]:.4g} 합 {J['E_sum'][-1]:.4g} J; "
      f"기록: 노드 {r.get('node')} 가우스 {r.get('gauss')} 면 {r.get('face')}", flush=True)
print("WOLFIP_AN_DONE", flush=True)
