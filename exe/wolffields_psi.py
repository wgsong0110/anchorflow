"""wolf H9(=H8) 판 · 정답의 입자별 세 양 (탄성 에너지 밀도, 운동 에너지, 칸 세로 두께 비) 을 프레임마다 재고 lb_render --colors 용 .npy 와 프레임별 전체값 json 을 만든다.
 - 탄성 에너지 밀도 psi: exe/wolfen_psi.py 와 같은 식 (Hencky, 입자 F), 로그 눈금, 두 판·프레임 1~끝 공통 고정 범위
 - 운동 에너지: 입자별 1/2 m v^2, v = (x_f - x_{f-1}) / frame_dt (프레임 0 은 0), 로그 눈금, 두 판 공통 고정 범위
 - 칸 높이 비 r: exe/cellh_an.py 정의 그대로 (프레임 f = 걸음 f-1 -> f, 입자가 든 칸의 h끝/h시작), 노드 덤프는 cellh c9_1 (H9 wolf 같은 명령 재실행). H9 만.
사용: python exe/wolffields_psi.py <pred.pt> <정답 디렉토리 sim_*.h5 + config.json> <wolf.pt> <노드 덤프 디렉토리 (nodes/f*.npz)> <출력 디렉토리>"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import h5py, numpy as np, torch
from tqdm import tqdm
pp, gtd, wpt, nd, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
from anchorflow import deform_model as DM, phys_resid as PR
W = torch.load(wpt, map_location="cpu", weights_only=False)
CFG = W["cfg"]; MU, LAM = PR.lame(CFG["E"], CFG["nu"])
MASS = DM.traj_mass(W, "cpu").double().numpy(); VOL = MASS / float(CFG["density"])
GC = json.load(open(os.path.join(gtd, "config.json"))); DT = float(GC.get("frame_dt", 1 / 60))
print(f"[재질] E {CFG['E']} nu {CFG['nu']} mu {MU:.4g} lam {LAM:.4g} 질량합 {MASS.sum():.4g} 부피합 {VOL.sum():.4g} 입자 {len(VOL)} frame_dt {DT:.5g}", flush=True)
EPS, Q = 1e-3, (1.0, 99.5)


def psi(F):
    sv = torch.linalg.svdvals(F.double().reshape(-1, 3, 3))
    ep = torch.log(sv.clamp_min(1e-14))
    return (MU * (ep ** 2).sum(-1) + 0.5 * LAM * ep.sum(-1) ** 2).numpy().astype(np.float32)


def xs(a):
    a = np.asarray(a); return a.T if (a.shape[0] == 3 and a.shape[1] != 3) else a


D = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
Fp = D["F"]; L, N = Fp.shape[:2]; assert N == len(VOL), (N, len(VOL))
fs = sorted(glob.glob(os.path.join(gtd, "sim_*.h5")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))[:L + 1]
assert len(fs) == L + 1, (len(fs), L)
GX, G = [], []
for i, f in enumerate(tqdm(fs, desc="정답", ncols=70)):
    with h5py.File(f, "r") as h:
        GX.append(xs(h["x"])[:N].astype(np.float64))
        G.append(np.zeros(N, np.float32) if i == 0 else psi(torch.from_numpy(np.array(h["F"]).reshape(-1, 3, 3).copy()).float()))
GX, G = np.stack(GX), np.stack(G)
PX = np.concatenate([GX[:1], D["pred"].double().numpy()[:, :N]], 0)
P = np.stack([np.zeros(N, np.float32)] + [psi(Fp[f].float()) for f in tqdm(range(L), desc="H9 psi", ncols=70)])


def ke(X):
    v = np.zeros_like(X); v[1:] = (X[1:] - X[:-1]) / DT
    return (0.5 * MASS[None] * (v ** 2).sum(-1)).astype(np.float32)


KP, KG = ke(PX), ke(GX)
# ---- 칸 높이 비 (cellh_an.py 정의) ----
pl = [b for b in GC["boundary_conditions"] if b.get("type") == "surface_collider"][0]
NR = np.array(pl["normal"], dtype=np.float64); NR /= np.linalg.norm(NR)
PN = float(np.array([float(q) for q in pl["point"]]) @ NR)
OFF = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)
R = np.full((L + 1, N), np.nan, np.float32); R[0] = 1.0
for g in tqdm(sorted(int(f[1:4]) for f in os.listdir(os.path.join(nd, "nodes")) if f.endswith(".npz")), desc="칸 높이 비", ncols=70):
    z = np.load(os.path.join(nd, "nodes", f"f{g:03d}.npz"))
    ip = z["ip"]; bmin = ip[0:3].astype(np.int64); nb = ip[6:9].astype(np.int64)
    A = z["A"].astype(np.float64); act = z["act"].astype(np.int64); nbx = int(nb[0] * nb[1] * nb[2])
    i0 = act // int(nb[1] * nb[2]); j0 = (act // int(nb[2])) % int(nb[1]); k0 = act % int(nb[2])
    h0 = (np.stack([i0, j0, k0], 1).astype(np.float64) + bmin) @ A.T @ NR - PN
    hE = h0 + z["Ust"][-1].astype(np.float64) @ NR
    B0 = np.full(nbx, np.nan); BE = np.full(nbx, np.nan); B0[act] = h0; BE[act] = hE
    pb = z["pb"].astype(np.int64); fx = z["fx"].astype(np.float32); assert len(pb) == N, (len(pb), N)
    c = pb + (fx >= 1.0)
    nid = ((c[:, None, 0] + OFF[None, :, 0]) * int(nb[1]) + (c[:, None, 1] + OFF[None, :, 1])) * int(nb[2]) + (c[:, None, 2] + OFF[None, :, 2])
    ok = (nid >= 0).all(1) & (nid < nbx).all(1); nid = np.where(ok[:, None], nid, 0)
    a0 = B0[nid]; aE = BE[nid]; ok &= np.isfinite(a0).all(1) & np.isfinite(aE).all(1)
    hs = a0.max(1) - a0.min(1); he = aE.max(1) - aE.min(1)
    R[g] = np.where(ok & (hs > 0), he / np.where(hs > 0, hs, 1), np.nan)
# ---- 색 범위 ----
import matplotlib; matplotlib.use("Agg")
cm = matplotlib.colormaps["turbo"]


def rng(A, B, eps, span=None):
    la, lb = np.log10(A[1:] + eps), np.log10(B[1:] + eps)
    lo, hi = (float(v) for v in np.percentile(np.concatenate([la.ravel(), lb.ravel()]), Q))
    if span: lo = max(lo, hi - span)
    return lo, hi


def paint(V, lo, hi):
    C = np.empty(V.shape + (3,), np.float16)
    for t in range(V.shape[0]): C[t] = cm(np.clip((V[t] - lo) / (hi - lo), 0, 1))[:, :3].astype(np.float16)
    return C


EK = 1e-30
elo, ehi = rng(P, G, EPS)
klo, khi = rng(KP, KG, EK, span=8)
print(f"[색 범위] 탄성 log10(psi+{EPS}) [{elo:.3f}, {ehi:.3f}] J/m^3 · 운동 log10(KE) [{klo:.3f}, {khi:.3f}] J (분위수 {Q}, 두 판 프레임 1~{L})", flush=True)
np.save(f"{out}/col_pred_en.npy", paint(np.log10(P + EPS), elo, ehi)); np.save(f"{out}/col_gt_en.npy", paint(np.log10(G + EPS), elo, ehi))
np.save(f"{out}/col_pred_ke.npy", paint(np.log10(KP + EK), klo, khi)); np.save(f"{out}/col_gt_ke.npy", paint(np.log10(KG + EK), klo, khi))
RLO, RHI = 0.3, 1.1
t = np.where(R < 1, 0.5 * (R - RLO) / (1 - RLO), 0.5 + 0.5 * (R - 1) / (RHI - 1)); t = np.clip(t, 0, 1)
CR = matplotlib.colormaps["RdBu"](t)[..., :3].astype(np.float16); CR[~np.isfinite(R)] = 0.75
np.save(f"{out}/col_pred_r.npy", CR)
rmed = [float(np.nanmedian(R[t])) for t in range(L + 1)]
IDX = np.sort(np.random.default_rng(0).choice(N, min(20000, N), replace=False))   # 그리기용 표본 (두 판 같은 번호, 계산·지표에는 안 씀)
np.savez(f"{out}/sample.npz", idx=IDX, P=P[:, IDX], G=G[:, IDX], KP=KP[:, IDX], KG=KG[:, IDX], R=R[:, IDX])
json.dump(dict(L=L, N=N, dt=DT, e_lo=elo, e_hi=ehi, k_lo=klo, k_hi=khi, r_lo=RLO, r_hi=RHI, eps=EPS, ek=EK,
               E_el_pred=(VOL[None] * P).sum(1).tolist(), E_el_gt=(VOL[None] * G).sum(1).tolist(),
               E_ke_pred=KP.sum(1).astype(float).tolist(), E_ke_gt=KG.sum(1).astype(float).tolist(), r_med=rmed,
               r_bad=[int(np.isnan(R[t]).sum()) for t in range(L + 1)]), open(f"{out}/fields.json", "w"))
print(f"[전체] 탄성 H9 f{L} {(VOL*P[-1]).sum():.4g} 정답 {(VOL*G[-1]).sum():.4g} J · 운동 H9 {KP[-1].sum():.4g} 정답 {KG[-1].sum():.4g} J · r 중앙 f{L} {rmed[-1]:.3f}", flush=True)
print("WOLFFIELDS_DONE", flush=True)
