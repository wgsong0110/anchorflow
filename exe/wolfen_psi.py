"""wolf H9(=H8) 판 · 정답의 입자별 탄성 에너지 밀도 psi (Hencky: mu*sum(eps^2) + lam/2*(sum eps)^2, eps = log sigma_i, E 5e7 nu 0.3; 절 CM110, exe/smin_an.py 와 같은 식) 를
프레임마다 재고 (프레임 0 = 항등 -> psi 0), 두 판 공통 고정 색 범위 (log10(psi+eps), 프레임 1~끝 · 두 판 · 전 입자 합친 분위수) 로 lb_render --colors 용 .npy 와 프레임별 전체 에너지 E = sum(부피*psi) [J] 를 만든다.
사용: python exe/wolfen_psi.py <pred.pt> <정답 디렉토리 sim_*.h5> <wolf.pt> <출력 디렉토리>"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import h5py, numpy as np, torch
from tqdm import tqdm
pp, gtd, wpt, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
from anchorflow import deform_model as DM, phys_resid as PR
W = torch.load(wpt, map_location="cpu", weights_only=False)
CFG = W["cfg"]; MU, LAM = PR.lame(CFG["E"], CFG["nu"])
VOL = (DM.traj_mass(W, "cpu").double() / float(CFG["density"])).numpy()
print(f"[재질] {CFG.get('material')} E {CFG['E']} nu {CFG['nu']} mu {MU:.4g} lam {LAM:.4g} 부피합 {VOL.sum():.4g} 입자 {len(VOL)}", flush=True)
EPS, Q = 1e-3, (1.0, 99.5)


def psi(F):
    sv = torch.linalg.svdvals(F.double().reshape(-1, 3, 3))
    ep = torch.log(sv.clamp_min(1e-14))
    return (MU * (ep ** 2).sum(-1) + 0.5 * LAM * ep.sum(-1) ** 2).numpy().astype(np.float32)


D = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
Fp = D["F"]; L, N = Fp.shape[:2]; assert N == len(VOL), (N, len(VOL))
P = [np.zeros(N, np.float32)] + [psi(Fp[f].float()) for f in tqdm(range(L), desc="H9 psi", ncols=70)]
fs = sorted(glob.glob(os.path.join(gtd, "sim_*.h5")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))[:L + 1]
assert len(fs) == L + 1, (len(fs), L)
G = []
for i, f in enumerate(tqdm(fs, desc="정답 psi", ncols=70)):
    with h5py.File(f, "r") as h:
        F = torch.from_numpy(np.array(h["F"]).reshape(-1, 3, 3).copy()).float()
    G.append(np.zeros(N, np.float32) if i == 0 else psi(F))
P, G = np.stack(P), np.stack(G)
np.save(f"{out}/psi_pred.npy", P); np.save(f"{out}/psi_gt.npy", G)
LP, LG = np.log10(P + EPS), np.log10(G + EPS)
lo, hi = (float(v) for v in np.percentile(np.concatenate([LP[1:].ravel(), LG[1:].ravel()]), Q))
print(f"[색 범위] log10(psi+{EPS}) [{lo:.4f}, {hi:.4f}] = psi {10**lo:.3g} ~ {10**hi:.3g} J/m^3 (분위수 {Q}, 두 판 프레임 1~{L})", flush=True)
import matplotlib; matplotlib.use("Agg")
cm = matplotlib.colormaps["turbo"]
for nm, V in (("pred", LP), ("gt", LG)):
    C = np.empty((L + 1, N, 3), np.float16)
    for t in range(L + 1): C[t] = cm(np.clip((V[t] - lo) / (hi - lo), 0, 1))[:, :3].astype(np.float16)
    np.save(f"{out}/col_{nm}.npy", C)
Ep, Eg = (VOL[None] * P).sum(1), (VOL[None] * G).sum(1)
json.dump(dict(lo=lo, hi=hi, eps=EPS, q=list(Q), cmap="turbo", E_pred=Ep.tolist(), E_gt=Eg.tolist(), L=L, N=N, mu=MU, lam=LAM, vol=float(VOL.sum()),
               pmax_pred=P.max(1).tolist(), pmax_gt=G.max(1).tolist()), open(f"{out}/en.json", "w"))
print(f"[E] H9 f1 {Ep[1]:.4g} f{L} {Ep[-1]:.4g} J, 정답 f1 {Eg[1]:.4g} f{L} {Eg[-1]:.4g} J", flush=True)
print("WOLFEN_PSI_DONE", flush=True)
