"""wolf 정답 · 여러 판의 입자별 탄성 에너지 밀도 psi 를 같은 식 (exe/wolfen_psi.py 그대로: Hencky mu*sum(eps^2) + lam/2*(sum eps)^2, eps = log sigma_i,
E·nu = wolf.pt cfg) 으로 프레임마다 재고 (프레임 0 = 항등 -> psi 0), 고정 색 범위 [lo, hi] (log10(psi+1e-3), 모든 칸·프레임 공통, 정규화 없음) 로
lb_render --colors 용 col_<이름>.npy, psi_<이름>.npy, 프레임별 전체 에너지 E = sum(부피*psi) [J] (en.json) 를 만든다.
사용: python exe/cvben_psi.py <정답 디렉토리 sim_*.h5> <wolf.pt> <출력 디렉토리> <lo> <hi> 이름=pred.pt [이름=pred.pt ...]"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import h5py, numpy as np, torch
from tqdm import tqdm
gtd, wpt, out, lo, hi = sys.argv[1:6]; lo, hi = float(lo), float(hi)
runs = [s.split("=", 1) for s in sys.argv[6:]]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
from anchorflow import deform_model as DM, phys_resid as PR
W = torch.load(wpt, map_location="cpu", weights_only=False)
CFG = W["cfg"]; MU, LAM = PR.lame(CFG["E"], CFG["nu"])
VOL = (DM.traj_mass(W, "cpu").double() / float(CFG["density"])).numpy()
print(f"[재질] {CFG.get('material')} E {CFG['E']} nu {CFG['nu']} mu {MU:.4g} lam {LAM:.4g} 부피합 {VOL.sum():.4g} 입자 {len(VOL)}", flush=True)
EPS = 1e-3


def psi(F):
    sv = torch.linalg.svdvals(F.double().reshape(-1, 3, 3))
    ep = torch.log(sv.clamp_min(1e-14))
    return (MU * (ep ** 2).sum(-1) + 0.5 * LAM * ep.sum(-1) ** 2).numpy().astype(np.float32)


N = len(VOL); ALL = {}
for nm, pp in runs:
    D = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
    Fp = D["F"]; L = Fp.shape[0]; assert Fp.shape[1] == N, (Fp.shape, N)
    ALL[nm] = np.stack([np.zeros(N, np.float32)] + [psi(Fp[f].float()) for f in tqdm(range(L), desc=f"{nm} psi", ncols=70)])
    print(f"[{nm}] {pp} 프레임 {L} part {D.get('part', False)}", flush=True)
L = min(v.shape[0] for v in ALL.values()) - 1
fs = sorted(glob.glob(os.path.join(gtd, "sim_*.h5")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))[:L + 1]
assert len(fs) == L + 1, (len(fs), L)
G = []
for i, f in enumerate(tqdm(fs, desc="정답 psi", ncols=70)):
    with h5py.File(f, "r") as h:
        F = torch.from_numpy(np.array(h["F"]).reshape(-1, 3, 3).copy()).float()
    G.append(np.zeros(N, np.float32) if i == 0 else psi(F))
ALL = {"gt": np.stack(G), **{k: v[:L + 1] for k, v in ALL.items()}}
import matplotlib; matplotlib.use("Agg")
cm = matplotlib.colormaps["turbo"]
J = dict(lo=lo, hi=hi, eps=EPS, cmap="turbo", L=L, N=N, mu=MU, lam=LAM, vol=float(VOL.sum()), E={}, pmax={}, frac_out={})
for nm, P in ALL.items():
    np.save(f"{out}/psi_{nm}.npy", P)
    V = np.log10(P + EPS)
    C = np.empty((L + 1, N, 3), np.float16)
    for t in range(L + 1): C[t] = cm(np.clip((V[t] - lo) / (hi - lo), 0, 1))[:, :3].astype(np.float16)
    np.save(f"{out}/col_{nm}.npy", C)
    J["E"][nm] = (VOL[None] * P).sum(1).tolist(); J["pmax"][nm] = P.max(1).tolist()
    J["frac_out"][nm] = [float((V[1:] < lo).mean()), float((V[1:] > hi).mean())]
    print(f"[{nm}] E f1 {J['E'][nm][1]:.4g} f{L} {J['E'][nm][-1]:.4g} J, 범위 밖 (아래·위) {J['frac_out'][nm]}, log10 분위수 1·50·99.5 "
          f"{np.percentile(V[1:], [1, 50, 99.5]).round(3).tolist()}", flush=True)
json.dump(J, open(f"{out}/en.json", "w"))
print("CVBEN_PSI_DONE", flush=True)
