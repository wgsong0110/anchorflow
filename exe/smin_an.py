"""wolf H8(=H9) 판 입자 F 의 σ_min · det F 프레임별 히스토그램 (CPU torch). 사용: python exe/smin_an.py <pred.pt> <gt 디렉토리 (sim_*.h5)> <out.json>
σ_min = F 의 최소 특이값, det F < 0 이면 -σ_min (부호 붙임). 히스토그램 범위·bin 전 프레임 고정. 프레임 f = 덤프 F[f-1] (프레임 0 = 항등)."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, torch
pp, gtd, outp = sys.argv[1:4]
torch.set_num_threads(8)
from anchorflow import deform_model as DM, phys_resid as PR
W = torch.load(os.path.join(os.path.dirname(os.path.abspath(gtd)), "..", "wolf.pt") if False else "/dev/shm/dkta/ag_data/wolf.pt", map_location="cpu", weights_only=False)
CFG = W["cfg"]; MU, LAM = PR.lame(CFG["E"], CFG["nu"])
VOL = (DM.traj_mass(W, "cpu").double() / float(CFG["density"])).numpy()
print(f"[재질] material {CFG.get('material')} E {CFG['E']} nu {CFG['nu']} density {CFG['density']} mu {MU:.4g} lam {LAM:.4g} 부피합 {VOL.sum():.4g} 입자 {len(VOL)}", flush=True)
SL, SH, SN = -1.0, 2.0, 150
DL, DH, DN = -1.0, 3.0, 200
ST = {}
def stats(F):
    F = F.double()
    s = torch.linalg.svdvals(F)[:, -1]
    d = torch.linalg.det(F)
    sg = torch.where(d < 0, -s, s)
    ep = torch.log(s.clamp_min(1e-14))                       # Hencky (모래 탄성, exe/fem_an_rt_sim.py 의 PR.lame + 절 AO 응력식): ψ = μ Σε² + λ/2 (Σε)², ε = log σ_i
    ST["psi"] = (MU * (ep ** 2).sum(-1) + 0.5 * LAM * ep.sum(-1) ** 2).numpy()
    return sg.numpy(), d.numpy()
def hist(v, lo, hi, n):
    h, _ = np.histogram(np.clip(v, lo, hi - 1e-9), bins=n, range=(lo, hi))
    return h.tolist()
def rec(sg, d):
    return dict(E=float((VOL * ST["psi"]).sum()), n=int(len(sg)), sh=hist(sg, SL, SH, SN), dh=hist(d, DL, DH, DN), med=float(np.median(sg)), p10=float(np.percentile(sg, 10)),
                lt05=float((sg < 0.5).mean()), neg=float((d < 0).mean()), dmed=float(np.median(d)), smin=float(sg.min()), smax=float(sg.max()),
                dmin=float(d.min()), dmax=float(d.max()), s_under=int((sg < SL).sum()), s_over=int((sg >= SH).sum()), d_under=int((d < DL).sum()), d_over=int((d >= DH).sum()))
D = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
Fp = D["F"]; L, Np = Fp.shape[:2]
print(f"[덤프] F {tuple(Fp.shape)} {Fp.dtype} part {D.get('part')} diverged {D.get('diverged')} keys {list(D.keys())}", flush=True)
assert len(VOL) == Np, (len(VOL), Np)
I = torch.eye(3).expand(Np, 3, 3)
out = dict(mu=MU, lam=LAM, E=CFG["E"], nu=CFG["nu"], density=float(CFG["density"]), L=L, Np=Np, srange=[SL, SH, SN], drange=[DL, DH, DN], part=bool(D.get("part", False)), diverged=D.get("diverged"), pred=[], gt=None, gt_note="")
for f in range(L + 1):
    out["pred"].append(rec(*stats(I if f == 0 else Fp[f - 1].float())))
    if f % 10 == 0: print(f"[pred] 프레임 {f}/{L} 중앙 {out['pred'][-1]['med']:.3f} <0.5 {out['pred'][-1]['lt05']:.3f}", flush=True)
try:
    import h5py
    fs = sorted(glob.glob(os.path.join(gtd, "**", "sim_*.h5"), recursive=True), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
    print(f"[gt] {gtd} h5 {len(fs)} 개; 목록 {os.listdir(gtd)[:20]}", flush=True)
    assert fs, "sim_*.h5 없음"
    gt = []
    for i, f in enumerate(fs[:L + 1]):
        with h5py.File(f, "r") as h:
            if i == 0: print("[gt] h5 키", list(h.keys()), flush=True)
            F = torch.from_numpy(np.array(h["F" if "F" in h else "f_tensor"]).reshape(-1, 3, 3).copy()).float()
        if i == 0 and float(F.abs().max()) == 0.0: F = torch.eye(3).expand_as(F).clone(); print("[gt] 프레임 0 F 가 0 -> 항등", flush=True)
        assert F.shape[0] == Np, (F.shape, Np)
        gt.append(rec(*stats(F)))
    out["gt"] = gt; out["gt_note"] = f"정답 h5 {len(fs)} 프레임의 F"
except Exception as e:  # noqa: BLE001
    out["gt_note"] = f"정답 F 없음: {e!r}"; print("[gt 실패]", repr(e), flush=True)
json.dump(out, open(outp, "w"))
print("[끝]", out["gt_note"], flush=True)
