"""ficus 프레임별 운동·탄성·총 에너지 (중력 0) -- 정의는 /home/dkta/work/fen.py·fen_c1.py 그대로.

운동 = ½Σ m|v|² (m = traj_mass + PG 영역 밀도), 탄성 = Σ V_p ψ_p (V_p = dx³/PG 칸 입자 수, ψ = psi_color 가 남긴 psi_*.npy).
정답 운동은 정답 .pt 의 v[t], 실행은 덤프 v[t-1] (프레임 t 끝 상태).

사용: python exe/ficus_energy_cmp.py --gt_pt F_pg_v0.pt --gt_psi psi_gt.npy --run "라벨=실행디렉토리" ... --out_json e.json --fig e.png
      (실행디렉토리에 roll_ov.pt, psi/psi_ov.npy)
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np, torch
import anchorflow.deform_model as DM
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ap = argparse.ArgumentParser()
ap.add_argument("--gt_pt", required=True)
ap.add_argument("--gt_psi", required=True)
ap.add_argument("--gt_label", default="정답 (PG)")
ap.add_argument("--run", action="append", required=True, help="라벨=실행디렉토리")
ap.add_argument("--frames", type=int, default=125)
ap.add_argument("--out_json", required=True)
ap.add_argument("--fig", required=True)
ap.add_argument("--title", default="ficus")
a = ap.parse_args()
for f in fm.findSystemFonts():
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
DM.REGION_DENSITY[0] = True
d = torch.load(a.gt_pt, map_location="cpu", weights_only=False)
m = DM.traj_mass(d, "cpu").double(); c = d["cfg"]; ng = int(c.get("n_grid", 100)); dx = float(c.get("grid_lim", 2.0)) / ng
x0 = d["x"][0].float(); vi = (x0 / dx).long().clamp(0, ng - 1); fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cn = torch.zeros(ng ** 3).index_add_(0, fl, torch.ones(x0.shape[0])); V = (dx ** 3 / cn[fl]).double().numpy(); mn = m.numpy()
pg = np.load(a.gt_psi, mmap_mode="r"); vg = d["v"]; T = a.frames
print(f"[에너지] 입자 {x0.shape[0]}, 질량 합 {mn.sum():.4f}, 부피 합 {V.sum():.4e}", flush=True)
res = {"K0": 0.5 * float((mn * vg[0].double().pow(2).sum(-1).numpy()).sum())}
res[a.gt_label] = dict(K=[0.5 * float((mn * vg[t].double().pow(2).sum(-1).numpy()).sum()) for t in range(1, T + 1)],
                       E=[float((V * np.asarray(pg[t], dtype=np.float64)).sum()) for t in range(1, T + 1)])
for lab, R in (q.split("=", 1) for q in a.run):
    r = torch.load(R + "/roll_ov.pt", map_location="cpu", weights_only=False, mmap=True)
    po = np.load(R + "/psi/psi_ov.npy", mmap_mode="r")
    vo = r["v"]; n = min(T, vo.shape[0])
    res[lab] = dict(K=[0.5 * float((mn * vo[t - 1].double().pow(2).sum(-1).numpy()).sum()) for t in range(1, n + 1)],
                    E=[float((V * np.asarray(po[t], dtype=np.float64)).sum()) for t in range(1, n + 1)], path=R)
json.dump(res, open(a.out_json, "w"), ensure_ascii=False)
fd = float(c["frame_dt"]); cols = ["#222222", "#d9480f", "#1c7ed6", "#2b8a3e", "#ae3ec9"]
names = [n for n in res if n != "K0"]
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
for ax_, (k, lab) in zip(ax, [("K", "운동 에너지"), ("E", "탄성 에너지"), ("T", "총 에너지 (운동+탄성)")]):
    for i, name in enumerate(names):
        R_ = res[name]; y = np.array(R_["K"]) + np.array(R_["E"]) if k == "T" else np.array(R_[k])
        ax_.plot(np.arange(1, len(y) + 1) * fd, y, c=cols[i % len(cols)], lw=2.0 if i == 0 else 1.4, label=name)
    if k in ("K", "T"):
        ax_.axhline(res["K0"], c="gray", ls=":", label=f"초기 운동 에너지 {res['K0']:.3g} J")
    ax_.set_xlabel("시간 (s)"); ax_.set_ylabel("J"); ax_.set_title(f"{a.title} {lab}"); ax_.legend(fontsize=8); ax_.set_yscale("log")
fig.tight_layout(); fig.savefig(a.fig, dpi=100)
for name in names:
    R_ = res[name]
    print(name, " ".join(f"f{i + 1}: K {R_['K'][i]:.3e} E {R_['E'][i]:.3e} 합 {R_['K'][i] + R_['E'][i]:.3e}"
                         for i in (0, 9, 39, 79, min(124, len(R_['K']) - 1))))
print("FICUS_ENERGY_OK")
