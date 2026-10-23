"""riem 감쇠 ε 판 비교 그림: 원래(ε = 0.01·λmax) / (a) ε 고정 / (b) 적응 LM.

프레임별 RMSE · 입자 이동량(직전 프레임 대비, 정답과 함께) · 장벽 셀 det 최소 · J<0.5 셀 수 · ε · 걸음 크기 |sΔ|.
원래 실행(c3)은 RMSE·이동량·ε 가 저장돼 있고, 셀 det·걸음은 같은 설정 재실행(--riem_damp rel, 로그 추가판)에서 읽는다.

  python exe/topo_eps_cmp.py --flow flow_wolf.npz --orig c3/wolf_tetmesh.npz --rel eps/wolf_rel.npz \
      --fix eps/wolf_fix.npz --lm eps/wolf_lm.npz --out wolf_tetmesh_eps_cmp.png --title wolf
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--orig", required=True)
ap.add_argument("--rel", default="")
ap.add_argument("--fix", required=True)
ap.add_argument("--lm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--font", default="/home/dkta/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib import font_manager                                  # noqa: E402
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False

F = np.load(a.flow)
GT = F["traj"].astype(np.float64)                                    # [T+1,N,3]


def stats(path):
    d = np.load(path, allow_pickle=True)
    L = float(d["L"]); m = d["metrics"]; T = m.shape[0]
    tr = np.concatenate([GT[:1], d["traj"].astype(np.float64)], 0)
    mv = np.linalg.norm(tr[1:] - tr[:-1], axis=-1).mean(1) / L * 100
    gmv = np.linalg.norm(GT[1:T + 1] - GT[:T], axis=-1).mean(1) / L * 100
    out = dict(t=m[:, 0], rmse=100 * m[:, 1], mv=mv, gmv=gmv, eps=d["stop"][:, 4], L=L)
    if "tet_log" in d.files and d["tet_log"].size:
        tl = d["tet_log"]
        out.update(jmin=tl[:, 1], n05=tl[:, 2], nneg=tl[:, 3], ncell=tl[0, 4])
    if "riem_log" in d.files and d["riem_log"].size:
        r = d["riem_log"]
        ts = np.unique(r[:, 0])
        out["step"] = np.array([np.median(r[r[:, 0] == t, 3]) for t in ts])
        out["eps_med"] = np.array([np.median(r[r[:, 0] == t, 2]) for t in ts])
        out["nbt"] = np.array([r[r[:, 0] == t, 5].mean() for t in ts])
        out["jlog"] = np.array([r[r[:, 0] == t, 7].min() for t in ts])
    return out


S = {"원래": stats(a.orig), "(a) ε 고정": stats(a.fix), "(b) 적응 LM": stats(a.lm)}
if a.rel:
    S["원래 재실행"] = stats(a.rel)
C = {"원래": "#444444", "원래 재실행": "#999999", "(a) ε 고정": "#1f77b4", "(b) 적응 LM": "#d62728"}
LS = {"원래 재실행": "--"}

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax = ax.ravel()
for k, s in S.items():
    kw = dict(color=C[k], ls=LS.get(k, "-"), lw=1.6, label=k)
    ax[0].plot(s["t"], s["rmse"], **kw)
    ax[1].plot(s["t"], s["mv"], **kw)
    if "jmin" in s:
        ax[2].plot(s["t"], np.maximum(s["jmin"], 1e-6), **kw)
        ax[3].plot(s["t"], s["n05"], **kw)
    if "eps_med" in s:
        ax[4].plot(s["t"], s["eps_med"], **kw)
    elif k == "원래":
        ax[4].plot(s["t"], s["eps"], **kw)
    if "step" in s:
        ax[5].plot(s["t"], np.maximum(s["step"], 1e-12), **kw)
s0 = S["원래"]
ax[1].plot(s0["t"], s0["gmv"], color="k", lw=2.4, alpha=0.35, label="정답")
ax[0].set_title("RMSE (%L)"); ax[1].set_title("입자 이동량: 직전 프레임 대비 평균 (%L)")
ax[2].set_title("장벽 셀 det 최소 (정지 대비)"); ax[3].set_title("det < 0.5 사면체 수")
ax[4].set_title("감쇠 ε (프레임 안 중앙값)"); ax[5].set_title("걸음 크기 |sΔ| (프레임 안 중앙값)")
for k in (2, 4, 5):
    ax[k].set_yscale("log")
for x in ax:
    x.set_xlabel("프레임"); x.grid(alpha=0.3, which="both"); x.legend(fontsize=9)
fig.suptitle(f"{a.title} 사면체 메쉬 riem: 감쇠 ε 설계 비교 (바꾼 것은 ε 와 가능성 선탐색뿐)", fontsize=14)
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print(f"[그림] {a.out}")
for k, s in S.items():
    stall = np.where((s["gmv"] > 0.05) & (s["mv"] < 0.1 * s["gmv"]))[0]
    print(f"[{k}] RMSE 평균 {s['rmse'].mean():.3f}%  마지막 {s['rmse'][-1]:.3f}%  "
          + (f"셀 det 최소 {s['jmin'].min():.4g}  J≤0 최대 {int(s['nneg'].max())}  J<0.5 마지막 {int(s['n05'][-1])}/{int(s['ncell'])}  "
             if "jmin" in s else "")
          + f"멈춤(이동 < 정답 10%) 첫 프레임 {int(s['t'][stall[0]]) if stall.size else '-'}  "
          + f"이동/정답 (마지막 10 프레임) {s['mv'][-10:].sum() / max(s['gmv'][-10:].sum(), 1e-12):.3f}", flush=True)
    if "eps_med" in s:
        print(f"    ε 처음 {s['eps_med'][0]:.3e} 끝 {s['eps_med'][-1]:.3e}  걸음 중앙 처음 {s['step'][2]:.3e} 끝 {s['step'][-1]:.3e}  "
              f"되돌림 평균 {s['nbt'].mean():.1f}", flush=True)
    np.savez(a.out.replace(".png", "_" + {"원래": "orig", "원래 재실행": "rel", "(a) ε 고정": "fix",
                                          "(b) 적응 LM": "lm"}[k] + ".npz"),
             **{kk: v for kk, v in s.items() if isinstance(v, np.ndarray)})
