"""사면체 메쉬 riem: 장벽 계량 판(원래 rel · ε 고정) 과 제약 처리 판(투영 ADMM · 증강 라그랑지안) 비교 그림.

topo_eps_cmp.py 의 stats 를 그대로 쓰고, proj·alm 의 cons_log(σ_min·δ 위반 수) 를 더 그린다.
패널: RMSE · 정답 대비 이동 비율 · 셀 det 최소 · det<0.5 사면체 수 · δ 위반 수 · 걸음 크기.
멈춤 = 정답 이동 > 0.05 %L 인데 모델 이동이 그 10 % 미만인 첫 프레임 (topo_eps_cmp 와 같은 정의). 요약은 <out>.json.

  python exe/topo_pa_cmp.py --flow flow_wolf.npz --run "원래 (riem rel)=eps/wolf_rel.npz" \
      --run "ε 고정=eps/wolf_fix.npz" --run "투영 ADMM=pa/wolf_proj.npz" --run "ALM=pa/wolf_alm.npz" \
      --out wolf_tetmesh_proj_alm.png --title wolf
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--run", action="append", required=True, help="라벨=경로 (순서대로 그린다)")
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--font", default="/home/dkta/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib import font_manager                                  # noqa: E402
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False

GT = np.load(a.flow)["traj"].astype(np.float64)                      # [T+1,N,3]


def stats(path):
    d = np.load(path, allow_pickle=True)
    L = float(d["L"]); m = d["metrics"]; T = m.shape[0]
    tr = np.concatenate([GT[:1], d["traj"].astype(np.float64)], 0)
    mv = np.linalg.norm(tr[1:] - tr[:-1], axis=-1).mean(1) / L * 100
    gmv = np.linalg.norm(GT[1:T + 1] - GT[:T], axis=-1).mean(1) / L * 100
    out = dict(t=m[:, 0], rmse=100 * m[:, 1], cd=100 * m[:, 2], mv=mv, gmv=gmv)
    if "tet_log" in d.files and d["tet_log"].size:
        tl = d["tet_log"]
        out.update(jmin=tl[:, 1], n05=tl[:, 2], nneg=tl[:, 3], ncell=tl[0, 4])
    if "riem_log" in d.files and d["riem_log"].size:
        r = d["riem_log"]
        ts = np.unique(r[:, 0])
        out["step"] = np.array([np.median(r[r[:, 0] == t, 3]) for t in ts])
    if "cons_log" in d.files and d["cons_log"].size:
        c = d["cons_log"]
        out.update(smin=c[:, 2], nsig=c[:, 3], ndet=c[:, 4])
    else:                                                            # 장벽 판: 저장된 노드로 σ_min·δ 위반을 다시 잰다
        if "nodes_u" in d.files:
            Xn, cells, U = d["Xn"].astype(np.float64), d["cells"], d["nodes_u"].astype(np.float64)
            X0 = Xn[cells]; D0 = np.transpose(X0[:, 1:] - X0[:, :1], (0, 2, 1)); D0i = np.linalg.inv(D0)
            sm, ns, nd = [], [], []
            for u in U:
                X = (Xn + u)[cells]; F = np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1)) @ D0i
                J = np.linalg.det(F); s = np.linalg.svd(F, compute_uv=False)[:, 2] * np.sign(J)
                sm.append(s.min()); ns.append((s < a.delta).sum()); nd.append((J < a.delta).sum())
            out.update(smin=np.array(sm), nsig=np.array(ns), ndet=np.array(nd))
    st = np.where((out["gmv"] > 0.05) & (out["mv"] < 0.1 * out["gmv"]))[0]
    out["stall"] = int(out["t"][st[0]]) if st.size else -1
    return out


S = {}
for r in a.run:
    k, p = r.split("=", 1)
    S[k] = stats(p)
COL = ["#444444", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax = ax.ravel()
for i, (k, s) in enumerate(S.items()):
    kw = dict(color=COL[i % len(COL)], lw=1.6, label=k)
    ax[0].plot(s["t"], s["rmse"], **kw)
    ax[1].plot(s["t"], s["mv"] / np.maximum(s["gmv"], 1e-12), **kw)
    if "jmin" in s:
        ax[2].plot(s["t"], np.maximum(s["jmin"], 1e-6), **kw)
        ax[3].plot(s["t"], s["n05"], **kw)
    if "nsig" in s:
        ax[4].plot(s["t"], s["nsig"], **kw)
        ax[4].plot(s["t"], s["ndet"], color=kw["color"], lw=1.0, ls="--")
    if "step" in s:
        ax[5].plot(s["t"], np.maximum(s["step"], 1e-12), **kw)
    if s["stall"] > 0:
        for j in (0, 1):
            ax[j].axvline(s["stall"], color=kw["color"], ls=":", lw=1.2)
ax[1].axhline(1.0, color="k", lw=2.0, alpha=0.35, label="정답 (1)")
ax[2].axhline(a.delta, color="k", lw=1.0, ls="--", alpha=0.5, label=f"δ = {a.delta}")
ax[0].set_title("RMSE (%L)")
ax[1].set_title("정답 대비 이동 비율 (직전 프레임 대비 평균 이동 / 정답 이동)")
ax[2].set_title("셀 det 최소 (정지 대비)")
ax[3].set_title("det < 0.5 사면체 수")
ax[4].set_title(f"δ 위반 사면체 수 (실선 σ_min < δ, 점선 det < δ)")
ax[5].set_title("걸음 크기 (프레임 안 중앙값)")
for k in (2, 5):
    ax[k].set_yscale("log")
ax[4].set_yscale("symlog", linthresh=1)
for x in ax:
    x.set_xlabel("프레임"); x.grid(alpha=0.3, which="both"); x.legend(fontsize=9)
fig.suptitle(f"{a.title} 사면체 메쉬: 장벽 계량(원래·ε 고정) 대 투영 ADMM·증강 라그랑지안 (점선 세로줄 = 멈춘 프레임)",
             fontsize=14)
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print(f"[그림] {a.out}")
SUM = {}
for k, s in S.items():
    SUM[k] = dict(rmse=float(s["rmse"].mean()), rmse_last=float(s["rmse"][-1]), cd=float(s["cd"].mean()),
                  stall=s["stall"],
                  move_ratio_last10=float(s["mv"][-10:].sum() / max(s["gmv"][-10:].sum(), 1e-12)),
                  **({"jmin": float(s["jmin"].min()), "nneg_max": int(s["nneg"].max()), "n05_last": int(s["n05"][-1]),
                      "ncell": int(s["ncell"])} if "jmin" in s else {}),
                  **({"smin": float(s["smin"].min()), "nsig_max": int(s["nsig"].max()), "nsig_last": int(s["nsig"][-1]),
                      "ndet_max": int(s["ndet"].max()), "ndet_last": int(s["ndet"][-1])} if "nsig" in s else {}))
    print(f"[{k}] " + "  ".join(f"{kk} {v:.4g}" if isinstance(v, float) else f"{kk} {v}" for kk, v in SUM[k].items()),
          flush=True)
json.dump(SUM, open(a.out.replace(".png", ".json"), "w"), ensure_ascii=False, indent=1)
