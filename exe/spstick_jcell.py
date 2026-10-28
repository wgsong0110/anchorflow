"""lego 새 바닥(spstick) 결과의 **셀(사면체)별 스텝 야코비안** J = I + ∇u 분포 영상.

입자에 누적된 F 가 아니라, 프레임 스텝 하나에서 사면체 4 꼭짓점 노드 변위(덤프 nodes 의 dpn)로
정해지는 셀 매핑의 야코비안이다. 입자가 들어 있는 사면체만 (사면체마다 한 번).
사면체 연결은 덤프에 없어서, 그 스텝 시작 입자 위치(0 스텝은 x0, 이후 pred[t-1])로 train_deform 과
같은 함수(simplex.grid_for_nodes·locate·active_nodes, node_h 고정·hz=h/√6·격자 하나)를 다시 불러
만든다 -- 재구성한 활성 노드 위치가 덤프 npos 와 일치하는지 프레임마다 검산하고, 어긋나면 멈춘다.
J 는 바닥 매핑(softplus·미끄러짐 방지) 이전의 순수 셀 매핑이다 (그 둘은 입자 단위 매핑이라 셀 값이 아니다).

  PYTHONPATH=<spstick 워크트리>/lib python exe/spstick_jcell.py --traj L000 --node_h 0.028284 --out DIR
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from tqdm import tqdm  # noqa: E402
from anchorflow import simplex as SX  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--n_nodes", type=float, default=32)
a = ap.parse_args()
T_ = a.traj
os.makedirs(f"{a.out}/fr", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = [("sp", "새 바닥 출력만 최적화", "#d9480f", f"{W}/lbours/spstick/roll_ov_{T_}.pt"),
        ("mm", "새 바닥 다섯 궤적 모델", "#1c7ed6", f"{W}/lbours/lego_psi_spstick/roll_mm_{T_}.pt")]
HZ = 1.0 / math.sqrt(6.0)
M, RAW, NC, CHK = {}, {}, {}, {}
for key, lab, _, p in RUNS:
    D = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    nodes = D["nodes"]; T = len(nodes)
    assert nodes is not None and T == D["pred"].shape[0], "노드 덤프 없음/길이 불일치"
    A, B, nc, chk = [], [], [], []
    for t in tqdm(range(T), desc=f"{key} 셀 야코비안"):
        xs = (D["x0"] if t == 0 else D["pred"][t - 1]).to(dev).float()
        npos_d, dpn, _sup, _pc = (q.to(dev) for q in nodes[t])
        lo, lat, nn = SX.grid_for_nodes(xs, a.n_nodes, hz_ratio=HZ, h_fix=a.node_h)
        idx, lam, aux = SX.locate(xs, lo, lat, nn)
        rows, uniq = SX.active_nodes(idx)
        npos = SX.node_pos(lo, lat, nn, uniq)
        if npos.shape != npos_d.shape or float((npos - npos_d.float()).abs().max()) > 1e-5:
            raise SystemExit(f"[중단] {key} 스텝 {t}: 재구성 노드 {tuple(npos.shape)} vs 덤프 {tuple(npos_d.shape)}"
                             f" -- 덤프 격자와 다르다 (서브스텝·격자 설정 차이 의심)")
        u, G = SX.g2p_jac_pre(rows, lam, aux, lat, dpn[:, :3].float())
        # 검산: 셀 매핑으로 옮긴 입자 vs 덤프 pred (바닥 매핑이 건드리지 않은 입자는 같아야 한다)
        dq = ((xs + u) - D["pred"][t].to(dev).float()).norm(dim=-1)
        chk.append((float((dq < 1e-5).float().mean()), float(dq.median())))
        tid = SX.tet_id(lo, lat, nn, aux)
        _, first = np.unique(tid.cpu().numpy(), return_index=True)
        Jc = torch.eye(3, device=dev) + G[torch.as_tensor(first, device=dev)]
        s = torch.linalg.svdvals(Jc); dv = (s - 1).abs().max(1).values; dt = torch.det(Jc)
        an = s[:, 0] / s[:, 2].clamp_min(1e-12)
        A.append(np.stack([torch.log10(dv + 1e-8).cpu().numpy(), dt.cpu().numpy(), torch.log10(an).cpu().numpy()]).astype(np.float32))
        B.append(np.stack([dv.cpu().numpy(), an.cpu().numpy()]).astype(np.float32))
        nc.append(len(first))
    M[key], RAW[key], NC[key], CHK[key] = A, B, np.array(nc), np.array(chk)
    print(f"[{key}] {lab}: {T} 스텝, 점유 사면체 수 {min(nc)}~{max(nc)}, 노드 재구성 전 스텝 일치; "
          f"셀 매핑 입자 = 덤프 pred 인 비율 평균 {CHK[key][:, 0].mean():.4f} (나머지는 바닥 매핑), 원본 {p}", flush=True)
T = len(M["sp"]); assert all(len(M[k]) == T for k in M)
rng = []
for i in range(3):
    v = np.concatenate([M[k][t][i] for k in M for t in range(T)])
    if i == 1:
        lo, hi = np.quantile(v, [1e-5, 1 - 1e-5]); lo = min(lo, 0.0); hi = max(hi, 1.0)
        pad = 0.03 * (hi - lo)
    else:
        lo, hi = float(v.min()), float(v.max()); pad = 0.02 * (hi - lo)
    rng.append((float(lo - pad), float(hi + pad)))
NB = 160
edges = [np.linspace(lo, hi, NB + 1) for lo, hi in rng]
cen = [0.5 * (e[1:] + e[:-1]) for e in edges]
H = {k: np.zeros((T, 3, NB)) for k in M}
ST = {k: np.zeros((T, 7)) for k in M}
for k in M:
    for t in range(T):
        n = M[k][t].shape[1]
        for i in range(3):
            H[k][t, i] = np.histogram(np.clip(M[k][t][i], edges[i][0], edges[i][-1]), edges[i])[0] / n
        d_, an = RAW[k][t]; J = M[k][t][1]
        ST[k][t] = (np.median(d_), np.quantile(d_, .99), np.median(J), np.quantile(J, .99),
                    np.median(an), np.quantile(an, .99), (J <= 0).sum())
np.savez(f"{a.out}/jcell_{T_}.npz", edges0=edges[0], edges1=edges[1], edges2=edges[2],
         **{f"H_{k}": H[k] for k in H}, **{f"ST_{k}": ST[k] for k in ST}, **{f"NC_{k}": NC[k] for k in NC},
         **{f"CHK_{k}": CHK[k] for k in CHK})
json.dump({"traj": T_, "steps": T, "range": rng, "bins": NB, "node_h": a.node_h, "hz_ratio": HZ,
           "runs": {k: p for k, _, _, p in RUNS}}, open(f"{a.out}/jrange_{T_}.json", "w"), indent=1, ensure_ascii=False)
with open(f"{a.out}/jstats_{T_}.txt", "w") as f:
    f.write(f"# lego {T_} 셀(사면체)별 스텝 야코비안 J=I+∇u, 점유 사면체만; 열: max|σ−1| 중앙·99%, det J 중앙·99%, σ1/σ3 중앙·99%, det≤0 개수 / 셀 수\n")
    for t in sorted({0, T // 2 - 1, T - 1}):
        for k, lab, _, _ in RUNS:
            s = ST[k][t]
            f.write(f"스텝 {t+1:3d} {k:3s} {s[0]:.4e} {s[1]:.4e}  {s[2]:.5f} {s[3]:.5f}  {s[4]:.5f} {s[5]:.5f}  {int(s[6])} / {NC[k][t]}\n")
print(open(f"{a.out}/jstats_{T_}.txt").read(), flush=True)

XL = ["log10 max|σᵢ−1|  (최대 늘임/압축)", "det J  (셀 부피비)", "log10(σ1/σ3)  (비등방)"]
ylo = 0.5 / max(int(NC[k].max()) for k in NC)
dpi = 100
for t in tqdm(range(T), desc="프레임 그림"):
    fig, ax = plt.subplots(1, 3, figsize=(19.2, 6.4), dpi=dpi)
    for i in range(3):
        for k, lab, c, _ in RUNS:
            h = np.where(H[k][t, i] > 0, H[k][t, i], ylo * 0.5)
            ax[i].step(cen[i], h, where="mid", color=c, lw=1.6, label=f"{lab} (셀 {NC[k][t]}개)")
            ax[i].fill_between(cen[i], h, ylo * 0.5, step="mid", color=c, alpha=0.10)
        ax[i].set_yscale("log"); ax[i].set_ylim(ylo, 1.2); ax[i].set_xlim(*rng[i])
        ax[i].set_xlabel(XL[i]); ax[i].set_ylabel("셀 비율 (로그 축)")
        if i == 1:
            ax[i].axvline(0, color="#c92a2a", lw=0.9, ls=":"); ax[i].axvline(1, color="#888", lw=0.8, ls=":")
        ax[i].spines[["top", "right"]].set_visible(False)
        fmt = "{:.2e}" if i == 0 else "{:.4f}"
        for r, (k, lab, c, _) in enumerate(RUNS):
            s = ST[k][t]
            ax[i].text(0.02, 0.98 - 0.06 * r, f"{lab}: 중앙 {fmt.format(s[2*i])} · 99% {fmt.format(s[2*i+1])} · det≤0 {int(s[6])}",
                       transform=ax[i].transAxes, va="top", ha="left", fontsize=8.5, color=c)
    ax[0].set_title("최대 늘임/압축 max|σᵢ−1| (원값 표기)", fontsize=11)
    ax[1].set_title("det J  (빨간 점선 0, 회색 1)", fontsize=11)
    ax[2].set_title("비등방 σ1/σ3 (원값 표기)", fontsize=11)
    ax[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"lego {T_} · 새 바닥(softplus + 미끄러짐 방지) · 셀(사면체)별 스텝 야코비안 J = I + ∇u · 프레임 {t}→{t+1}"
                 f"   (점유 사면체만, 바닥 매핑 전, 축 영상 전체 고정, 범위 밖은 끝 칸에 합침)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{a.out}/fr/f{t:04d}.png", dpi=dpi)
    plt.close(fig)
print("SPJCELL_TRAJ_OK", T_, T, flush=True)
