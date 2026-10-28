"""lego 새 바닥(spstick) 결과의 입자 F 찌그러짐 분포 영상 (프레임 진행).

세 지표 (F 특이값 σ1≥σ2≥σ3): log10 max_i|σ_i−1|, det F, log10(σ1/σ3).
겹침: 정답 PG (lbdata h5 F (N,9) float32) | 새 바닥 출력만 최적화 (spstick/roll_ov_T.pt)
      | 새 바닥 다섯 궤적 모델 (lego_psi_spstick/roll_mm_T.pt, float32 F 재롤아웃 덤프).
덤프 F[t] ↔ 정답 h5 프레임 t+1 (legofd.py·psi_color_lb.py 와 같은 대응). 시뮬·롤아웃 없음, 기존 덤프만 읽는다.
히스토그램 y 는 로그, 축 범위는 영상 전체(세 실행·전 프레임)에서 고정.

  python exe/spstick_fdist.py --traj L000 --out DIR
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from tqdm import tqdm  # noqa: E402

for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T_ = a.traj
os.makedirs(f"{a.out}/fr", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
gt_dir = f"{W}/lbdata/test_low/L000" if T_ == "L000" else f"{W}/lbdata/test/{T_}"
pt = f"{W}/lbpt/test_low/L000.pt" if T_ == "L000" else f"{W}/lbpt/test/{T_}.pt"
RUNS = [("gt", "정답 (PG)", "#222222", None),
        ("sp", "새 바닥 출력만 최적화", "#d9480f", f"{W}/lbours/spstick/roll_ov_{T_}.pt"),
        ("mm", "새 바닥 다섯 궤적 모델", "#1c7ed6", f"{W}/lbours/lego_psi_spstick/roll_mm_{T_}.pt")]
sel = np.asarray(torch.load(pt, map_location="cpu", weights_only=False, mmap=True)["sel"]).astype(np.int64)
fs = sorted(glob.glob(f"{gt_dir}/sim_*.h5"))
with h5py.File(fs[0]) as h:
    NF = h["F"].shape[0]
assert np.array_equal(sel, np.arange(NF)), "sel 이 항등이 아니다"
N = NF


def metrics(F):
    F = torch.as_tensor(F).to(dev).float().reshape(-1, 3, 3)
    s = torch.linalg.svdvals(F)
    dev_ = (s - 1).abs().max(1).values
    J = torch.det(F)
    an = s[:, 0] / s[:, 2].clamp_min(1e-12)
    return (torch.log10(dev_ + 1e-8).cpu().numpy(), J.cpu().numpy(), torch.log10(an).cpu().numpy(),
            dev_.cpu().numpy(), an.cpu().numpy())


M = {}   # run -> (T, 3, N) float32 : log dev, det, log aniso
RAW = {}  # run -> (T, 2, N) : raw dev, raw aniso (통계용)
for key, lab, _, p in RUNS:
    if key == "gt":
        Fs = [np.array(h5py.File(fs[t], "r")["F"], dtype=np.float32)[sel] for t in tqdm(range(1, len(fs)), desc="정답 F 읽기")]
    else:
        D = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        assert D["F"].dtype == torch.float32, f"{p}: F 가 float32 가 아니다 ({D['F'].dtype})"
        assert D["F"].shape[1] == N, (D["F"].shape, N)
        Fs = D["F"]
    T = len(Fs)
    A = np.zeros((T, 3, N), np.float32); B = np.zeros((T, 2, N), np.float32)
    for t in tqdm(range(T), desc=f"{key} 특이값"):
        ld, J, la, d_, an = metrics(Fs[t])
        A[t] = (ld, J, la); B[t] = (d_, an)
    M[key] = A; RAW[key] = B
    print(f"[{key}] {lab}: {T} 프레임, N {N}, 원본 {p or gt_dir}", flush=True)
T = min(v.shape[0] for v in M.values())
assert all(v.shape[0] == T for v in M.values()), {k: v.shape for k, v in M.items()}
# 정렬 검산: 덤프 gt 위치 vs h5 프레임 1
D = torch.load(RUNS[1][3], map_location="cpu", weights_only=False, mmap=True)
if "gt" in D:
    x1 = np.array(h5py.File(fs[1], "r")["x"], dtype=np.float32)[sel]
    print("[정렬 확인] sp 덤프 gt[0] vs 정답 h5 프레임 1 위치 차 최대", float(np.abs(D["gt"][0].float().numpy() - x1).max()), flush=True)

# 고정 축: 영상 전체·세 실행
allv = [np.concatenate([M[k][:, i].ravel() for k in M]) for i in range(3)]
rng = []
for i, v in enumerate(allv):
    if i == 1:  # det: 극단값 몇 개가 축을 지배하지 않도록 1e-5 / 99.999% 분위수, 0 은 반드시 포함
        lo, hi = np.quantile(v, [1e-5, 1 - 1e-5]); lo = min(lo, 0.0); hi = max(hi, 1.0)
        pad = 0.03 * (hi - lo); lo, hi = lo - pad, hi + pad
    else:
        lo, hi = float(v.min()), float(v.max()); pad = 0.02 * (hi - lo); lo, hi = lo - pad, hi + pad
    rng.append((float(lo), float(hi)))
del allv
NB = 160
edges = [np.linspace(lo, hi, NB + 1) for lo, hi in rng]
H = {k: np.zeros((T, 3, NB)) for k in M}
ST = {k: np.zeros((T, 7)) for k in M}  # dev 중앙·99, det 중앙·99, aniso 중앙·99, det<0
OUT = {k: np.zeros((T, 3), int) for k in M}  # 범위 밖(끝 칸에 합침) 개수
for k in M:
    for t in range(T):
        for i in range(3):
            v = M[k][t, i]
            OUT[k][t, i] = int(((v < rng[i][0]) | (v > rng[i][1])).sum())
            H[k][t, i] = np.histogram(np.clip(v, edges[i][0], edges[i][-1]), edges[i])[0] / N
        d_, an = RAW[k][t]; J = M[k][t, 1]
        ST[k][t] = (np.median(d_), np.quantile(d_, .99), np.median(J), np.quantile(J, .99),
                    np.median(an), np.quantile(an, .99), (J < 0).sum())
np.savez(f"{a.out}/fdist_{T_}.npz", edges0=edges[0], edges1=edges[1], edges2=edges[2],
         **{f"H_{k}": H[k] for k in H}, **{f"ST_{k}": ST[k] for k in ST}, **{f"OUT_{k}": OUT[k] for k in OUT})
json.dump({"traj": T_, "N": N, "frames": T, "range": rng, "bins": NB,
           "runs": {k: (p or gt_dir) for k, _, _, p in RUNS},
           "out_of_range_total": {k: OUT[k].sum(0).tolist() for k in OUT}}, open(f"{a.out}/range_{T_}.json", "w"), indent=1, ensure_ascii=False)
with open(f"{a.out}/stats_{T_}.txt", "w") as f:
    f.write(f"# lego {T_} 입자 F 찌그러짐, N {N}; 열: max|σ−1| 중앙·99%, det F 중앙·99%, σ1/σ3 중앙·99%, det<0 개수\n")
    for t in sorted({0, T // 2 - 1, T - 1}):
        for k, lab, _, _ in RUNS:
            s = ST[k][t]
            f.write(f"프레임 {t+1:3d} {k:3s} {s[0]:.4e} {s[1]:.4e}  {s[2]:.4f} {s[3]:.4f}  {s[4]:.4f} {s[5]:.4f}  {int(s[6])}\n")
print(open(f"{a.out}/stats_{T_}.txt").read(), flush=True)

XL = ["log10 max|σᵢ−1|  (최대 늘임/압축)", "det F  (부피비)", "log10(σ1/σ3)  (비등방)"]
SN = [("중앙", 0, 1, "{:.2e}"), ("det", 2, 3, "{:.3f}"), ("σ1/σ3", 4, 5, "{:.3f}")]
ylo = 0.5 / N
cen = [0.5 * (e[1:] + e[:-1]) for e in edges]
dpi = 100
for t in tqdm(range(T), desc="프레임 그림"):
    fig, ax = plt.subplots(1, 3, figsize=(19.2, 6.4), dpi=dpi)
    for i in range(3):
        for k, lab, c, _ in RUNS:
            h = H[k][t, i]
            ax[i].step(cen[i], np.where(h > 0, h, ylo * 0.5), where="mid", color=c, lw=1.6, label=lab)
            ax[i].fill_between(cen[i], np.where(h > 0, h, ylo * 0.5), ylo * 0.5, step="mid", color=c, alpha=0.10)
        ax[i].set_yscale("log"); ax[i].set_ylim(ylo, 1.2); ax[i].set_xlim(*rng[i])
        ax[i].set_xlabel(XL[i]); ax[i].set_ylabel("입자 비율 (로그 축)")
        if i == 1:
            ax[i].axvline(0, color="#c92a2a", lw=0.9, ls=":"); ax[i].axvline(1, color="#888", lw=0.8, ls=":")
        else:
            ax[i].axvline(0 if i == 2 else -8, color="#888", lw=0.8, ls=":")
        ax[i].spines[["top", "right"]].set_visible(False)
        lines = []
        for k, lab, c, _ in RUNS:
            s = ST[k][t]; j0, j1 = SN[i][1], SN[i][2]
            v0 = s[j0]; v1 = s[j1]
            fmt = "{:.2e}" if i == 0 else "{:.3f}"
            lines.append((f"{lab}: 중앙 {fmt.format(v0)} · 99% {fmt.format(v1)} · det<0 {int(s[6])}", c))
        for r, (tx, c) in enumerate(lines):
            ax[i].text(0.02, 0.98 - 0.06 * r, tx, transform=ax[i].transAxes, va="top", ha="left", fontsize=8.5, color=c)
    ax[0].set_title(f"최대 늘임/압축 max|σᵢ−1| (원값 표기)", fontsize=11)
    ax[1].set_title("det F  (빨간 점선 0, 회색 1)", fontsize=11)
    ax[2].set_title("비등방 σ1/σ3 (원값 표기)", fontsize=11)
    ax[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"lego {T_} · 새 바닥(softplus + 미끄러짐 방지) · 입자 F 찌그러짐 분포 · 프레임 {t+1}/{T}"
                 f"   (입자 {N}개 전부, 축 영상 전체 고정, 범위 밖은 끝 칸에 합침)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{a.out}/fr/f{t:04d}.png", dpi=dpi)
    plt.close(fig)
print("SPFDIST_TRAJ_OK", T_, T, flush=True)
