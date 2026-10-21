"""셀(사면체)별 입자 F 퍼짐 비교 -- 정답(PG h5) 과 우리 롤아웃 덤프들.

셀 분할: 우리 격자 규칙 그대로 (simplex.grid_for_nodes(x, n_nodes, hz_ratio=1/√6, h_fix=node_h, off=None) ->
locate -> tet_id, Kuhn 사면체). 궤적마다 **자기 현재 입자 위치**에 프레임마다 새로 깐다 (셀끼리 짝짓지 않음).
입자 >= 4 개인 셀만 쓴다. 덤프에 노드 위치(nodes)가 있으면 프레임마다 재구성한 활성 노드 위치가 덤프와
같은지 검산하고, 어긋나면 멈춘다 (격자 규칙이 실행과 같다는 확인).

셀별 지표 (질량가중, m_p = 정답 0 프레임 배치로 잰 PG 입자 질량 = deform_model.traj_mass, 영역 밀도 포함, 모든 궤적 공통):
  0 σ_F      = sqrt(Σ m‖F_p − F̄_c‖²_F / Σ m),  F̄_c = 질량가중 평균
  1 σ_logU   = 같은 식, log U_p (F = R U 극분해, log U = V diag(log s) Vᵀ)
  2 σ(s1)    = 입자 최대 특이값 s1 의 셀 내 질량가중 표준편차
  3 방향 각  = 입자 최대 늘임 방향 v_p (V 의 첫 열, 부호 무시) 과 셀 방향(Σ m s1 v vᵀ 의 최대 고유벡터) 사이 각의
               질량가중 평균 [도]; s1/s2 < 1.01 입자는 뺀다 (뺀 비율 기록), 남은 입자 2 개 미만이면 셀 값 없음
  4 σ(log det F) = 질량가중 표준편차; det F <= 0 입자는 뺀다 (개수 기록), 남은 입자 2 개 미만이면 셀 값 없음
  5 회전 각  = 입자 R_p 와 셀 평균 회전(F̄_c 극분해의 R) 사이 각의 질량가중 평균 [도]
극분해는 det R = +1 이 되게 (det F < 0 이면 마지막 특이벡터 부호를 뒤집는다). 계산은 float64.

  python exe/cell_fspread.py --config CFG --node_h 0.056569 --out OUT \
     --traj gt:정답:#222222:h5:DIR --traj lin:선형:#d9480f:dump:roll_ov.pt --traj c1:C1:#1c7ed6:dump:roll_ov.pt \
     --video ficus_c1.mp4 --mp4 out.mp4 --ts ts.png
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX  # noqa: E402
from anchorflow import deform_model as DM  # noqa: E402

MET = [("sF", r"$\sigma_F$", "1"), ("slogU", r"$\sigma_{\log U}$", "1"), ("ss1", r"$\sigma(s_1)$", "1"),
       ("ang_dir", "최장축 방향 각", "도"), ("sldet", r"$\sigma(\log\det F)$", "1"), ("ang_rot", "회전 각", "도")]
MTITLE = ["1. F 퍼짐 " + r"$\sigma_F$", "2. log U 퍼짐 " + r"$\sigma_{\log U}$ (회전 제거)",
          "3. 최장축 늘임 퍼짐 " + r"$\sigma(s_1)$", "4. 최장축 방향 퍼짐 (각, 도)",
          "5. 부피 퍼짐 " + r"$\sigma(\log\det F)$", "6. 회전 퍼짐 (각, 도)"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="정답 궤적 config (질량: n_grid·grid_lim·영역 밀도)")
ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--hz_ratio", type=float, default=SX.HZ_CUBE)
ap.add_argument("--n_nodes", type=float, default=32)
ap.add_argument("--traj", action="append", required=True, help="키:이름:색:h5|dump:경로 (첫 번째가 정답 h5)")
ap.add_argument("--out", required=True)
ap.add_argument("--video", default="", help="렌더 영상 (칸 순서 = --traj 순서), 프레임 n = 시뮬 프레임 n")
ap.add_argument("--mp4", default="")
ap.add_argument("--ts", default="")
ap.add_argument("--fps", type=float, default=0, help="0 이면 --video 의 fps")
ap.add_argument("--min_pts", type=int, default=4)
ap.add_argument("--dir_ratio", type=float, default=1.01)
ap.add_argument("--bins", type=int, default=80)
ap.add_argument("--max_frames", type=int, default=0, help="시험용: 앞 프레임만")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
TR = []
for s in a.traj:
    k, lab, col, kind, p = s.split(":", 4)
    TR.append(dict(key=k, lab=lab, col=col, kind=kind, path=p))
cfg = json.load(open(a.config))
cfg.setdefault("n_grid", 50); cfg.setdefault("grid_lim", 2.0)
LOG = open(f"{a.out}/cell_fspread.log", "a")


def say(*m):
    s = " ".join(str(q) for q in m)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


say(f"[설정] node_h {a.node_h} hz_ratio {a.hz_ratio:.6f} n_nodes {a.n_nodes} (h_fix 라 무관) off None, 입자 >= {a.min_pts}, "
    f"방향 s1/s2 >= {a.dir_ratio}")


# ---------------- 궤적 읽기 ----------------
def load_traj(t):
    if t["kind"] == "h5":
        fs = sorted(glob.glob(os.path.join(t["path"], "**", "sim_*.h5"), recursive=True),
                    key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
        X, F = [], []
        for f in tqdm(fs, desc=f"{t['key']} h5 읽기"):
            with h5py.File(f, "r") as h:
                assert h["x"].dtype == np.float32 and h["f_tensor"].dtype == np.float32, (h["x"].dtype, h["f_tensor"].dtype)
                X.append(torch.from_numpy(np.array(h["x"]).T.copy()))
                F.append(torch.from_numpy(np.array(h["f_tensor"]).T.reshape(-1, 3, 3).copy()))
        if float(F[0].abs().max()) == 0.0:
            F[0] = torch.eye(3).expand_as(F[0]).clone()      # PG 0 프레임 미초기화 -> 항등 (b670068 과 같게)
            say(f"[{t['key']}] 0 프레임 F 가 0 (PG 미초기화) -> 항등")
        t["X"], t["F"], t["nodes"] = X, F, None
        say(f"[{t['key']}] h5 {len(fs)} 프레임, 입자 {X[0].shape[0]}, x/F float32")
    else:
        D = torch.load(t["path"], map_location="cpu", weights_only=False, mmap=True)
        assert D.get("F") is not None, "덤프에 F 없음"
        say(f"[{t['key']}] 덤프 F dtype {D['F'].dtype} shape {tuple(D['F'].shape)}, pred {D['pred'].dtype} "
            f"{tuple(D['pred'].shape)}, t0 {D.get('t0')}, nodes {'있음 ' + str(len(D['nodes'])) if D.get('nodes') else '없음'}")
        if D["F"].dtype != torch.float32:
            say(f"[중단] {t['key']} F 가 {D['F'].dtype} -- float32 아님"); sys.exit(3)
        assert int(D.get("t0", 0)) == 0
        L = D["F"].shape[0]
        I = torch.eye(3).expand(D["F"].shape[1], 3, 3)
        t["X"] = [D["x0"].float()] + [D["pred"][i] for i in range(L)]     # 덤프 pred[i] = 프레임 i+1
        t["F"] = [I] + [D["F"][i] for i in range(L)]
        t["nodes"] = D.get("nodes")
    return t


for t in TR:
    load_traj(t)
T = min(len(t["X"]) for t in TR)
if a.max_frames:
    T = min(T, a.max_frames)
N = TR[0]["X"][0].shape[0]
for t in TR:
    assert t["X"][0].shape[0] == N, "입자 수가 다르다"
    if len(t["X"]) != T:
        say(f"[주의] {t['key']} 프레임 {len(t['X'])} -> 공통 {T}")
say(f"[공통] 프레임 0..{T - 1}, 입자 {N}")
# 0 프레임 위치가 같은지 (같은 입자 집합)
for t in TR[1:]:
    d0 = float((t["X"][0].float() - TR[0]["X"][0].float()).abs().max())
    say(f"[검산] {t['key']} 0 프레임 위치 vs 정답 최대 차 {d0:.3e}")
DM.REGION_DENSITY[0] = True
MASS = DM.traj_mass({"cfg": cfg, "x": [TR[0]["X"][0].float()]}, dev=dev).double()
say(f"[질량] 합 {float(MASS.sum()):.6g}, 최소 {float(MASS.min()):.4g} 최대 {float(MASS.max()):.4g} "
    f"(n_grid {cfg['n_grid']}, grid_lim {cfg['grid_lim']}, 영역 밀도 {[r.get('density') for r in cfg.get('additional_material_params', [])]})")


def proper_svd(F):
    W, S, Vh = torch.linalg.svd(F)
    neg = torch.linalg.det(W @ Vh) < 0
    W = W.clone(); W[neg, :, 2] *= -1
    return W, S, Vh


@torch.no_grad()
def cell_stats(x32, F32, nodes_t=None):
    lo, lat, nn = SX.grid_for_nodes(x32, a.n_nodes, hz_ratio=a.hz_ratio, off=None, h_fix=a.node_h)
    idx, _lam, aux = SX.locate(x32, lo, lat, nn)
    chk = None
    if nodes_t is not None:
        _rows, uniq = SX.active_nodes(idx)
        npos = SX.node_pos(lo, lat, nn, uniq)
        nd = nodes_t[0].to(dev).float()
        if npos.shape != nd.shape:
            return None, f"노드 수 {tuple(npos.shape)} vs 덤프 {tuple(nd.shape)}"
        chk = float((npos - nd).abs().max())
        if chk > 1e-5:
            return None, f"노드 위치 최대 차 {chk:.3e}"
    tid = SX.tet_id(lo, lat, nn, aux)
    u, inv, cnt = torch.unique(tid, return_inverse=True, return_counts=True)
    n_occ = int(u.numel())
    ok_c = cnt >= a.min_pts
    newid = torch.cumsum(ok_c.long(), 0) - 1
    keep = ok_c[inv]
    c = newid[inv][keep]
    C = int(ok_c.sum())
    m = MASS[keep]
    F = F32.to(dev)[keep].double()
    W, S, Vh = proper_svd(F)
    R = W @ Vh
    V = Vh.transpose(1, 2)
    logU = V @ torch.diag_embed(torch.log(S.clamp_min(1e-300))) @ Vh
    detF = torch.linalg.det(F)

    def wsum(q, w=m):
        sh = (C,) + tuple(q.shape[1:])
        return torch.zeros(sh, device=dev, dtype=torch.float64).index_add_(0, c, q * w.reshape((-1,) + (1,) * (q.dim() - 1)))

    M = wsum(torch.ones_like(m))
    out = torch.full((C, 6), float("nan"), device=dev, dtype=torch.float64)
    # 1 σ_F
    Fb = wsum(F) / M[:, None, None]
    out[:, 0] = (wsum(((F - Fb[c]) ** 2).sum((1, 2))) / M).sqrt()
    # 2 σ_logU
    Lb = wsum(logU) / M[:, None, None]
    out[:, 1] = (wsum(((logU - Lb[c]) ** 2).sum((1, 2))) / M).sqrt()
    # 3 σ(s1)
    s1 = S[:, 0]
    s1b = wsum(s1) / M
    out[:, 2] = (wsum((s1 - s1b[c]) ** 2) / M).clamp_min(0).sqrt()
    # 4 방향 각
    vd = V[:, :, 0]
    okd = (S[:, 0] / S[:, 1].clamp_min(1e-300)) >= a.dir_ratio
    md = m * okd
    Tn = wsum(vd[:, :, None] * vd[:, None, :], md * s1)
    ev = torch.linalg.eigh(Tn)[1][:, :, 2]
    ang = torch.rad2deg(torch.acos((vd * ev[c]).sum(-1).abs().clamp(0, 1)))
    nd_ = torch.zeros(C, device=dev, dtype=torch.float64).index_add_(0, c, okd.double())
    Md = wsum(torch.ones_like(m), md)
    out[:, 3] = torch.where(nd_ >= 2, wsum(ang, md) / Md.clamp_min(1e-300), torch.full_like(Md, float("nan")))
    # 5 σ(log det F)
    okj = detF > 0
    mj = m * okj
    lj = torch.log(detF.clamp_min(1e-300))
    Mj = wsum(torch.ones_like(m), mj)
    lb = wsum(lj, mj) / Mj.clamp_min(1e-300)
    nj = torch.zeros(C, device=dev, dtype=torch.float64).index_add_(0, c, okj.double())
    out[:, 4] = torch.where(nj >= 2, (wsum((lj - lb[c]) ** 2, mj) / Mj.clamp_min(1e-300)).clamp_min(0).sqrt(),
                            torch.full_like(Mj, float("nan")))
    # 6 회전 각
    Wb, _Sb, Vhb = proper_svd(Fb)
    Rb = Wb @ Vhb
    cth = ((Rb[c].transpose(1, 2) @ R).diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    out[:, 5] = wsum(torch.rad2deg(torch.acos(cth.clamp(-1, 1)))) / M
    info = dict(n_occ=n_occ, n_cells=C, n_pts_in_cells=int(keep.sum()), n_pts_total=int(x32.shape[0]),
                dir_excl_frac=float(1.0 - okd.double().mean()), dir_cells=int((nd_ >= 2).sum()),
                det_nonpos=int((~okj).sum()), ldet_cells=int((nj >= 2).sum()),
                grid_lo=[float(q) for q in lo], grid_nn=[int(q) for q in nn], node_chk=chk)
    return out.float().cpu().numpy(), info


# ---------------- 계산 ----------------
VALS = {t["key"]: [] for t in TR}
INFO = {t["key"]: [] for t in TR}
for t in TR:
    for f in tqdm(range(T), desc=f"{t['key']} 셀 지표"):
        nod = None
        if t["nodes"] is not None and f < len(t["nodes"]):
            nod = t["nodes"][f]       # 스텝 f 의 격자 = 프레임 f 위치에서 깐 것
        v, info = cell_stats(t["X"][f].to(dev).float(), t["F"][f], nod)
        if v is None:
            say(f"[중단] {t['key']} 프레임 {f}: 격자 재구성이 덤프와 다르다 -- {info}"); sys.exit(4)
        VALS[t["key"]].append(v); INFO[t["key"]].append(info)
    nchk = sum(1 for q in INFO[t["key"]] if q["node_chk"] is not None)
    mx = max([q["node_chk"] for q in INFO[t["key"]] if q["node_chk"] is not None], default=None)
    say(f"[격자 검산] {t['key']}: 덤프 노드와 비교한 프레임 {nchk}/{T}, 활성 노드 위치 최대 차 {mx}")
for t in TR:
    k = t["key"]
    off = np.cumsum([0] + [len(v) for v in VALS[k]])
    np.savez_compressed(f"{a.out}/cells_{k}.npz", vals=np.concatenate(VALS[k]), offsets=off,
                        metrics=np.array([q[0] for q in MET]))


def q(v, p):
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if v.size else float("nan")


SUM = {"metrics": [q_[0] for q_ in MET], "frames": T, "N": N, "node_h": a.node_h, "min_pts": a.min_pts,
       "dir_ratio": a.dir_ratio, "traj": {}}
for t in TR:
    k = t["key"]
    SUM["traj"][k] = dict(lab=t["lab"], path=t["path"], info=INFO[k],
                          med=[[q(v[:, j], 50) for j in range(6)] for v in VALS[k]],
                          p99=[[q(v[:, j], 99) for j in range(6)] for v in VALS[k]])
json.dump(SUM, open(f"{a.out}/summary.json", "w"), indent=0)
for f in (1, 40, 120):
    if f >= T:
        continue
    for t in TR:
        k = t["key"]; I_ = INFO[k][f]
        say(f"[프레임 {f}] {t['lab']}: 셀 {I_['n_cells']}/{I_['n_occ']} (입자 {I_['n_pts_in_cells']}/{N}), 방향 제외 "
            f"{100 * I_['dir_excl_frac']:.1f}%, det<=0 {I_['det_nonpos']}; 중앙/99% "
            + ", ".join(f"{MET[j][0]} {SUM['traj'][k]['med'][f][j]:.3e}/{SUM['traj'][k]['p99'][f][j]:.3e}" for j in range(6)))

# ---------------- 그림 ----------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
_FONTS = [os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")] + [p for p in fm.findSystemFonts() if "NotoSansCJK" in os.path.basename(p)]
for _p in _FONTS:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p); plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
say(f"[글꼴] {plt.rcParams['font.family']}")

# 가로축: 지표마다 영상 전체 고정 로그 구간. 하한 = 1..T-1 프레임 세 궤적 합친 양수 값의 0.1% 분위수, 상한 = 최대 x1.05.
EDG = []
for j in range(6):
    allv = np.concatenate([v[:, j] for t in TR for v in VALS[t["key"]][1:]])
    allv = allv[np.isfinite(allv) & (allv > 0)]
    lo_, hi_ = np.percentile(allv, 0.1), allv.max() * 1.05
    EDG.append(np.logspace(np.log10(lo_), np.log10(hi_), a.bins + 1))
    say(f"[구간] {MET[j][0]}: [{lo_:.3e}, {hi_:.3e}] 로그 {a.bins} 칸 (하한 아래·0 은 첫 칸에)")
HIST = {}
for t in TR:
    k = t["key"]; H = np.zeros((T, 6, a.bins))
    for f in range(T):
        for j in range(6):
            v = VALS[k][f][:, j]; v = v[np.isfinite(v)]
            if v.size:
                h, _ = np.histogram(np.clip(v, EDG[j][0], EDG[j][-1]), EDG[j])
                H[f, j] = h / v.size
    HIST[k] = H
np.savez_compressed(f"{a.out}/hist.npz", edges=np.stack(EDG), **{f"h_{k}": v for k, v in HIST.items()})
pos = np.concatenate([HIST[t["key"]][1:][HIST[t["key"]][1:] > 0] for t in TR])
YLO, YHI = pos.min() / 2, 1.5
FMT = lambda x: f"{x:.2e}"  # noqa: E731


def hist_fig(f, W_px, H_px, dpi=100):
    fig, axs = plt.subplots(2, 3, figsize=(W_px / dpi, H_px / dpi), dpi=dpi)
    for j, ax in enumerate(axs.ravel()):
        for t in TR:
            h = HIST[t["key"]][f, j]
            ax.stairs(h, EDG[j], color=t["col"], lw=1.6, label=t["lab"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(EDG[j][0], EDG[j][-1]); ax.set_ylim(YLO, YHI)
        ax.set_title(MTITLE[j], fontsize=13)
        ax.set_xlabel(("값 [도]" if MET[j][2] == "도" else "값 (무차원)") + " — 가로축 로그", fontsize=10)
        if j % 3 == 0:
            ax.set_ylabel("셀 비율 (세로축 로그)", fontsize=10)
        txt = "\n".join(f"{t['lab']}: 중앙 {FMT(SUM['traj'][t['key']]['med'][f][j])}  99% {FMT(SUM['traj'][t['key']]['p99'][f][j])}"
                        for t in TR)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.grid(alpha=0.25, which="major")
    axs[0, 2].legend(loc="upper right", fontsize=10)
    cnt = "   ".join(f"{t['lab']} 셀 {INFO[t['key']][f]['n_cells']}" for t in TR)
    fig.suptitle(f"셀(사면체, node_h {a.node_h}, 입자 ≥ {a.min_pts})별 입자 F 퍼짐 분포 · 프레임 {f}   ({cnt})   "
                 f"하한 아래 값·0 은 첫 칸에", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


if a.video and a.mp4:
    import imageio.v2 as imageio
    rd = imageio.get_reader(a.video)
    meta = rd.get_meta_data()
    fps = a.fps or meta["fps"]
    frames = [fr for fr in rd]
    rd.close()
    say(f"[영상] {a.video}: {len(frames)} 프레임 {frames[0].shape} fps {meta['fps']}")
    assert len(frames) == T or a.max_frames, f"렌더 영상 프레임 {len(frames)} != 시뮬 프레임 {T}"
    Wv = frames[0].shape[1]
    os.makedirs(f"{a.out}/hist_frames", exist_ok=True)
    wr = imageio.get_writer(a.mp4, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for f in tqdm(range(T), desc="영상"):
        hi = hist_fig(f, Wv, 1100)
        if f in (0, 1, 2, 40, 60, 120):
            imageio.imwrite(f"{a.out}/hist_frames/hist_{f:03d}.png", hi)
        fr = np.concatenate([frames[f][..., :3], hi[:, :Wv]], 0)
        hh = fr.shape[0] // 8 * 8
        wr.append_data(fr[:hh])
    wr.close()
    say(f"[영상] {a.mp4}")

if a.ts:
    fig, axs = plt.subplots(2, 4, figsize=(24, 10), dpi=110)
    tt = np.arange(1, T)
    for j in range(6):
        ax = axs.ravel()[j]
        for t in TR:
            md = np.array(SUM["traj"][t["key"]]["med"])[1:, j]; p9 = np.array(SUM["traj"][t["key"]]["p99"])[1:, j]
            ax.plot(tt, md, color=t["col"], lw=1.6, label=f"{t['lab']} 중앙값")
            ax.plot(tt, p9, color=t["col"], lw=1.2, ls="--", label=f"{t['lab']} 99%")
        ax.set_yscale("log"); ax.set_title(MTITLE[j], fontsize=12); ax.set_xlabel("프레임")
        ax.set_ylabel(("[도]" if MET[j][2] == "도" else "무차원") + " (세로축 로그)"); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    ax = axs[1, 2]
    for t in TR:
        ax.plot(tt, [INFO[t["key"]][f]["n_cells"] for f in tt], color=t["col"], label=t["lab"])
    ax.set_title(f"쓴 셀 수 (입자 ≥ {a.min_pts})"); ax.set_xlabel("프레임"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[1, 3]
    for t in TR:
        ax.plot(tt, [100 * INFO[t["key"]][f]["dir_excl_frac"] for f in tt], color=t["col"], label=t["lab"])
    ax.set_title(f"방향 지표에서 뺀 입자 비율 (s1/s2 < {a.dir_ratio}) [%]"); ax.set_xlabel("프레임"); ax.grid(alpha=0.3)
    fig.suptitle(f"셀별 입자 F 퍼짐 — 프레임별 셀 중앙값(실선)·99%(점선), 0 프레임 제외 (전부 0)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.ts); plt.close(fig)
    say(f"[시계열] {a.ts}")

# outputs.md / tex 용 요약
with open(f"{a.out}/summary_table.txt", "w") as fo:
    for f in (1, 40, 120):
        if f >= T:
            continue
        for t in TR:
            k = t["key"]
            fo.write(f"{f}\t{t['lab']}\t{INFO[k][f]['n_cells']}\t{100 * INFO[k][f]['dir_excl_frac']:.1f}\t"
                     + "\t".join(f"{SUM['traj'][k]['med'][f][j]:.2e}/{SUM['traj'][k]['p99'][f][j]:.2e}" for j in range(6)) + "\n")
say("[끝]")
