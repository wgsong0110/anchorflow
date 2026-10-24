"""ficus: 셀 안 F 주축 방향의 흩어짐 (S² 축 방향 Fréchet/Karcher 평균과 측지 분산) -- 정답 PG 대 선형 출력만 최적화.

입자마다 F = U Σ Vᵀ (σ1 ≥ σ2 ≥ σ3). 최장축 = U[:,0], 최단축 = U[:,2] (현재 형상의 방향 = 왼쪽 특이벡터).
h5 의 F 는 F_sim·R (오른쪽 곱) 이라 왼쪽 특이벡터는 R 에 무관하다 (그래도 두 궤적 R 을 확인해 적는다).
축은 ±v 가 같으므로 RP² 의 측지 거리 d(v, m) = arccos|v·m| ∈ [0, π/2] 를 쓴다.

평균: 셀마다 초기값 = 산란 행렬 Σ v vᵀ 의 최대 고유벡터. 반복마다 (1) 각 v 를 m 쪽 반구로 뒤집고 (v·m < 0 이면 -v)
(2) 로그 사상 log_m(v) = θ u/|u| (u = v - (v·m) m, θ = atan2(|u|, v·m)) 의 셀 평균 t 를 구해 (3) m ← exp_m(t).
|t| < tol 까지. 분산 = 수렴한 m 에서 θ² 의 셀 평균 (rad²).

제외: σ1/σ2 < 1+eps 이면 최장축 무효, σ2/σ3 < 1+eps 이면 최단축 무효. 셀은 그 프레임 유효 입자 ≥ min_cnt 인 것만.
셀 = 시작 위치 기준 Kuhn 사면체 (node_h, 정수 칸 + 소수부 크기 순서 6 개; /home/dkta/work/celldir.py 와 같은 정의), 두 궤적 같은 분할.

  python exe/ficus_axisvar.py --stage stats   # 원자료 npz/json, 렌더 색 npy
  python exe/ficus_axisvar.py --stage plot    # 분위수 띠 png, 히스토그램 영상
  python exe/ficus_axisvar.py --stage compose --left gt.mp4 --right ov.mp4   # 렌더 두 개 나란히 + 색 막대
"""
import argparse
import glob
import json
import os

import numpy as np

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["stats", "plot", "compose"])
ap.add_argument("--gt", default=f"{W}/lbours/ficus_v0/gt")
ap.add_argument("--dump", default=f"{W}/lbours/ficus_v0/roll_ov.pt")
ap.add_argument("--ov_h5", default=f"{W}/lbours/ficus_v0/h5_ov", help="선형 롤아웃 h5 (R 확인·F 대조용)")
ap.add_argument("--out", default=f"{W}/lbours/ficus_axisvar")
ap.add_argument("--node_h", type=float, default=0.056569)
ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--min_cnt", type=int, default=5)
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--max_it", type=int, default=200)
ap.add_argument("--left", default="")
ap.add_argument("--right", default="")
ap.add_argument("--png", default="")
ap.add_argument("--hist_mp4", default="")
ap.add_argument("--render_mp4", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
NAMES = {"gt": "정답 (PG)", "ov": "선형 출력만 최적화"}
AXN = {"long": "최장축", "short": "최단축"}


def kuhn_cells(x0, h):
    q = x0 / h; c = np.floor(q).astype(np.int64); f = q - c; perm = np.argsort(-f, axis=1)
    pid = perm[:, 0] * 9 + perm[:, 1] * 3 + perm[:, 2]
    key = np.concatenate([c, pid[:, None]], 1); _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.ravel()


def karcher_axis(v, cell, nc, valid):
    """셀별 축 방향 Fréchet 평균. 반환: m [nc,3], var [nc] (유효 0 개 셀은 nan), cnt [nc], 반복 수, 최종 max|t|, 미수렴 셀 수."""
    vv = v[valid]; cc = cell[valid]
    cnt = np.bincount(cc, minlength=nc).astype(np.float64)
    S = np.zeros((nc, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            S[:, i, j] = S[:, j, i] = np.bincount(cc, vv[:, i] * vv[:, j], nc)
    _, U = np.linalg.eigh(S); m = U[:, :, -1].copy()
    m[cnt == 0] = (0.0, 0.0, 1.0)
    cs = np.maximum(cnt, 1.0)
    it = 0
    for it in range(1, a.max_it + 1):
        mc = m[cc]; d = np.sum(vv * mc, 1); s = np.where(d < 0, -1.0, 1.0)
        vf = vv * s[:, None]; d = d * s
        u = vf - d[:, None] * mc; nu = np.linalg.norm(u, axis=1); th = np.arctan2(nu, d)
        lg = np.where(nu[:, None] > 1e-300, u * (th / np.maximum(nu, 1e-300))[:, None], 0.0)
        t = np.stack([np.bincount(cc, lg[:, k], nc) for k in range(3)], 1) / cs[:, None]
        nt = np.linalg.norm(t, axis=1)
        dirt = np.where(nt[:, None] > 0, t / np.maximum(nt, 1e-300)[:, None], 0.0)
        m = np.cos(nt)[:, None] * m + np.sin(nt)[:, None] * dirt
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        if nt.max() < a.tol:
            break
    mc = m[cc]; d0 = np.sum(vv * mc, 1); d = np.abs(d0); u = vv * np.where(d0 < 0, -1.0, 1.0)[:, None] - d[:, None] * mc
    th = np.arctan2(np.linalg.norm(u, axis=1), d)
    var = np.bincount(cc, th ** 2, nc) / cs
    var[cnt == 0] = np.nan
    return m, var, cnt, it, float(nt.max()), int(((nt > a.tol) & (cnt > 0)).sum())


def stats():
    import h5py
    import torch
    fs = sorted(glob.glob(f"{a.gt}/sim_*.h5"))
    Xg, Fg = [], []
    for f in fs:
        with h5py.File(f) as h:
            Xg.append(np.array(h["x"])); Fg.append(np.array(h["F"]).reshape(-1, 3, 3))
    Xg, Fg = np.stack(Xg), np.stack(Fg)
    Rg = np.array(json.load(open(f"{a.gt}/meta.json"))["R"])
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    x0 = d["x0"].float().numpy()
    Xo = np.concatenate([x0[None], d["pred"].float().numpy()])
    Fd = d["F"].float().numpy()
    Ro = np.array(json.load(open(f"{a.ov_h5}/meta.json"))["R"])
    # 선형 프레임 0 = 정답 프레임 0 (lb_roll_to_h5 기본 규약: x 와 F_sim·R 을 정답 0 프레임에서 그대로), 그 뒤 = 덤프 F · R (h5 와 같은 좌표계)
    Fo = np.concatenate([Fg[:1], Fd @ Ro.astype(np.float32)])
    del d
    with h5py.File(f"{a.ov_h5}/sim_{len(Fd):010d}.h5") as h:
        fchk = float(np.abs(np.array(h["F"]).reshape(-1, 3, 3) - Fo[-1]).max())
        xchk = float(np.abs(np.array(h["x"]) - Xo[-1]).max())
    info = dict(T_gt=int(len(Xg)), T_ov=int(len(Xo)), N=int(Xg.shape[1]), R_gt=Rg.tolist(), R_ov=Ro.tolist(),
                x0_vs_gt0_maxabs=float(np.abs(Xo[0] - Xg[0]).max()), ovh5_last_F_maxabs=fchk, ovh5_last_x_maxabs=xchk,
                node_h=a.node_h, eps=a.eps, min_cnt=a.min_cnt, tol=a.tol, max_it=a.max_it)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    assert len(Xg) == len(Xo)
    cell = kuhn_cells(Xg[0].astype(np.float64), a.node_h); nc = int(cell.max() + 1)
    tot = np.bincount(cell, minlength=nc)
    info.update(n_cells=nc, cell_cnt_mean=float(tot.mean()), cell_cnt_min=int(tot.min()), cell_cnt_max=int(tot.max()),
                n_cells_ge_min=int((tot >= a.min_cnt).sum()))
    T, N = len(Xg), Xg.shape[1]
    out = {}
    for tr, Fs in (("gt", Fg), ("ov", Fo)):
        for ax in ("long", "short"):
            out[f"{tr}_{ax}_var"] = np.full((T, nc), np.nan)
            out[f"{tr}_{ax}_mean"] = np.full((T, nc, 3), np.nan)
            out[f"{tr}_{ax}_cnt"] = np.zeros((T, nc), np.int32)
            out[f"{tr}_{ax}_excl"] = np.zeros(T)
            out[f"{tr}_{ax}_ncell"] = np.zeros(T, np.int32)
            out[f"{tr}_{ax}_iters"] = np.zeros(T, np.int32)
            out[f"{tr}_{ax}_nonconv"] = np.zeros(T, np.int32)
        out[f"{tr}_sv"] = np.zeros((T, 3, 3))  # 프레임별 σ1..3 의 5·50·95% 분위
        for t in range(T):
            Uu, Ss, _ = np.linalg.svd(Fs[t].astype(np.float64))
            out[f"{tr}_sv"][t] = np.percentile(Ss, [5, 50, 95], axis=0)
            for ax, col, ok in (("long", 0, Ss[:, 0] >= (1 + a.eps) * Ss[:, 1]),
                                ("short", 2, Ss[:, 1] >= (1 + a.eps) * Ss[:, 2])):
                v = Uu[:, :, col]
                k = f"{tr}_{ax}"
                out[f"{k}_excl"][t] = 1.0 - ok.mean()
                if ok.sum() == 0:
                    continue
                m, var, cnt, it, res, ncv = karcher_axis(v, cell, nc, ok)
                good = cnt >= a.min_cnt
                out[f"{k}_var"][t, good] = var[good]; out[f"{k}_mean"][t, good] = m[good]
                out[f"{k}_cnt"][t] = cnt; out[f"{k}_ncell"][t] = good.sum()
                out[f"{k}_iters"][t] = it; out[f"{k}_nonconv"][t] = ncv
            if t % 10 == 0 or t == T - 1:
                print(f"[{tr}] 프레임 {t}  제외 최장 {out[f'{tr}_long_excl'][t]:.4f} 최단 {out[f'{tr}_short_excl'][t]:.4f}  "
                      f"셀 {out[f'{tr}_long_ncell'][t]}/{out[f'{tr}_short_ncell'][t]}  반복 {out[f'{tr}_long_iters'][t]}/{out[f'{tr}_short_iters'][t]}  "
                      f"중앙 분산 {np.nanmedian(out[f'{tr}_long_var'][t]) if out[f'{tr}_long_ncell'][t] else float('nan'):.4g}/"
                      f"{np.nanmedian(out[f'{tr}_short_var'][t]) if out[f'{tr}_short_ncell'][t] else float('nan'):.4g}", flush=True)
    np.savez_compressed(f"{a.out}/axisvar.npz", cell=cell, cell_tot=tot, **out)
    # 렌더 색: 최장축 셀 분산, 두 궤적 공통 고정 범위 [0, vmax], vmax = 두 궤적 전 프레임 유효 셀 분산의 99%
    pool = np.concatenate([out["gt_long_var"][np.isfinite(out["gt_long_var"])], out["ov_long_var"][np.isfinite(out["ov_long_var"])]])
    vmax = float(np.percentile(pool, 99))
    info["render_vmax_rad2"] = vmax
    import matplotlib
    cmap = matplotlib.colormaps["turbo"]
    for tr in ("gt", "ov"):
        V = out[f"{tr}_long_var"]
        C = np.zeros((T, N, 3), np.float16)
        for t in range(T):
            pv = V[t][cell]; fin = np.isfinite(pv)
            c = np.full((N, 3), 0.6)
            c[fin] = cmap(np.clip(pv[fin] / vmax, 0, 1))[:, :3]
            C[t] = c.astype(np.float16)
        np.save(f"{a.out}/col_{tr}_long.npy", C)
    json.dump(info, open(f"{a.out}/info.json", "w"), ensure_ascii=False, indent=1)
    print("AXISVAR_STATS_DONE", flush=True)


def _font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    c = [os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")] + [p for p in fm.findSystemFonts() if "NotoSansCJK" in os.path.basename(p) or "NotoSansKR" in os.path.basename(p)]
    for p in c:
        if os.path.exists(p):
            fm.fontManager.addfont(p); plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False


COLS = {"gt": "#2a78d6", "ov": "#eb6834"}


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    _font()
    Z = np.load(f"{a.out}/axisvar.npz"); info = json.load(open(f"{a.out}/info.json"))
    T = info["T_gt"]; tt = np.arange(T)
    summ = {}
    fin_all = np.concatenate([Z[f"{tr}_{ax}_var"][np.isfinite(Z[f"{tr}_{ax}_var"])] for tr in ("gt", "ov") for ax in ("long", "short")])
    lo = max(float(np.percentile(fin_all, 0.5)), 1e-12); hi = float(fin_all.max())
    lo = 10 ** np.floor(np.log10(lo)); hi = 10 ** np.ceil(np.log10(hi))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for j, ax in enumerate(("long", "short")):
        A = axs[j]
        for tr in ("gt", "ov"):
            V = Z[f"{tr}_{ax}_var"]; q = np.full((T, 5), np.nan); mu = np.full(T, np.nan)
            for t in range(T):
                f = V[t][np.isfinite(V[t])]
                if len(f):
                    q[t] = np.percentile(f, [5, 25, 50, 75, 95]); mu[t] = f.mean()
            summ[f"{tr}_{ax}"] = dict(q5=q[:, 0].tolist(), q25=q[:, 1].tolist(), q50=q[:, 2].tolist(), q75=q[:, 3].tolist(),
                                      q95=q[:, 4].tolist(), mean=mu.tolist(), ncell=Z[f"{tr}_{ax}_ncell"].tolist(),
                                      excl=Z[f"{tr}_{ax}_excl"].tolist())
            c = COLS[tr]
            A.fill_between(tt, q[:, 0], q[:, 4], color=c, alpha=0.13, lw=0)
            A.fill_between(tt, q[:, 1], q[:, 3], color=c, alpha=0.30, lw=0)
            A.plot(tt, q[:, 2], color=c, lw=2, label=f"{NAMES[tr]} 중앙값")
            A.plot(tt, mu, color=c, lw=1.2, ls="--", label=f"{NAMES[tr]} 평균")
        A.set_yscale("log"); A.set_ylim(lo, hi); A.set_xlim(0, T - 1)
        A.set_title(f"{AXN[ax]} 방향: 셀 안 측지 분산 (띠: 25–75%, 5–95%)")
        A.set_xlabel("프레임"); A.grid(alpha=0.25, lw=0.6)
        if j == 0:
            A.set_ylabel("셀 분산 (rad²)")
        A.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"ficus: 셀(Kuhn 사면체, 입자 ≥ {info['min_cnt']}) 안 F 주축 방향의 흩어짐 -- 정답 대 선형 출력만 최적화", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.png, dpi=130); plt.close(fig)
    json.dump(summ, open(f"{a.out}/quantiles.json", "w"))
    # 히스토그램 영상: 로그 구간, 두 궤적·전 프레임 같은 구간·같은 세로축
    bins = np.logspace(np.log10(lo), np.log10(hi), 61)
    ymax = 0.0
    for tr in ("gt", "ov"):
        for ax in ("long", "short"):
            V = Z[f"{tr}_{ax}_var"]
            for t in range(T):
                f = V[t][np.isfinite(V[t])]
                if len(f):
                    ymax = max(ymax, np.histogram(f, bins)[0].max() / len(f))
    wr = imageio.get_writer(a.hist_mp4, fps=10, codec="libx264", quality=8, macro_block_size=1)
    for t in range(T):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for j, ax in enumerate(("long", "short")):
            A = axs[j]
            for tr in ("gt", "ov"):
                f = Z[f"{tr}_{ax}_var"][t]; f = f[np.isfinite(f)]
                if len(f):
                    A.hist(f, bins, weights=np.full(len(f), 1.0 / len(f)), histtype="stepfilled", alpha=0.35, color=COLS[tr])
                    A.hist(f, bins, weights=np.full(len(f), 1.0 / len(f)), histtype="step", lw=1.8, color=COLS[tr],
                           label=f"{NAMES[tr]} (셀 {len(f)}, 중앙 {np.median(f):.2e})")
                else:
                    A.plot([], [], color=COLS[tr], label=f"{NAMES[tr]} (유효 셀 없음)")
            A.set_xscale("log"); A.set_xlim(lo, hi); A.set_ylim(0, ymax * 1.08)
            A.set_title(f"{AXN[ax]} 방향 셀 분산"); A.set_xlabel("셀 분산 (rad²)"); A.grid(alpha=0.25, lw=0.6)
            if j == 0:
                A.set_ylabel("셀 비율")
            A.legend(fontsize=8.5, loc="upper left")
        fig.suptitle(f"프레임 {t}", fontsize=16, color="#c00000")
        fig.tight_layout()
        fig.canvas.draw(); im = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        wr.append_data(im); plt.close(fig)
    wr.close()
    print("AXISVAR_PLOT_DONE", flush=True)


def compose():
    import imageio.v2 as imageio
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager as fm
    info = json.load(open(f"{a.out}/info.json")); vmax = info["render_vmax_rad2"]
    cand = [os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")] + [f for f in fm.findSystemFonts() if "Noto" in f and ("CJK" in f or "KR" in f)]
    cand = [p for p in cand if os.path.exists(p)]
    rl, rr = imageio.get_reader(a.left), imageio.get_reader(a.right)
    wr = imageio.get_writer(a.render_mp4, fps=30, codec="libx264", quality=8, macro_block_size=1)
    cmap = matplotlib.colormaps["turbo"]
    fnt = None
    for t, (fl, fr) in enumerate(zip(rl, rr)):
        H, Wd = fl.shape[:2]
        if fnt is None:
            fnt = (ImageFont.truetype(cand[0], int(H * 0.045)), ImageFont.truetype(cand[0], int(H * 0.03)), ImageFont.truetype(cand[0], int(H * 0.07)))
            bh = int(H * 0.16)
            bar = np.ones((bh, 2 * Wd, 3), np.uint8) * 255
            x0, x1 = int(2 * Wd * 0.2), int(2 * Wd * 0.8)
            g = cmap(np.linspace(0, 1, x1 - x0))[:, :3]
            bar[int(bh * 0.15):int(bh * 0.45), x0:x1] = (g[None] * 255).astype(np.uint8)
            bim = Image.fromarray(bar); bd = ImageDraw.Draw(bim)
            for k in range(5):
                xx = x0 + int((x1 - x0) * k / 4)
                bd.text((xx - 20, int(bh * 0.5)), f"{vmax * k / 4:.3g}", fill=(0, 0, 0), font=fnt[1])
            bd.text((x1 + 15, int(bh * 0.12)), "rad²", fill=(0, 0, 0), font=fnt[1])
            bd.text((10, int(bh * 0.12)), "최장축 셀 분산", fill=(0, 0, 0), font=fnt[1])
            bd.text((10, int(bh * 0.55)), "회색 = 유효 셀 아님", fill=(90, 90, 90), font=fnt[1])
            bar = np.asarray(bim)
        im = Image.fromarray(np.concatenate([fl, fr], 1)); dr = ImageDraw.Draw(im)
        dr.text((int(Wd * 0.03), int(H * 0.03)), NAMES["gt"], fill=(0, 0, 0), font=fnt[0])
        dr.text((Wd + int(Wd * 0.03), int(H * 0.03)), NAMES["ov"], fill=(0, 0, 0), font=fnt[0])
        dr.text((int(Wd * 0.85), int(H * 0.02)), f"{t}", fill=(230, 0, 0), font=fnt[2])
        wr.append_data(np.concatenate([np.asarray(im), bar], 0))
    wr.close()
    print("AXISVAR_COMPOSE_DONE", flush=True)


{"stats": stats, "plot": plot, "compose": compose}[a.stage]()
