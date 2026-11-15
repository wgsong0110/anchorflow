"""절 CM182 GF 공개 장면 후처리: (1) 위상 변화 (같은 표본·같은 잣대로 정답 / 우리) (2) 렌더 PNG 쌍 -> PSNR·SSIM·LPIPS + 정답 | 우리 영상.
  python gfs_post.py topo --gt_pt GT.pt --pred pred.pt --out out.json
  python gfs_post.py vid --gt_png DIR --our_png DIR --out_json J --out_mp4 V --font F [--fps 24]
위상 지표 (measure_gf_tearing 과 같은 정의): 초기 k-NN (k=16, 표본 4000) 이웃이 초기 거리의 3 배 넘게 벌어진 비율 (프레임별, 최댓값·마지막),
조각 수 = 초기 k-NN 간선 중 현재 거리 <= 3 x 초기 거리 인 것만 남긴 그래프의 연결 성분 수 (표본 입자 기준; 입자 1% 미만 성분은 따로 세지 않고 큰 조각 수만 = 표본의 1% 이상 성분)."""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("mode")
ap.add_argument("--gt_pt"); ap.add_argument("--pred"); ap.add_argument("--out", default="")
ap.add_argument("--gt_png"); ap.add_argument("--our_png"); ap.add_argument("--out_json"); ap.add_argument("--out_mp4"); ap.add_argument("--font", default=""); ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if a.mode == "topo":
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    gt = torch.load(a.gt_pt, map_location="cpu", weights_only=False)["x"].float().numpy()
    pr = torch.load(a.pred, map_location="cpu", weights_only=False)["pred"].float().numpy()
    T = min(gt.shape[0], pr.shape[0]); fin = [bool(np.isfinite(pr[t]).all()) for t in range(T)]
    nfin = 0
    while nfin < T and fin[nfin]: nfin += 1
    rng = np.random.default_rng(a.seed); N = gt.shape[1]
    ok0 = np.isfinite(gt[0]).all(1); idx = np.sort(rng.choice(np.where(ok0)[0], size=min(4000, int(ok0.sum())), replace=False))
    x0 = gt[0][idx]; tr = cKDTree(x0); d0, nb = tr.query(x0, k=17); d0, nb = d0[:, 1:], nb[:, 1:]
    def stat(X, t):
        x = X[t][idx]; xn = X[t][idx[nb]]                      # 표본 이웃도 같은 입자 (표본 안 이웃)
        okp = np.isfinite(x).all(1)
        d = np.linalg.norm(xn - x[:, None], axis=-1)
        esc = (d > 3 * d0) | ~np.isfinite(d)
        r = float(esc[okp].mean()) if okp.any() else float("nan")
        keep = (~esc) & okp[:, None]
        n = x.shape[0]; rows = np.repeat(np.arange(n), 16)
        g = coo_matrix((np.ones(keep.sum()), (rows[keep.ravel()], nb.ravel()[keep.ravel()])), shape=(n, n))
        nc, lab = connected_components(g, directed=False)
        sz = np.bincount(lab); big = int((sz >= max(1, 0.01 * n)).sum())
        return r, big, int(nc)
    def series(X, TT):
        s = [stat(X, t) for t in range(TT)]; return dict(esc=[q[0] for q in s], pieces=[q[1] for q in s], comps=[q[2] for q in s])
    sg = series(gt, T); so = series(pr, nfin) if nfin else dict(esc=[], pieces=[], comps=[])
    o = dict(frames_gt=int(gt.shape[0]), frames_pred=int(pr.shape[0]), finite_pred_frames=nfin, sample=int(len(idx)),
             gt=dict(esc_max=max(sg["esc"]), esc_last=sg["esc"][-1], pieces_last=sg["pieces"][-1], pieces_max=max(sg["pieces"]), series=sg),
             ours=dict(esc_max=max(so["esc"]) if nfin else None, esc_last=so["esc"][-1] if nfin else None, pieces_last=so["pieces"][-1] if nfin else None,
                       pieces_max=max(so["pieces"]) if nfin else None, series=so))
    json.dump(o, open(a.out, "w")); print("[위상]", {k: o[k] for k in ("finite_pred_frames",)}, {k: v for k, v in o["gt"].items() if k != "series"}, {k: v for k, v in o["ours"].items() if k != "series"}, flush=True)
else:
    import glob, imageio, lpips, torch
    from PIL import Image, ImageDraw, ImageFont
    from skimage.metrics import structural_similarity as ssim
    gf = sorted(glob.glob(a.gt_png + "/*.png")); of = sorted(glob.glob(a.our_png + "/*.png")); n = min(len(gf), len(of))
    lp = lpips.LPIPS(net="alex").cuda().eval(); P, S, L = [], [], []
    fnt = ImageFont.truetype(a.font, 28) if a.font and os.path.exists(a.font) else ImageFont.load_default()
    W = imageio.get_writer(a.out_mp4, fps=a.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2, ffmpeg_params=["-movflags", "+faststart"])
    for i in range(n):
        g = np.array(Image.open(gf[i]).convert("RGB")); o = np.array(Image.open(of[i]).convert("RGB"))
        mse = ((g.astype(np.float64) - o) ** 2).mean(); P.append(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10))))
        S.append(float(ssim(g, o, channel_axis=2, data_range=255)))
        tg = torch.from_numpy(g).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1; to = torch.from_numpy(o).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        with torch.no_grad(): L.append(float(lp(tg, to)))
        im = Image.fromarray(np.concatenate([g, o], 1)); d = ImageDraw.Draw(im)
        d.text((12, 8), "정답 (GaussianFluent)", fill=(0, 0, 0), font=fnt); d.text((g.shape[1] + 12, 8), "우리", fill=(0, 0, 0), font=fnt)
        W.append_data(np.array(im))
    W.close()
    o = dict(frames=n, psnr=float(np.mean(P)), ssim=float(np.mean(S)), lpips=float(np.mean(L)), psnr_by_frame=P, ssim_by_frame=S, lpips_by_frame=L)
    json.dump(o, open(a.out_json, "w")); print(f"[시각] {n} 프레임 PSNR {o['psnr']:.2f} SSIM {o['ssim']:.4f} LPIPS {o['lpips']:.4f}", flush=True)
