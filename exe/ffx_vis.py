#!/usr/bin/env python3
"""[final_fix_phys] 시각 품질 지표 (파레토 방법 x 장면 한 칸): 같은 렌더러 (exe/lb_render.py, F 반영 3DGS, 정답 렌더와 같은 카메라) PNG -> json.

정답 없이:
  NIQE, BRISQUE (낮을수록 좋음), MUSIQ (KonIQ), CLIP-IQA (높을수록 좋음) -- pyiqa 표준 구현 (create_metric 기본 가중치), 프레임 1..T 평균
  시간 깜빡임 warping error (Lai et al. ECCV 2018 정의): 연속 프레임 I_t, I_{t+1} 사이 광류를 RAFT (torchvision raft_large,
    Raft_Large_Weights.C_T_SKHT_V2) 로 양방향 추정, I_t 를 뒤 광류 F_{t+1->t} 로 끌어와 (bilinear) 차이를 잼.
    가림 마스크 M (Sundaram et al. 2010 앞뒤 일관성): |F_b(x) + F_f(x + F_b(x))|^2 < 0.01 (|F_b|^2 + |F_f(x + F_b)|^2) + 0.5 (픽셀) 이고 영상 안에 들어오는 픽셀만 1.
    E_t = sum_x M ||I_{t+1}(x) - W(I_t)(x)||^2 / sum_x M (RGB 0~1, 채널 합), warp_err = t 평균 (프레임 0->1 .. T-1->T).
정답 렌더 대비 (참고): PSNR, SSIM (skimage, 채널별), LPIPS (alex) -- exe/stab_vis.py 와 같은 정의, 프레임 1..T 평균.

  python exe/ffx_vis.py --png RUN/png [--ref GT/png] --out vis.json
"""
import argparse
import glob
import json

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--png", required=True)
ap.add_argument("--ref", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--iqa", default="niqe,brisque,musiq,clipiqa")
a = ap.parse_args()
dev = "cuda"
FR = sorted(glob.glob(f"{a.png}/*.png"))
RF = sorted(glob.glob(f"{a.ref}/*.png")) if a.ref else []


def img(p):
    return imageio.imread(p)[..., :3].astype(np.float32) / 255


def tens(x):
    return torch.as_tensor(x).permute(2, 0, 1)[None].to(dev)


res = dict(frames=len(FR) - 1)
# ---- 무참조 IQA ----
import pyiqa  # noqa: E402
res["pyiqa_version"] = getattr(pyiqa, "__version__", "?")
for name in [q for q in a.iqa.split(",") if q]:
    try:
        M = pyiqa.create_metric(name, device=dev)
        vs = []
        with torch.no_grad():
            for p in FR[1:]:
                vs.append(float(M(tens(img(p)).clamp(0, 1))))
        res[name] = float(np.mean(vs)); res[name + "_lower_better"] = bool(M.lower_better)
        del M; torch.cuda.empty_cache()
    except Exception as ex:                                                           # noqa: BLE001
        res[name] = None; res[name + "_err"] = repr(ex)[:300]
    print(name, res.get(name), flush=True)

# ---- warping error (RAFT) ----
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights  # noqa: E402
W8 = Raft_Large_Weights.C_T_SKHT_V2
RAFT = raft_large(weights=W8, progress=False).to(dev).eval()
TF = W8.transforms()


def pad8(x):
    H, W = x.shape[-2:]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    return Fn.pad(x, (0, pw, 0, ph), mode="replicate"), H, W


def flow(i1, i2):
    a_, b_ = TF(i1, i2)
    a_, H, W = pad8(a_); b_, _, _ = pad8(b_)
    with torch.no_grad():
        f = RAFT(a_, b_)[-1]
    return f[..., :H, :W]


def warp(x, f):
    """x [1,C,H,W] 를 f [1,2,H,W] (x 좌표 = 픽셀 + f) 에서 끌어옴. -> (끌어온 값, 영상 안 여부)"""
    _, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    gx = xx[None] + f[:, 0]; gy = yy[None] + f[:, 1]
    inside = (gx >= 0) & (gx <= W - 1) & (gy >= 0) & (gy <= H - 1)
    grid = torch.stack([2 * gx / (W - 1) - 1, 2 * gy / (H - 1) - 1], -1)
    return Fn.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True), inside


ws, occ = [], []
prev = tens(img(FR[0]))
for p in FR[1:]:
    cur = tens(img(p))
    fb = flow(cur, prev)                                                              # t+1 -> t
    ff = flow(prev, cur)                                                              # t -> t+1
    wprev, ins = warp(prev, fb)
    ffw, _ = warp(ff, fb)
    s = fb + ffw
    Mk = ((s ** 2).sum(1) < 0.01 * ((fb ** 2).sum(1) + (ffw ** 2).sum(1)) + 0.5) & ins
    e = ((cur - wprev) ** 2).sum(1)
    n = Mk.sum()
    ws.append(float((e * Mk).sum() / n.clamp_min(1))); occ.append(float(1 - Mk.float().mean()))
    prev = cur
res["warp_err"] = float(np.mean(ws)); res["warp_occ_frac"] = float(np.mean(occ))
res["raft_weights"] = "torchvision raft_large C_T_SKHT_V2"
print("warp_err", res["warp_err"], flush=True)
del RAFT; torch.cuda.empty_cache()

# ---- 정답 렌더 대비 ----
if RF:
    import lpips
    from skimage.metrics import structural_similarity as ssim
    LP = lpips.LPIPS(net="alex").to(dev).eval()
    T = min(len(FR), len(RF))
    ps, sv, ls = [], [], []
    for t in range(1, T):
        x, y = img(FR[t]), img(RF[t])
        ps.append(10 * np.log10(1.0 / max(float(((x - y) ** 2).mean()), 1e-12)))
        sv.append(float(ssim(x, y, channel_axis=2, data_range=1.0)))
        with torch.no_grad():
            ls.append(float(LP(tens(x) * 2 - 1, tens(y) * 2 - 1)))
    res.update(psnr=float(np.mean(ps)), ssim=float(np.mean(sv)), lpips=float(np.mean(ls)), ref_frames=T - 1)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True)
