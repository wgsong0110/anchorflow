"""렌더 프레임 두 벌로 시각 품질 셋을 잰다: FID / FVD / KVD.

규약 (작은 표본이라 **어떻게 쟀는지**를 같이 기록한다):
  * FID  : pytorch-fid 의 InceptionV3 pool3 (2048 차원). 프레임을 하나의 표본으로
           본다. 표본이 수십 장이라 절대값은 편향돼 있다 -- **같은 프레임 수로
           잰 값끼리만** 비교할 것.
  * FVD  : StyleGAN-V 가 쓰는 I3D torchscript 특징(400 차원 logits 전 단계).
           16 프레임 창을 stride 1 로 밀어 클립을 만든다 (40 프레임 -> 25 클립).
  * KVD  : 같은 I3D 특징에 다항 커널 MMD² (KID 의 영상판, 블록 100 개 평균 대신
           표본이 작아 전체 한 블록으로 쓴다).

  python exe/vq_metrics.py --ref <참조 png 디렉토리> --test <대상 png 디렉토리> \
      --out out.json [--n 40] [--win 16]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn

W = "/home/dkta/work"
I3D = os.environ.get("AF_I3D", f"{W}/metricw/i3d_torchscript.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--n", type=int, default=0, help="앞 n 프레임만 (0=전부)")
ap.add_argument("--win", type=int, default=16, help="FVD/KVD 클립 길이")
ap.add_argument("--label", default="")
ap.add_argument("--dev", default="cuda:0")
a = ap.parse_args()
dev = torch.device(a.dev if torch.cuda.is_available() else "cpu")


def load_dir(d, n=0):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    if not fs:
        raise SystemExit(f"png 이 없다: {d}")
    if n:
        fs = fs[:n]
    import imageio.v2 as iio
    im = np.stack([iio.imread(f)[..., :3] for f in fs])      # [T,H,W,3] uint8
    return torch.as_tensor(im).permute(0, 3, 1, 2).float() / 255.0


def inception_feats(x):
    """[T,3,H,W] (0~1) -> [T,2048]"""
    from pytorch_fid.inception import InceptionV3
    m = InceptionV3([3]).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            b = x[i:i + 16].to(dev)
            b = Fn.interpolate(b, size=(299, 299), mode="bilinear",
                               align_corners=False)
            out.append(m(b)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out).double()


def i3d_feats(x, win):
    """[T,3,H,W] (0~1) -> [클립수, C]  (16 프레임 창, stride 1)"""
    m = torch.jit.load(I3D).to(dev).eval()
    T = x.shape[0]
    if T < win:
        raise SystemExit(f"프레임이 {T} 장이라 창 {win} 을 못 만든다")
    v = Fn.interpolate(x, size=(224, 224), mode="bilinear",
                       align_corners=False)
    v = v * 2.0 - 1.0                                   # I3D 는 [-1,1]
    out = []
    with torch.no_grad():
        for s in range(0, T - win + 1):
            clip = v[s:s + win].permute(1, 0, 2, 3).unsqueeze(0).to(dev)
            f = m(clip, rescale=False, resize=False, return_features=True)
            out.append(f.reshape(1, -1).cpu())
    return torch.cat(out).double()


def frechet(f1, f2):
    from scipy import linalg
    m1, m2 = f1.mean(0).numpy(), f2.mean(0).numpy()
    s1 = np.cov(f1.numpy(), rowvar=False)
    s2 = np.cov(f2.numpy(), rowvar=False)
    diff = m1 - m2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2)
                 - 2 * np.trace(covmean))


def poly_mmd2(f1, f2, degree=3, gamma=None, coef0=1.0):
    """KID/KVD 의 다항 커널 MMD² (불편추정)."""
    x, y = f1.numpy(), f2.numpy()
    d = x.shape[1]
    gamma = 1.0 / d if gamma is None else gamma

    def k(u, v):
        return (gamma * u.dot(v.T) + coef0) ** degree
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = x.shape[0], y.shape[0]
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sxy = kxy.sum() / (m * n)
    return float(sxx + syy - 2 * sxy)


R = load_dir(a.ref, a.n)
T = load_dir(a.test, a.n)
n = min(R.shape[0], T.shape[0])
R, T = R[:n], T[:n]
print(f"[입력] 참조 {a.ref} / 대상 {a.test}  프레임 {n}  크기 "
      f"{tuple(R.shape[-2:])}", flush=True)

fid = frechet(inception_feats(R), inception_feats(T))
fr, ft = i3d_feats(R, a.win), i3d_feats(T, a.win)
fvd = frechet(fr, ft)
kvd = poly_mmd2(fr, ft)
rmse = float(((R - T) ** 2).mean().sqrt())
psnr = float(-10.0 * np.log10(max(rmse ** 2, 1e-12)))
print(f"[결과] FID {fid:.3f}  FVD {fvd:.3f}  KVD {kvd:.5f}  "
      f"(클립 {fr.shape[0]} 개, 특징 {fr.shape[1]} 차원)  픽셀 PSNR {psnr:.2f} dB",
      flush=True)

if a.out:
    json.dump(dict(ref=a.ref, test=a.test, label=a.label, frames=n,
                   win=a.win, n_clips=int(fr.shape[0]),
                   fid=fid, fvd=fvd, kvd=kvd, psnr=psnr),
              open(a.out, "w"), indent=1)
    print(f"[저장] {a.out}", flush=True)
