"""절 CM106.8: 레고 TC 뒤집힌 사면체 영상의 두 칸 그림 (클러스터 카드에서; 글자는 로컬에서 얹음).
왼쪽 = 일반 3DGS 렌더 (exe/lb_render.py 머리 부분의 카메라·가우시안을 그대로 씀), 오른쪽 = 같은 카메라의 입자 점 그림 + 뒤집힌 사면체 무게중심 (빨간 큰 점).
  python exe/tetc_inv_render.py --sim H5DIR --config cfg.json --shape lego --inv DIR(inv_*.npz) --out PNGDIR
파일 i (sim_i.h5, i>=1) 의 뒤집힘 = inv_{i-1}.npz (기록 i-1 = i 번째 걸음 끝). PNG: L_%04d.png · R_%04d.png.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LB = os.path.join(HERE, "lb_render.py")
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True); ap.add_argument("--config", required=True); ap.add_argument("--shape", default="lego")
ap.add_argument("--inv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pos", default="cur", choices=["cur", "ref", "pmean"])
ap.add_argument("--dot", type=int, default=7)
b = ap.parse_args()
os.makedirs(b.out, exist_ok=True)
sys.argv = ["lb_render.py", "--sim", b.sim, "--config", b.config, "--shape", b.shape, "--out", "/dev/null"]
G = {"__name__": "lbr_head", "__file__": LB}
exec(compile(open(LB).read().split("_FONT = None")[0], LB, "exec"), G)                       # 카메라·가우시안 준비까지만
torch, np, h5py, imageio, glob = G["torch"], G["np"], G["h5py"], G["imageio"], G["glob"]
dev, RAST, SHS, OPA, C0, ROT, RT, SO, MEAN, wv, pj, cam, NG = (G[k] for k in ("dev", "RAST", "SHS", "OPA", "C0", "ROT", "_RT", "SO", "MEAN", "wv", "pj", "cam", "NG"))
FPROJ = (wv[None] @ pj[None])[0]
Wd, Hd = int(cam["width"]), int(cam["height"])


def world(x):
    P = (x - 1.0) / SO + MEAN
    return P @ RT if ROT else P


def proj(P):
    ph = torch.cat([P, torch.ones_like(P[:, :1])], 1) @ FPROJ
    nd = ph[:, :3] / (ph[:, 3:4] + 1e-7)
    return torch.stack([((nd[:, 0] + 1.0) * Wd - 1.0) * 0.5, ((nd[:, 1] + 1.0) * Hd - 1.0) * 0.5], 1), ph[:, 3]


def disk(img, px, r, col):
    for cx, cy in px:
        x0, x1, y0, y1 = int(cx) - r, int(cx) + r + 1, int(cy) - r, int(cy) + r + 1
        if x1 <= 0 or y1 <= 0 or x0 >= Wd or y0 >= Hd:
            continue
        yy, xx = np.mgrid[max(y0, 0):min(y1, Hd), max(x0, 0):min(x1, Wd)]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[max(y0, 0):min(y1, Hd), max(x0, 0):min(x1, Wd)][m] = col


files = sorted(glob.glob(f"{b.sim}/sim_*.h5"))
for fi, p in enumerate(files):
    with h5py.File(p) as h:
        xa = np.array(h["x"]); xa = xa.T if xa.shape[0] == 3 else xa
        Fa = np.array(h["F"]).reshape(-1, 9)[:NG]
    x = torch.as_tensor(xa, device=dev).float()
    if not bool(torch.isfinite(x).all()) or not np.isfinite(Fa).all():
        print(f"[멈춤] 파일 {fi} 비유한", flush=True); break
    F = torch.as_tensor(Fa, device=dev).float().reshape(-1, 3, 3)
    cov = F @ C0 @ F.transpose(1, 2)
    if ROT:
        cov = RT.T @ cov @ RT
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    Pm = world(x[:NG])
    img = G["RAST"](means3D=Pm, means2D=torch.zeros_like(Pm), shs=SHS, colors_precomp=None, opacities=OPA, scales=None, rotations=None, cov3D_precomp=cc)[0]
    L = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(b.out, f"L_{fi:04d}.png"), L)
    R = np.full((Hd, Wd, 3), 255, np.uint8)
    px, _ = proj(world(x))
    pn = px.cpu().numpy(); ok = (pn[:, 0] >= 0) & (pn[:, 0] < Wd - 1) & (pn[:, 1] >= 0) & (pn[:, 1] < Hd - 1)
    ix, iy = pn[ok, 0].astype(int), pn[ok, 1].astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            R[iy + dy, ix + dx] = (60, 80, 110)
    nv = 0
    if fi >= 1 and os.path.exists(os.path.join(b.inv, f"inv_{fi - 1:03d}.npz")):
        z = np.load(os.path.join(b.inv, f"inv_{fi - 1:03d}.npz"))
        c = z[b.pos]
        c = c[np.isfinite(c).all(1)]
        nv = c.shape[0]
        if nv:
            pc, _ = proj(world(torch.as_tensor(c, device=dev).float()))
            disk(R, pc.cpu().numpy(), b.dot, (230, 0, 0))
    imageio.imwrite(os.path.join(b.out, f"R_{fi:04d}.png"), R)
    print(f"[프레임] {fi} 뒤집힌 {nv}", flush=True)
print(f"[끝] {len(files)} 파일", flush=True)
