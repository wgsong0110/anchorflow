"""같은 씬·같은 제어점 조작에서 여러 방법을 한 자리에 놓고 잰다.

정답은 **PhysGaussian** 궤적이다. 견주는 것들:

  정지        첫 프레임 그대로 (아무것도 안 한 기준선)
  등속        t0 의 속도로 계속 간다
  제어점 끌기  물리 없이 제어점 변위를 가우시안 감쇠로 퍼뜨린다 (드래그 도구)
  다른 솔버    같은 입자·같은 제어점을 GaussianFluent 의 warp 솔버로 (선택)
  학생        앵커 스테퍼 (우리 것)

지표는 두 갈래다.
  입자 -- 같은 입자끼리의 RMSE, 그리고 대응을 가정하지 않는 CD / EMD.
          셋 다 **물체 크기**로 나눈다 (자기 변위로 나누면 덜 움직인 쪽이 유리하다).
          EMD 는 O(n^3) 이라 부분표본으로 재고, 두 구름에서 **같은 색인**을 쓴다.
  영상 -- 3DGS 로 렌더해서 PSNR / SSIM (같은 카메라, 같은 프레임).

  PYTHONPATH=<PG>:<GS>:<lib> python exe/eval_scene_baselines.py \
      --model_path <3DGS> --config <씬 config> --gt_h5 <정답 h5> \
      --data <롤아웃 dir> --traj eval_000 --ckpt <학생.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--gt_h5", required=True, help="정답(PG) h5 디렉토리")
ap.add_argument("--data", required=True)
ap.add_argument("--traj", default="eval_000")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--gf_h5", default=None, help="다른 솔버 h5 디렉토리 (선택)")
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--cd_sample", type=int, default=20000)
ap.add_argument("--emd_sample", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_video", action="store_true")
a = ap.parse_args()

dev = "cuda:0"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                      # noqa: E402
import taichi as ti                                               # noqa: E402
from PIL import Image, ImageDraw                                  # noqa: E402

ti.init(arch=ti.cuda, device_memory_GB=4.0)
from anchorflow.gsrender import GSScene                           # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, bc_features,  # noqa: E402
                               grid_knn, skin, skin_with_jacobian)


# ------------------------------------------------------------------ 정답 궤적
def frames_of(d):
    return sorted(glob.glob(os.path.join(d, "sim_*.h5")))


def rd(p, key="x"):
    with h5py.File(p, "r") as h:
        if key not in h:
            return None
        v = np.array(h[key])
    return (v.T if v.shape[0] in (3, 9) else v).astype(np.float32)


FS = frames_of(a.gt_h5)
if not FS:
    raise SystemExit(f"정답 h5 가 없다: {a.gt_h5}")
NT = len(FS)
X0 = torch.from_numpy(rd(FS[0])).to(dev)
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_ALL = X0.shape[0]
cz = np.load(os.path.join(a.gt_h5, "control.npz"))
CP = torch.from_numpy(cz["pos"]).to(dev)                     # [T,K,3]
CIDX = cz["idx"]
MEM = [cz["members"][cz["member_ptr"][i]:cz["member_ptr"][i + 1]]
       for i in range(len(CIDX))]
CCFG = json.loads(str(cz["cfg"])) if "cfg" in cz else {}
RAD = float(CCFG.get("radius", 0.1)) * EXT
print(f"[씬] 입자 {N_ALL}, 크기 {EXT:.3f}, 프레임 {NT}, 제어점 {len(CIDX)} "
      f"(반경 {RAD:.3f}, 잡은 입자 {[len(m) for m in MEM]})", flush=True)

T0, L = a.t0, min(a.frames, NT - 1 - a.t0)
GT = [torch.from_numpy(rd(FS[T0 + i])).to(dev) for i in range(L + 1)]
GTF = []
for i in range(L + 1):
    f = rd(FS[T0 + i], "f_tensor")
    GTF.append(torch.from_numpy(f).to(dev).reshape(-1, 3, 3) if f is not None
               else None)
V0 = (GT[0] - torch.from_numpy(rd(FS[max(T0 - 1, 0)])).to(dev)) \
    / float(json.load(open(a.config))["frame_dt"])
DT = float(json.load(open(a.config))["frame_dt"])
MEM_T = [torch.from_numpy(m.astype(np.int64)).to(dev) for m in MEM]
OFF = [GT[0][m] - GT[0][int(c)] for m, c in zip(MEM_T, CIDX)]


def force(x, t):
    """제어점이 잡은 입자는 어느 방법이든 궤적 값으로 박는다 (같은 입력 신호)."""
    x = x.clone()
    tt = min(T0 + t, CP.shape[0] - 1)
    for k, m in enumerate(MEM_T):
        if k < CP.shape[1]:
            x[m] = CP[tt, k] + OFF[k]
    return x


# ------------------------------------------------------------------ 베이스라인
def bl_still():
    return [(GT[0].clone(), None) for _ in range(L + 1)]


def bl_linear():
    out = []
    for i in range(L + 1):
        out.append((force(GT[0] + V0 * (i * DT), i), None))
    return out


def bl_drag():
    """제어점 변위를 가우시안 감쇠로 퍼뜨린다. 물리는 없다.

    u(x) = sum_k w_k (c_k(t) - c_k(t0)),  w_k = exp(-|x-c_k(t0)|^2 / 2 r^2)
    J    = I - sum_k w_k du_k (x) (x - c_k) / r^2
    """
    c0 = CP[min(T0, CP.shape[0] - 1)]
    d = torch.cdist(GT[0], c0)                                # [N,K]
    w = torch.exp(-0.5 * (d / RAD) ** 2)
    out = []
    for i in range(L + 1):
        dc = CP[min(T0 + i, CP.shape[0] - 1)] - c0            # [K,3]
        u = w @ dc
        x = force(GT[0] + u, i)
        J = torch.eye(3, device=dev).expand(N_ALL, 3, 3).clone()
        for k in range(dc.shape[0]):
            rel = (GT[0] - c0[k]) / (RAD ** 2)
            J -= (w[:, k:k + 1] * dc[k]).unsqueeze(-1) * rel.unsqueeze(-2)
        out.append((x, J))
    return out


def bl_h5(dirname):
    fs = frames_of(dirname)
    if not fs:
        return None
    out = []
    for i in range(L + 1):
        j = min(T0 + i, len(fs) - 1)
        x = torch.from_numpy(rd(fs[j])).to(dev)
        f = rd(fs[j], "f_tensor")
        out.append((x, torch.from_numpy(f).to(dev).reshape(-1, 3, 3)
                    if f is not None else None))
    return out


def bl_student():
    d = torch.load(os.path.join(a.data, a.traj + ".pt"), map_location="cpu",
                   weights_only=False)
    cfg = d["cfg"]
    Xs0 = d["x"][0]
    ext_s = float((Xs0.max(0).values - Xs0.min(0).values).norm())
    n_sub = Xs0.shape[0]
    ng = int(cfg.get("n_grid", 100))
    dxg = float(cfg.get("grid_lim", 2.0)) / ng
    vi = (Xs0.to(dev) / dxg).long().clamp(0, ng - 1)
    flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
    cnt = torch.zeros(ng ** 3, device=dev).index_add_(
        0, flat, torch.ones(n_sub, device=dev))
    mass = ((dxg ** 3) / cnt[flat]) * float(cfg["density"])
    vel_scale = ext_s / DT
    mat = torch.cat([torch.tensor(
        [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
         np.log(float(cfg["density"]))], device=dev, dtype=torch.float32),
        torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])
    cp = d["ctrl_pos"].to(dev) if "ctrl_pos" in d else None
    ctrl = []
    if cp is not None:
        for k_, mm in enumerate(d["ctrl_mem"]):
            loc = mm.to(dev)
            ctrl.append((loc, (d["x"][0][mm] - d["x"][0][d["ctrl"][k_]]).to(dev)))
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta, aidx, H = st["args"], st["aidx"].to(dev), float(st["H"])
    n_ctrl, use_ctrl = int(ta.get("n_ctrl", 4)), bool(ta.get("control", False))
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * ext_s, h=H, ext=ext_s, seed=int(ta["seed"]),
                    damage=bool(ta.get("damage", False))).to(dev).eval()
    net.load_state_dict(st["net"])
    k = int(ta["k"])

    def cfeat(t, pa):
        if cp is None:
            return torch.zeros(pa.shape[0], 6 * n_ctrl, device=dev)
        tt = min(t, cp.shape[0] - 1)
        c = cp[tt]
        dc = cp[min(tt + 1, cp.shape[0] - 1)] - c
        if c.shape[0] < n_ctrl:
            z = torch.zeros(n_ctrl - c.shape[0], 3, device=dev)
            c, dc = torch.cat([c, z]), torch.cat([dc, z])
        c, dc = c[:n_ctrl], dc[:n_ctrl]
        return torch.cat([((pa.unsqueeze(1) - c.unsqueeze(0)) / H).reshape(pa.shape[0], -1),
                          (dc.unsqueeze(0).expand(pa.shape[0], n_ctrl, 3) / H)
                          .reshape(pa.shape[0], -1)], -1)

    x = d["x"][T0].to(dev)
    v = (x - d["x"][max(T0 - 1, 0)].to(dev)) / DT
    p = d["x"][T0][aidx.cpu()].to(dev)
    xc = d["x"][0].to(dev)
    g = GT[0].clone()
    Fg = torch.eye(3, device=dev).expand(N_ALL, 3, 3).contiguous()
    out = [(g.clone(), Fg.clone())]
    for it in range(L):
        idx, _ = grid_knn(x, p, k)
        feat, _ = aggregate(x, v / vel_scale, xc, mass, idx, p.shape[0], H, pa=p)
        extra = torch.cat([mat.reshape(1, -1).expand(p.shape[0], -1),
                           bc_features(p, cfg) / H], -1)
        if use_ctrl:
            extra = torch.cat([extra, cfeat(T0 + it, p)], -1)
        o = net(p, torch.cat([feat, extra], -1), DT)
        dp, lr_, lt_ = o[0], o[1], o[2]
        x2, _ = skin(x, p, dp, lr_, lt_, idx, H)
        if use_ctrl and cp is not None:
            t1 = min(T0 + it + 1, cp.shape[0] - 1)
            x2 = x2.clone()
            for k_, (loc, off) in enumerate(ctrl):
                if k_ < cp.shape[1] and loc.numel():
                    x2[loc] = cp[t1, k_] + off
        # 가우시안은 19 만 개라 한 번에 이웃을 찾으면 카드가 터진다 -- 토막 낸다
        gs_, Js_ = [], []
        for s0 in range(0, g.shape[0], 40000):
            gg = g[s0:s0 + 40000]
            gi, _ = grid_knn(gg, p, k)
            o2, _w, Jb = skin_with_jacobian(gg, p, dp, lr_, lt_, gi, H)
            gs_.append(o2); Js_.append(Jb)
        g2, J = torch.cat(gs_), torch.cat(Js_)
        Fg = torch.bmm(J, Fg)
        g2 = force(g2, it + 1)
        v, p, x, g = (x2 - x) / DT, p + dp, x2, g2
        out.append((g.clone(), Fg.clone()))
    return out


METHODS = [("정지", bl_still()), ("등속", bl_linear()), ("제어점 끌기", bl_drag())]
if a.gf_h5:
    gf = bl_h5(a.gf_h5)
    if gf is not None and gf[0][0].shape[0] == N_ALL:
        METHODS.append(("다른 솔버(GF)", gf))
    elif gf is not None:
        print(f"[주의] 다른 솔버의 입자 수가 다르다 ({gf[0][0].shape[0]} vs "
              f"{N_ALL}) -- 입자 지표는 CD/EMD 만 유효하다", flush=True)
        METHODS.append(("다른 솔버(GF)", gf))
METHODS.append(("학생(ours)", bl_student()))
print(f"[방법] {', '.join(n for n, _ in METHODS)}", flush=True)


# ------------------------------------------------------------------ 지표
rng = np.random.default_rng(a.seed)
SUB = torch.from_numpy(np.sort(rng.choice(N_ALL, min(a.cd_sample, N_ALL),
                                          replace=False))).to(dev)
SUB_E = SUB[torch.from_numpy(
    np.sort(rng.choice(len(SUB), min(a.emd_sample, len(SUB)),
                       replace=False))).to(dev)]
FREE = torch.ones(N_ALL, dtype=torch.bool, device=dev)
for m in MEM_T:
    FREE[m] = False
print(f"[지표] 자유 입자 {int(FREE.sum())}/{N_ALL}, CD {len(SUB)} 점, "
      f"EMD {len(SUB_E)} 점", flush=True)


def chamfer(p, q):
    d = torch.cdist(p, q)
    return 0.5 * float(d.min(1).values.mean() + d.min(0).values.mean())


def emd(p, q):
    from scipy.optimize import linear_sum_assignment
    d = torch.cdist(p, q).double().cpu().numpy()
    r, c = linear_sum_assignment(d)
    return float(d[r, c].mean())


# LPIPS 는 있으면 쓰고 없으면 건너뛴다 (설치를 강요하지 않는다)
try:
    import lpips as _lpips
    LP = _lpips.LPIPS(net="alex").to(dev)
    print("[지표] LPIPS(alex) 사용", flush=True)
except Exception as _e:                                    # noqa: BLE001
    LP = None
    print(f"[지표] LPIPS 없음 ({type(_e).__name__}) -- PSNR/SSIM 만", flush=True)


def lpips_of(x, y):
    if LP is None:
        return None
    def t(im):
        v = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
        return v.permute(2, 0, 1).unsqueeze(0).to(dev)
    return float(LP(t(x), t(y)).item())


res = {n: dict(rmse=[], cd=[], emd=[], psnr=[], ssim=[], lpips=[])
       for n, _ in METHODS}
scene = GSScene(a.model_path, a.config, X0)
print(f"[3DGS] 가우시안 {scene.gs_num} / 렌더 입자 {scene.n}, "
      f"첫 프레임 대응 오차 {scene.fit:.2e}", flush=True)
scene.mark(np.concatenate(MEM))


def ssim(x, y):
    """전역 SSIM (회색조). 창 단위가 아니라 값은 낙관적이지만 순위 비교용이다."""
    x = x.astype(np.float64).mean(-1) / 255.0
    y = y.astype(np.float64).mean(-1) / 255.0
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    vxy = ((x - mx) * (y - my)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mx * my + c1) * (2 * vxy + c2))
                 / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


vid = []
for i in range(L + 1):
    gt = GT[i]
    gt_img = scene.render(gt, GTF[i], i)
    tiles = [("정답(PG)", gt_img)]
    for name, seq in METHODS:
        x, F = seq[min(i, len(seq) - 1)]
        if x.shape[0] == N_ALL:
            res[name]["rmse"].append(
                float((x[FREE] - gt[FREE]).norm(dim=-1).mean()) / EXT)
        res[name]["cd"].append(chamfer(x[SUB] if x.shape[0] == N_ALL else x,
                                       gt[SUB]) / EXT)
        if i % 5 == 0:
            res[name]["emd"].append(
                emd(x[SUB_E] if x.shape[0] == N_ALL else x[:len(SUB_E)],
                    gt[SUB_E]) / EXT)
        img = scene.render(x, F, i)
        mse = np.mean((img.astype(np.float64) - gt_img.astype(np.float64)) ** 2)
        res[name]["psnr"].append(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))))
        res[name]["ssim"].append(ssim(img, gt_img))
        _lp = lpips_of(img, gt_img)
        if _lp is not None:
            res[name]["lpips"].append(_lp)
        tiles.append((name, img))
    if not a.no_video:
        row = []
        for nm, im in tiles:
            pil = Image.fromarray(im)
            dr = ImageDraw.Draw(pil)
            dr.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
            dr.text((4, 2), f"{nm}  f{i:03d}", fill=(255, 255, 255))
            row.append(np.array(pil))
        vid.append(np.concatenate(row, 1))
    if i % 10 == 0:
        print(f"  프레임 {i}/{L}", flush=True)

os.makedirs(a.out, exist_ok=True)
if vid:
    mp4 = os.path.join(a.out, f"compare_{a.traj}_t{T0}.mp4")
    imageio.mimsave(mp4, vid, fps=a.fps, quality=8)
    print(f"[저장] {mp4}  {len(vid)} 프레임, {vid[0].shape[1]}x{vid[0].shape[0]}",
          flush=True)

tab = {}
for name, _ in METHODS:
    r = res[name]
    tab[name] = dict(
        rmse=100 * float(np.mean(r["rmse"])) if r["rmse"] else None,
        cd=100 * float(np.mean(r["cd"])), emd=100 * float(np.mean(r["emd"])),
        psnr=float(np.mean(r["psnr"])), ssim=float(np.mean(r["ssim"])),
        lpips=float(np.mean(r["lpips"])) if r["lpips"] else None)
json.dump(dict(scene=dict(n=N_ALL, ext=EXT, frames=L, t0=T0), per_frame=res,
               summary=tab), open(os.path.join(a.out, "metrics.json"), "w"),
          indent=1, ensure_ascii=False)

print("\n" + "=" * 83)
print(f"{'방법':<16}{'RMSE%':>9}{'CD%':>9}{'EMD%':>9}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}")
print("-" * 83)
for name, v in tab.items():
    rm = f"{v['rmse']:.3f}" if v["rmse"] is not None else "  -  "
    lp = f"{v['lpips']:.4f}" if v["lpips"] is not None else "  -  "
    print(f"{name:<16}{rm:>9}{v['cd']:>9.3f}{v['emd']:>9.3f}"
          f"{v['psnr']:>9.2f}{v['ssim']:>9.4f}{lp:>9}")
print("=" * 83)
print("(RMSE/CD/EMD 는 물체 크기 대비 %, 낮을수록 좋다. PSNR/SSIM 은 정답 렌더 대비)")
print("EVAL_BASELINES_OK", flush=True)
