"""교사 MPM 과 학생 예측을 **3DGS 스플랫**으로 나란히 그린다.

입자를 점으로 찍으면 "어디가 얼마나 어긋났나" 는 보이지만 실제로 눈에 보이는
장면이 어떻게 무너지는지는 알 수 없다. 여기서는 PhysGaussian/GaussianFluent 가
쓰는 것과 같은 전처리·같은 카메라·같은 래스터라이저로 칸을 나눠 그린다.

  GT      교사 MPM 이 굴린 입자 위치와 변형구배 F 로 가우시안을 옮기고 늘린다
  정지    첫 프레임 그대로
  학생    앵커 스테퍼가 낸 변위장으로 가우시안을 스키닝한다 (야코비안이 F 역할)

  PYTHONPATH=<PG 또는 GF>:<gaussian-splatting>:<lib> python exe/render_gs_compare.py \
     --model_path <3DGS> --config <씬 config> --h5 <교사 h5 디렉토리> \
     [--data <롤아웃 dir> --traj eval_000 --ckpt <학생.pt>] --out out.mp4
"""
from __future__ import annotations

import argparse
import glob
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
ap.add_argument("--h5", required=True, help="교사 h5 디렉토리 (sim_*.h5)")
ap.add_argument("--data", default=None, help="롤아웃 .pt 디렉토리 (학생 칸용)")
ap.add_argument("--traj", default=None)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=int, default=3)
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--png_dir", default=None,
                help="mp4 대신 프레임을 PNG 로 남긴다 (토막 렌더용)")
ap.add_argument("--white_bg", type=int, default=1)
ap.add_argument("--mark_ctrl", type=int, default=1,
                help="제어점이 잡은 가우시안을 빨갛게 (h5 옆 control.npz)")
ap.add_argument("--grips", default=None,
                help="gripseq.npy -- 평행 집게 두 판을 실제로 그려 준다")
ap.add_argument("--no_F", action="store_true",
                help="변형구배로 커널 모양을 바꾸지 않는다 (렌더가 불안정할 때)")
ap.add_argument("--hide_stretch", type=float, default=None,
                help="||F||_F 가 이 값을 넘은 가우시안은 그리지 않는다 (찢긴 자리)")
ap.add_argument("--panels", default="gt,still",
                help="그릴 칸: gt,still (학생은 --ckpt 를 주면 자동으로 붙는다)")
a = ap.parse_args()

dev = "cuda:0"
torch.set_grad_enabled(False)
import imageio.v2 as imageio                                      # noqa: E402
import taichi as ti                                               # noqa: E402
from PIL import Image, ImageDraw                                  # noqa: E402
from scipy.spatial import ConvexHull                              # noqa: E402

# taichi 가 통째로 잡는 양. 카드를 다른 작업과 나눠 쓸 때는 줄여야 죽지 않는다
ti.init(arch=ti.cuda, device_memory_GB=float(os.environ.get("AF_TI_GB", "4.0")))
from anchorflow.gsrender import GSScene                           # noqa: E402


def rd(p, key="x"):
    with h5py.File(p, "r") as h:
        if key not in h:
            return None
        v = np.array(h[key])
    return (v.T if v.shape[0] in (3, 9) else v).astype(np.float32)


FS = sorted(glob.glob(os.path.join(a.h5, "sim_*.h5")))
if not FS:
    raise SystemExit(f"h5 가 없다: {a.h5}")
NT = len(FS)
X0 = torch.from_numpy(rd(FS[0])).to(dev)
scene = GSScene(a.model_path, a.config, X0, device=dev, white_bg=bool(a.white_bg))
print(f"[대응] 가우시안 {scene.gs_num} / 입자 {X0.shape[0]}, 첫 프레임 최대 차 "
      f"{scene.fit:.3e} {'(일치)' if scene.fit < 1e-4 else '(어긋난다!)'}", flush=True)

import json as _js
FRAME_DT_G = float(_js.load(open(a.config))["frame_dt"])

CNP = os.path.join(a.h5, "control.npz")
CZ = np.load(CNP) if os.path.exists(CNP) else None
if a.mark_ctrl and CZ is not None:
    print(f"[표시] 제어점이 잡은 가우시안 {scene.mark(CZ['members'])} 개를 빨갛게",
          flush=True)

# ------------------------------------------------- 집게 접촉면 (시뮬이 남긴 자세)
GRIPS = None
if a.grips and os.path.exists(a.grips):
    # 시뮬이 프레임마다 기록한 (중심, 회전, 벌어짐) 을 그대로 읽는다.
    # 자세를 렌더에서 다시 계산하면 시뮬과 어긋난다 (겪었다).
    POSE = [list(p) for p in np.load(a.grips, allow_pickle=True)]
    _gs = _js.load(open(a.config)).get("grip_seq", {})
    _gp = _js.load(open(a.config)).get("grip_plate", {})
    EXT0 = float((X0.max(0).values - X0.min(0).values).norm())   # 도구 크기는 고정
    if _gp:
        # 물리 집게: 기록된 간격비(1=벌림, grip=물었을 때)를 그대로 쓴다
        JAW = float(_gp.get("jaw", 0.10)) * EXT0
        HALF = np.array([float(_gp.get("thick", 0.02)),
                         float(_gp.get("pad_w", 0.10)),
                         float(_gp.get("pad_d", 0.10))]) * EXT0
        PLATE_MODE = "gap"
    else:
        JAW = float(_gs.get("jaw", 0.10)) * EXT0
        HALF = np.array([0.035, float(_gs.get("pad_w", 0.22)),
                         float(_gs.get("pad_d", 0.22))]) * EXT0
        PLATE_MODE = "open"

    def _box(hx, hy, hz, step):
        ax = [np.arange(-h, h + 1e-9, min(step, max(2 * h, 1e-6)))
              for h in (hx, hy, hz)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)

    def _pads(g):
        """닿는 판 두 장. g 는 간격비(물리 집게) 또는 벌어짐 0~1(예전 방식)."""
        # 물리 집게는 자세 기록의 세 번째 값이 **절대 반간격**이다
        gap = float(g) if PLATE_MODE == "gap" else JAW * (1.0 + 1.8 * g)
        plate = _box(HALF[0], HALF[1], HALF[2], 0.35 * float(HALF[1]))
        out = []
        for sgn in (+1, -1):
            q = plate.copy()
            q[:, 0] += sgn * (gap + HALF[0])
            out.append(q)
        return np.concatenate(out)

    def _corners(g):
        """판 두 장의 8 꼭지점 (집게 국소 좌표)."""
        # 물리 집게는 자세 기록의 세 번째 값이 **절대 반간격**이다
        gap = float(g) if PLATE_MODE == "gap" else JAW * (1.0 + 1.8 * g)
        out = []
        for sgn in (+1, -1):
            cs = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        cs.append([sgn * (gap + HALF[0]) + sx * HALF[0],
                                   sy * HALF[1], sz * HALF[2]])
            out.append(np.array(cs))
        return out

    def grip_quads(frame):
        """이 프레임 판들의 꼭지점 (시뮬 좌표) 목록."""
        if frame >= len(POSE) or len(POSE[frame]) == 0:
            return []
        out = []
        for (c, R, g) in POSE[frame]:
            for cs in _corners(float(g)):
                out.append(np.asarray(c) + cs @ np.asarray(R))
        return out

    def grip_points(frame):
        if frame >= len(POSE) or len(POSE[frame]) == 0:
            return None
        out = []
        for (c, R, op) in POSE[frame]:
            q = _pads(float(op))
            out.append(np.asarray(c) + q @ np.asarray(R))
        return torch.from_numpy(np.concatenate(out)).float() if out else None

    GRIPS = grip_points
    print(f"[집게] 자세 기록 {len(POSE)} 프레임, 도구 크기 고정 "
          f"(물체 초기 지름 {EXT0:.3f} 기준)", flush=True)

# ------------------------------------------------- 학생 롤아웃 (앵커 스키닝)
STU = None
if a.ckpt and a.data and a.traj:
    from anchorflow.deform import (DeformNet, aggregate, bc_features,  # noqa
                                   grid_knn, skin, skin_with_jacobian)
    d = torch.load(os.path.join(a.data, a.traj + ".pt"), map_location="cpu",
                   weights_only=False)
    cfg = d["cfg"]
    DT = float(cfg["frame_dt"])
    Xs0 = d["x"][0]
    EXT = float((Xs0.max(0).values - Xs0.min(0).values).norm())
    n_sub = Xs0.shape[0]
    ng = int(cfg.get("n_grid", 100))
    dxg = float(cfg.get("grid_lim", 2.0)) / ng
    vi = (Xs0.to(dev) / dxg).long().clamp(0, ng - 1)
    flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
    cnt = torch.zeros(ng ** 3, device=dev).index_add_(
        0, flat, torch.ones(n_sub, device=dev))
    MASS = ((dxg ** 3) / cnt[flat]) * float(cfg["density"])
    VEL_SCALE = EXT / DT
    MAT = torch.cat([torch.tensor(
        [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
         np.log(float(cfg["density"]))], device=dev, dtype=torch.float32),
        torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0])
    CP = d["ctrl_pos"].to(dev) if "ctrl_pos" in d else None
    CTRL = []
    if CP is not None:
        for k_, mm in enumerate(d["ctrl_mem"]):
            CTRL.append((mm.to(dev),
                         (d["x"][0][mm] - d["x"][0][d["ctrl"][k_]]).to(dev)))
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta, AIDX, H = st["args"], st["aidx"].to(dev), float(st["H"])
    n_ctrl, use_ctrl = int(ta.get("n_ctrl", 4)), bool(ta.get("control", False))
    net = DeformNet(n_feat=int(st["n_feat"]), hidden=int(ta["hidden"]),
                    depth=int(ta["depth"]), heads=int(ta["heads"]),
                    scale=0.02 * EXT, h=H, ext=EXT, seed=int(ta["seed"]),
                    damage=bool(ta.get("damage", False))).to(dev).eval()
    net.load_state_dict(st["net"])
    k = int(ta["k"])
    # 교사가 박은 입자는 전체 구름에서도 궤적 값으로 덮어쓴다
    GMEM, GOFF = [], []
    if CZ is not None:
        mp = CZ["member_ptr"]
        for i in range(len(CZ["idx"])):
            m = torch.from_numpy(
                CZ["members"][mp[i]:mp[i + 1]].astype(np.int64)).to(dev)
            GMEM.append(m)
            GOFF.append(X0[m] - X0[int(CZ["idx"][i])])
    GCP = torch.from_numpy(CZ["pos"]).to(dev) if CZ is not None else None

    def cfeat(t, pa):
        if CP is None:
            return torch.zeros(pa.shape[0], 6 * n_ctrl, device=dev)
        tt = min(t, CP.shape[0] - 1)
        c = CP[tt]
        dc = CP[min(tt + 1, CP.shape[0] - 1)] - c
        if c.shape[0] < n_ctrl:
            z = torch.zeros(n_ctrl - c.shape[0], 3, device=dev)
            c, dc = torch.cat([c, z]), torch.cat([dc, z])
        c, dc = c[:n_ctrl], dc[:n_ctrl]
        A = pa.shape[0]
        return torch.cat([((pa.unsqueeze(1) - c.unsqueeze(0)) / H).reshape(A, -1),
                          (dc.unsqueeze(0).expand(A, n_ctrl, 3) / H).reshape(A, -1)], -1)

    x = d["x"][a.t0].to(dev)
    v = (x - d["x"][max(a.t0 - 1, 0)].to(dev)) / DT
    p = d["x"][a.t0][AIDX.cpu()].to(dev)
    XC = d["x"][0].to(dev)
    g = torch.from_numpy(rd(FS[min(a.t0, NT - 1)])).to(dev)
    Fg = torch.eye(3, device=dev).expand(g.shape[0], 3, 3).contiguous()
    STU = [(g.clone(), Fg.clone())]
    for it in range(a.frames):
        idx, _ = grid_knn(x, p, k)
        feat, _ = aggregate(x, v / VEL_SCALE, XC, MASS, idx, p.shape[0], H, pa=p)
        extra = torch.cat([MAT.reshape(1, -1).expand(p.shape[0], -1),
                           bc_features(p, cfg) / H], -1)
        if use_ctrl:
            extra = torch.cat([extra, cfeat(a.t0 + it, p)], -1)
        o = net(p, torch.cat([feat, extra], -1), DT)
        dp, lr_, lt_ = o[0], o[1], o[2]
        x2, _ = skin(x, p, dp, lr_, lt_, idx, H)
        if use_ctrl and CP is not None:
            t1 = min(a.t0 + it + 1, CP.shape[0] - 1)
            x2 = x2.clone()
            for k_, (loc, off) in enumerate(CTRL):
                if k_ < CP.shape[1] and loc.numel():
                    x2[loc] = CP[t1, k_] + off
        # 가우시안은 수십만 개라 한 번에 이웃을 찾으면 카드가 터진다 -- 토막 낸다
        gs_, Js_ = [], []
        for s0 in range(0, g.shape[0], 40000):
            gg = g[s0:s0 + 40000]
            gi, _ = grid_knn(gg, p, k)
            o2, _w, Jb = skin_with_jacobian(gg, p, dp, lr_, lt_, gi, H)
            gs_.append(o2); Js_.append(Jb)
        g2, J = torch.cat(gs_), torch.cat(Js_)
        Fg = torch.bmm(J, Fg)
        if GCP is not None:
            t1 = min(a.t0 + it + 1, GCP.shape[0] - 1)
            for k_, m in enumerate(GMEM):
                if k_ < GCP.shape[1]:
                    g2[m] = GCP[t1, k_] + GOFF[k_]
        v, p, x, g = (x2 - x) / DT, p + dp, x2, g2
        STU.append((g.clone(), Fg.clone()))

def draw_grips(img, frame, cam_frame):
    """집게 판을 2D 다각형으로 덧그린다.

    가우시안으로 섞어 래스터라이저에 넘기면 카메라 뒤 점 때문에 터진다 (겪었다).
    판은 평면이라 투영해서 채워 그리면 충분하다.
    """
    quads = grip_quads(frame)
    if not quads:
        return img
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil, "RGBA")
    for q in quads:
        px, dep = scene.project(torch.from_numpy(q).float(), cam_frame)
        if (dep <= 0.2).any():
            continue
        pts = px[ConvexHull(px).vertices] if len(px) > 2 else px
        dr.polygon([tuple(p) for p in pts], fill=(38, 40, 46, 235))
    return np.array(pil)


# ------------------------------------------------- 그리기
want = [w.strip() for w in a.panels.split(",") if w.strip()]
frames = []
for i in range(a.frames + 1):
    f = min(a.t0 + i, NT - 1)
    xg = torch.from_numpy(rd(FS[f])).to(dev)
    ft = rd(FS[f], "f_tensor")
    Fgt = (torch.from_numpy(ft).to(dev).reshape(-1, 3, 3)
           if (ft is not None and not a.no_F) else None)
    panes = []
    if "gt" in want:
        _img = scene.render(xg, Fgt, i, a.hide_stretch)
        if GRIPS is not None:
            _img = draw_grips(_img, a.t0 + i, i)
        panes.append(("GT (교사 MPM)", _img))
    if "still" in want:
        panes.append(("정지", scene.render(X0, None, i)))
    if STU is not None:
        gs, Fs = STU[min(i, len(STU) - 1)]
        panes.append(("학생", scene.render(gs, Fs, i, a.hide_stretch)))
    imgs = []
    for nm, im in panes:
        pil = Image.fromarray(im)
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
        dr.text((4, 2), f"{nm}   frame {i:03d}", fill=(255, 255, 255))
        imgs.append(np.array(pil))
    row = np.concatenate(imgs, 1)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
        Image.fromarray(row).save(os.path.join(a.png_dir, f"f{a.t0 + i:05d}.png"))
    else:
        frames.append(row)
    if i % 20 == 0:
        print(f"  프레임 {i}/{a.frames}", flush=True)

if a.png_dir:
    print(f"[저장] {a.png_dir} 에 PNG {a.frames + 1} 장", flush=True)
    print("GSVID_OK", flush=True)
    raise SystemExit
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
imageio.mimsave(a.out, frames, fps=a.fps, quality=8)
print(f"[저장] {a.out}  {len(frames)} 프레임, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}", flush=True)
print("GSVID_OK", flush=True)
