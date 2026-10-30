"""lb_gen.py 궤적을 3DGS 알파블렌딩으로 렌더 (rep_ip 와 같은 렌더러·카메라 규약).

가우시안 ↔ 입자: lego 는 채움 캐시 앞 NG 개가 불투명도 필터를 거친 가우시안 그대로라 **같은 번호**로 대응한다
(rep_ip 처럼 0 프레임 최근접으로 짝지으면 lb_gen 의 초기 회전 때문에 틀어진다). 시작 시 회전 전 입력 h5 로 대응을 검산한다.
공분산 = F C0 Fᵀ (F 에 초기 회전이 실려 있다).

  python exe/lb_render.py --sim DIR --config cfg.json --shape lego --out DIR/video.mp4
"""
import argparse
import glob
import json
import math
import os
import sys

W = "/home/dkta/work"
MODEL = {"lego": "lego_whitebg-trained", "mic": "mic_whitebg-trained", "ficus": "ficus_whitebg-trained",
         "hotdog": "hotdog_whitebg-trained", "wolf": "wolf_whitebg-trained", "bread": "bread-trained"}          # 절 AG: PG 소성 씬
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--shape", default="lego")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--iter", type=int, default=-1, help="3DGS 체크포인트 반복. -1 이면 PG load_checkpoint 처럼 가장 큰 반복")
ap.add_argument("--colors", default="", help="가우시안 색 덮어쓰기 .npy [T,>=NG,3] (h5 행 순서, 앞 NG 개를 쓴다) (0~1, 프레임 순서 = sim_*.h5 정렬 순). "
                "주면 SH 대신 colors_precomp 로 래스터화한다 (불투명도·공분산은 그대로)")
ap.add_argument("--iso", type=float, default=0.0, help="공분산을 등방형 (iso × 가우시안 축 길이 중앙값)² I 로 (F 무시). 0 이면 FΣFᵀ")
ap.add_argument("--frame_label", type=int, default=None, help="주면 프레임 번호 (이 값 + 영상 프레임 순번) 를 크게 빨간색으로 찍는다")
ap.add_argument("--cam_back", type=float, default=1.6, help="카메라를 물체 중심에서 이 배로 뒤로 (기본 1.6, 기존 그대로)")
ap.add_argument("--caption", default="", help="--frame_label 과 함께 왼쪽 위에 찍을 설명 (한국어 가능)")
ap.add_argument("--frames", type=int, default=0, help="앞 이 상태 번호까지만 (0 = 전부; sim_*.h5 정렬 순 0..frames)")
ap.add_argument("--mark", default="", help="사용자 10-30 발산 셀 입자 표시: 입자 번호 .npy -- 그 입자를 빨강으로 (가우시안 < NG 는 색만, 채움 입자 ≥ NG 는 "
                "등방 가우시안 (축 = 가우시안 축 길이 중앙값 x --mark_size, 공분산 F s² Fᵀ) 을 더해 그린다). 나머지는 SH 그대로 (래스터라이저와 같은 식으로 colors_precomp)")
ap.add_argument("--mark_fade", type=float, default=1.0, help="--mark: 표시 안 한 가우시안 불투명도 배율 (투시 칸, 1 = 그대로)")
ap.add_argument("--mark_size", type=float, default=1.0)
ap.add_argument("--mark_px", default="", help="--mark: 프레임마다 표시 입자 화면 위치 (중앙·범위 픽셀) -> json (확대 칸 중심)")
a = ap.parse_args()
import h5py                                                      # noqa: E402
import imageio.v2 as imageio                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
torch.set_grad_enabled(False)
dev = "cuda"
cfg = json.load(open(a.config))
meta = json.load(open(f"{a.sim}/meta.json"))
sys.path.append(f"{W}/i-physgaussian"); sys.path.append(f"{W}/i-physgaussian/gaussian-splatting")
_cwd = os.getcwd(); os.chdir(f"{W}/i-physgaussian")
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.transformation_utils import transform2origin, shift2center111   # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix      # noqa: E402
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer   # noqa: E402
os.chdir(_cwd)
mp = f"{W}/pgmodel/{MODEL[a.shape]}"
gs = GaussianModel(3)
_it = a.iter if a.iter >= 0 else max(int(q.split("_")[-1]) for q in os.listdir(f"{mp}/point_cloud") if q.startswith("iteration_"))
print(f"[3DGS] {mp} iteration_{_it}", flush=True)
gs.load_ply(f"{mp}/point_cloud/iteration_{_it}/point_cloud.ply")
KIDX = torch.nonzero(gs.get_opacity[:, 0] > float(cfg.get("opacity_threshold", 0.02))).squeeze(1)
# 절 AG: PG 전처리 회전 (config rotation_degree/axis -- PG generate_rotation_matrix 와 같은 식, apply_rotations 순서대로) -- 없으면 항등 (기존과 같다)
_RT = torch.eye(3, device=dev)
for _dg, _ax in zip(cfg.get("rotation_degree") or [], cfg.get("rotation_axis") or []):
    _c, _s = math.cos(_dg / 180.0 * 3.1415926), math.sin(_dg / 180.0 * 3.1415926)
    _R = {0: [[1, 0, 0], [0, _c, -_s], [0, _s, _c]], 1: [[_c, 0, _s], [0, 1, 0], [-_s, 0, _c]], 2: [[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]]}[int(_ax)]
    _RT = torch.tensor(_R, device=dev, dtype=torch.float32) @ _RT
ROT = bool((_RT - torch.eye(3, device=dev)).abs().max() > 0)
TP, SO, MEAN = transform2origin(gs.get_xyz[KIDX] @ _RT.T if ROT else gs.get_xyz[KIDX], float(cfg.get("scale", 1.0)))
TP = shift2center111(TP)
NG = KIDX.numel()
with h5py.File(meta["h5"]) as h:
    x0 = np.array(h["x"]); x0 = x0.T if x0.shape[0] == 3 else x0
err = float((torch.as_tensor(x0[:NG], device=dev).float() - TP).norm(dim=1).max())
print(f"[대응] 가우시안 {NG}, 회전 전 입자 앞 {NG} 개와 위치 차이 최대 {err:.2e}", flush=True)
assert err < 1e-4, "채움 캐시 앞부분이 가우시안과 같은 번호가 아니다"
c6 = gs.get_covariance()[KIDX]
C0 = torch.zeros(NG, 3, 3, device=dev)
C0[:, 0, 0], C0[:, 0, 1], C0[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
C0[:, 1, 1], C0[:, 1, 2], C0[:, 2, 2] = c6[:, 3], c6[:, 4], c6[:, 5]
C0[:, 1, 0], C0[:, 2, 0], C0[:, 2, 1] = c6[:, 1], c6[:, 2], c6[:, 4]
if ROT:
    C0 = _RT @ C0 @ _RT.T                                            # 시뮬 공간 (회전 뒤) 공분산
SHS, OPA = gs.get_features[KIDX], gs.get_opacity[KIDX]
cam = json.load(open(f"{mp}/cameras.json"))[0]
Rw, pos = np.array(cam["rotation"]), np.array(cam["position"])
_ctr = ((MEAN @ _RT) if ROT else MEAN).cpu().numpy().astype(np.float64)
pos = _ctr + a.cam_back * (pos - _ctr)                                     # 회전·낙하·이동이 다 들어오게 뒤로
W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
fx_ = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy_ = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fx_, fovY=fy_).transpose(0, 1).to(dev).float()
RAST = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
    image_height=int(cam["height"]), image_width=int(cam["width"]), tanfovx=math.tan(fx_ * 0.5), tanfovy=math.tan(fy_ * 0.5),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3,
    campos=wv.inverse()[3, :3], prefiltered=False, debug=False))
_FONT = None


def _label(fr, n):
    """프레임 번호를 크게 빨간색으로 (그리고 --caption)."""
    global _FONT
    from PIL import Image, ImageDraw, ImageFont
    if _FONT is None:
        from matplotlib import font_manager as fm
        _cand = [f for f in fm.findSystemFonts() if "Noto" in f and ("CJK" in f or "KR" in f)] or \
                [f for f in fm.findSystemFonts() if "DejaVuSans-Bold" in f] or [fm.findfont("DejaVu Sans")]
        _H = fr.shape[0]
        _FONT = (ImageFont.truetype(_cand[0], int(_H * 0.09)), ImageFont.truetype(_cand[0], int(_H * 0.035)))
    im = Image.fromarray(fr)
    dr = ImageDraw.Draw(im)
    dr.text((int(fr.shape[1] * 0.03), int(fr.shape[0] * 0.02)), f"{n}", fill=(230, 0, 0), font=_FONT[0])
    if a.caption:
        dr.text((int(fr.shape[1] * 0.03), int(fr.shape[0] * 0.13)), a.caption, fill=(230, 0, 0), font=_FONT[1])
    return np.asarray(im)


WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8)
files = sorted(glob.glob(f"{a.sim}/sim_*.h5"))
COL = None
if a.colors:
    COL = np.load(a.colors, mmap_mode="r")
    assert COL.shape[0] == len(files) and COL.shape[1] >= NG and COL.shape[2] == 3, (COL.shape, len(files), NG)
    print(f"[색 덮어쓰기] {a.colors} {tuple(COL.shape)}", flush=True)
if a.frames:
    files = files[:a.frames + 1]
MK = None
if a.mark:
    import dc_mark as DCM                                                               # 색칠만 (exe/dc_mark.py)
    MG, MF = DCM.load_mark(a.mark, NG, dev)
    MK = torch.zeros(NG, dtype=torch.bool, device=dev)
    MK[MG] = True
    S_EX = float(gs.get_scaling[KIDX].median()) * a.mark_size
    CAMPOS = wv.inverse()[3, :3]
    FPROJ = (wv[None] @ pj[None])[0]
    print(f"[표시] {a.mark}: 가우시안 {MG.numel()} + 채움 입자 {MF.numel()} (등방 축 {S_EX:.4g}), 나머지 불투명도 x {a.mark_fade}", flush=True)
    PXL = []
for fi, p in enumerate(files):
    with h5py.File(p) as h:
        x = torch.as_tensor(np.array(h["x"][:NG]), device=dev).float()
        F = torch.as_tensor(np.array(h["F"][:NG]), device=dev).float().reshape(-1, 3, 3)
        if MK is not None and MF.numel():
            _xa = np.array(h["x"]); _xa = _xa.T if _xa.shape[0] == 3 else _xa
            _Fa = np.array(h["F"]); _Fa = _Fa.T if _Fa.shape[0] == 9 else _Fa
            _sel = MF.cpu().numpy()
            xe = torch.as_tensor(_xa[_sel], device=dev).float()
            Fe = torch.as_tensor(_Fa.reshape(-1, 9)[_sel], device=dev).float().reshape(-1, 3, 3)
    if a.iso > 0:
        cov = torch.eye(3, device=dev).expand(NG, 3, 3) * (a.iso * float(gs.get_scaling[KIDX].median())) ** 2
    else:
        cov = F @ C0 @ F.transpose(1, 2)
        if ROT:
            cov = _RT.T @ cov @ _RT
    cc = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1)
    Pm = (x - 1.0) / SO + MEAN
    if ROT:
        Pm = Pm @ _RT
    _cp = torch.as_tensor(np.asarray(COL[fi][:NG], dtype=np.float32), device=dev) if COL is not None else None
    _op = OPA
    if MK is not None:                                                                  # --mark: 색·불투명도만 (위치·공분산 그대로)
        if _cp is None:
            _cp = DCM.sh_rgb(SHS, Pm, CAMPOS)
        _cp = _cp.clone()
        _cp[MK] = torch.tensor(DCM.RED, device=dev)
        _op = OPA.clone()
        if a.mark_fade != 1.0:
            _op[~MK] = _op[~MK] * a.mark_fade
        _mp = [Pm[MK]]
        if MF.numel():
            _ok = torch.isfinite(xe).all(1) & torch.isfinite(Fe).all(2).all(1)
            Pe = (xe[_ok] - 1.0) / SO + MEAN
            ce, cove = DCM.extra_cov(Fe[_ok], S_EX)
            if ROT:
                Pe = Pe @ _RT
                cove = _RT.T @ cove @ _RT
                ce = torch.stack([cove[:, 0, 0], cove[:, 0, 1], cove[:, 0, 2], cove[:, 1, 1], cove[:, 1, 2], cove[:, 2, 2]], 1)
            Pm = torch.cat([Pm, Pe]); cc = torch.cat([cc, ce])
            _cp = torch.cat([_cp, torch.tensor(DCM.RED, device=dev).expand(Pe.shape[0], 3)])
            _op = torch.cat([_op, torch.ones(Pe.shape[0], 1, device=dev)])
            _mp.append(Pe)
        _mp = torch.cat(_mp)
        _mp = _mp[torch.isfinite(_mp).all(1)]
        if _mp.shape[0]:
            _px = DCM.project_px(_mp, FPROJ, int(cam["width"]), int(cam["height"]))
            PXL.append(dict(fi=fi, med=_px.median(0).values.tolist(), lo=_px.min(0).values.tolist(), hi=_px.max(0).values.tolist(), n=int(_mp.shape[0])))
        else:
            PXL.append(dict(fi=fi, med=None))
    img = RAST(means3D=Pm, means2D=torch.zeros_like(Pm), shs=(SHS if _cp is None else None), colors_precomp=_cp, opacities=_op, scales=None,
               rotations=None, cov3D_precomp=cc)[0]
    _fr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if a.frame_label is not None:
        _fr = _label(_fr, a.frame_label + fi)
    WR.append_data(_fr)
WR.close()
if MK is not None and a.mark_px:
    json.dump(dict(W=int(cam["width"]), H=int(cam["height"]), frames=PXL, n_gauss=int(MG.numel()), n_fill=int(MF.numel())), open(a.mark_px, "w"))
print(f"[영상] {a.out} ({len(files)} 프레임)", flush=True)
