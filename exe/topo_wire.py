"""토폴로지 추적 (C) riem 사면체 메쉬 실행의 정체 분석 + 메쉬 겹침 영상.

단계 analyze: 실행 결과 npz (rep_track2 --save_traj) 의 가우시안 위치에서 메쉬 노드 변위 u 를 되찾는다.
  사면체 메쉬는 연결·무게중심 좌표가 첫 프레임에 고정이라 y = G0 + Λu (Λ: 입자 x 노드, 행마다 4 개)
  -> 프레임마다 정규방정식 ΛᵀΛ u = Λᵀ(y - G0) 를 CG 로 (직전 프레임 해에서 시작). 메쉬는 rep_track2 TetMesh 와
  같은 식으로 다시 만든다 (Kuhn 격자, node_h = PG dx·(2√2)^(1/3), 첫 프레임 정지 위치).
  프레임별 RMSE·노드 이동량(직전 대비 평균·최대)·손실 감소율·정답 이동량을 표로 낸다.
단계 render: 정답 | 사면체 메쉬(3DGS) + 메쉬 겹침. 렌더·카메라는 rep_render.py (--iso 1, 영상
  vid_wolf_tetmesh.mp4 를 만든 설정) 와 같은 코드. 겹침은 같은 full_proj 행렬로 노드를 투영.
  겹치는 것: 경계면(사면체 하나에만 속한 면) 중 카메라를 향한 면의 모서리와 꼭짓점.

  python exe/topo_wire.py analyze --flow topo/flow_wolf.npz --aux topo/aux_wolf.npz \
      --res topo/c3/wolf_tetmesh.npz --out topo/wire/wolf_tetmesh_nodes.npz
  python exe/topo_wire.py render  ... --nodes topo/wire/wolf_tetmesh_nodes.npz --stall 20 --video out.mp4
"""
import argparse
import math
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["analyze", "render"])
ap.add_argument("--flow", required=True)
ap.add_argument("--aux", required=True)
ap.add_argument("--res", required=True)
ap.add_argument("--nodes", required=True, help="analyze 출력 / render 입력")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--video", default="")
ap.add_argument("--ref_video", default="", help="비교용 기존 영상 (같은 렌더인지 확인)")
ap.add_argument("--stall", type=int, default=-1)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--iso", type=float, default=1.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--font", default="/home/dkta/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()

# 빈 카드 (없으면 메모리를 가장 적게 쓰는 카드)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    use = sorted((int(m), int(i)) for i, m in (l.split(",") for l in q))
    print("[GPU] 메모리 사용 (MiB):", {i: m for m, i in use}, "-> 카드", use[0][1], flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use[0][1])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
from anchorflow import simplex as sx                                 # noqa: E402

dev = "cuda"
D = np.load(a.flow)
AX = np.load(a.aux, allow_pickle=True)
FIDX = D["idx"]
G0 = torch.as_tensor(AX["G"], dtype=torch.float32, device=dev)[torch.as_tensor(FIDX, device=dev)].contiguous()
NG = G0.shape[0]
L = float((G0.max(0).values - G0.min(0).values).norm())
S_NORM = float(AX["s"])
print(f"[자료] 가우시안 {NG} (aux 전체 {AX['G'].shape[0]}, idx 가 0..N-1 그대로: "
      f"{bool(np.array_equal(FIDX, np.arange(len(FIDX))))})  지름 L {L:.4f}", flush=True)


def build_mesh():
    """rep_track2 TetMesh: Lattice.rebind(G0) 와 같은 식 (float32)."""
    h = (2.0 / 100.0) * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0) / S_NORM
    lo, lat, nn = sx.grid_for_nodes(G0, 1.0, h_fix=h)
    idx, lam, _ = sx.locate(G0, lo, lat, nn)
    rows, uniq = sx.active_nodes(idx)
    Xn = sx.node_pos(lo, lat, nn, uniq)
    cells = torch.unique(rows, dim=0)
    return h, rows, lam, Xn, cells


def surface(cells):
    """경계면 (사면체 하나에만 속한 면) -- 바깥 방향으로 정렬한 삼각형 [F,3] + 맞은편 꼭짓점."""
    c = cells
    fl = torch.cat([c[:, [1, 2, 3]], c[:, [0, 2, 3]], c[:, [0, 1, 3]], c[:, [0, 1, 2]]], 0)
    opp = torch.cat([c[:, 0], c[:, 1], c[:, 2], c[:, 3]], 0)
    key = fl.sort(1).values
    uq, inv, cnt = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    b = cnt[inv] == 1
    return fl[b], opp[b]


if a.stage == "analyze":
    h, rows, lam, Xn, cells = build_mesh()
    M, C = Xn.shape[0], cells.shape[0]
    R = np.load(a.res)
    dof = R["dof"]
    print(f"[메쉬] 노드 간격 h {h:.5f} (정규화), 꼭짓점 {M}, 사면체 {C}, 자유도 3M {3 * M} "
          f"(실행 기록 {int(dof[0])} -> 일치 {3 * M == int(dof[0])})", flush=True)
    Y = R["traj"]                                                      # [T,NG,3] 프레임 1..T
    T = Y.shape[0]
    print(f"[실행] traj {Y.shape} {Y.dtype}, 프레임 {T}", flush=True)
    # Λ (float64 희소)
    ii = torch.arange(NG, device=dev).repeat_interleave(4)
    A = torch.sparse_coo_tensor(torch.stack([ii, rows.reshape(-1)]), lam.reshape(-1).double(),
                                (NG, M)).coalesce().to_sparse_csr()
    AT = torch.sparse_coo_tensor(torch.stack([rows.reshape(-1), ii]), lam.reshape(-1).double(),
                                 (M, NG)).coalesce().to_sparse_csr()
    dg = torch.zeros(M, dtype=torch.float64, device=dev).index_add_(0, rows.reshape(-1), lam.reshape(-1).double() ** 2)
    print(f"[관측] 노드별 Σλ² 최소 {float(dg.min()):.2e}, 중앙 {float(dg.median()):.3f}, "
          f"1e-3 미만 노드 {int((dg < 1e-3).sum())}", flush=True)
    G0d = G0.double()
    u = torch.zeros(M, 3, dtype=torch.float64, device=dev)
    U = [u.clone()]
    rec = []
    for t in range(T):
        b = torch.as_tensor(Y[t], device=dev).double() - G0d
        rhs = AT @ b
        x = U[-1].clone()
        r = rhs - AT @ (A @ x)
        dgc = dg.clamp_min(1e-12)[:, None]
        z = r / dgc
        p = z.clone(); rz = (r * z).sum(0)
        n0 = float(rhs.norm())
        k = -1
        for k in range(3000):                                          # 오른쪽이 0 (프레임 1·2: 정답이 안 움직임) 이면 그대로
            if float(r.norm()) <= 1e-11 * n0 or n0 == 0.0:
                break
            Ap = AT @ (A @ p)
            al = rz / (p * Ap).sum(0).clamp_min(1e-300)
            x += al * p; r -= al * Ap
            z = r / dgc
            rz2 = (r * z).sum(0); p = z + (rz2 / rz.clamp_min(1e-300)) * p; rz = rz2
        U.append(x.clone())
        e = (A @ x - b).norm(dim=1)
        rec.append((float(e.mean()) / L, float(e.max()) / L, k + 1))
    U = torch.stack(U)                                                 # [T+1,M,3]
    P = Xn.double()[None] + U
    met = R["metrics"]                                                 # t, rmse, cd, emd, detmin, inv
    curve = R["curve"]                                                 # t, it, l2, stp, nbt
    TR = D["traj"]
    out = []
    print("\n[표] t | RMSE% | 노드 이동 평균·최대 (%L, 직전 대비) | 정답 입자 이동 평균 %L | 모델 입자 이동 평균 %L | "
          "노드/정답 비 | 손실 처음 -> 끝 감소% | 보폭 중앙 | 되돌림 평균 | 셀 det 최소 | 복원 오차 평균·최대 %L", flush=True)
    for t in range(1, T + 1):
        dn = (P[t] - P[t - 1]).norm(dim=1)
        gt = float(np.linalg.norm(TR[t].astype(np.float64) - TR[t - 1], axis=1).mean())
        ym = Y[t - 1].astype(np.float64); yp = (Y[t - 2].astype(np.float64) if t > 1 else G0d.cpu().numpy())
        md = float(np.linalg.norm(ym - yp, axis=1).mean())
        cv = curve[curve[:, 0] == t]
        l0, l1 = cv[0, 2], cv[-1, 2]
        Xc = P[t][cells]
        X0 = Xn.double()[cells]
        J = torch.linalg.det((Xc[:, 1:] - Xc[:, :1]).transpose(1, 2)) / torch.linalg.det((X0[:, 1:] - X0[:, :1]).transpose(1, 2))
        row = (t, 100 * met[t - 1, 1], 100 * float(dn.mean()) / L, 100 * float(dn.max()) / L, 100 * gt / L,
               100 * md / L, float(dn.mean()) / max(gt, 1e-30), l0, l1, 100 * (l0 - l1) / max(l0, 1e-30),
               float(np.median(cv[:, 3])), float(cv[:, 4].mean()), float(J.min()), 100 * rec[t - 1][0], 100 * rec[t - 1][1])
        out.append(row)
        print(f"t={t:2d} | {row[1]:7.3f} | {row[2]:.5f} {row[3]:.4f} | {row[4]:.4f} | {row[5]:.5f} | {row[6]:.4f} | "
              f"{l0:.3e} -> {l1:.3e} {row[9]:7.3f}% | {row[10]:.2e} | {row[11]:.1f} | {row[12]:.4f} | "
              f"{row[13]:.1e} {row[14]:.1e} (CG {rec[t - 1][2]})", flush=True)
    sf, so = surface(cells)
    np.savez_compressed(a.nodes, nodes=P.float().cpu().numpy(), cells=cells.cpu().numpy(),
                        sface=sf.cpu().numpy(), sopp=so.cpu().numpy(), table=np.array(out), L=L, h=h,
                        cols=np.array(["t", "rmse%", "node_mean%L", "node_max%L", "gt_mean%L", "model_mean%L",
                                       "node/gt", "l2_first", "l2_last", "l2_drop%", "step_med", "nbt_mean",
                                       "cellJ_min", "rec_mean%L", "rec_max%L"]))
    se = torch.unique(torch.cat([sf[:, [0, 1]], sf[:, [1, 2]], sf[:, [0, 2]]]).sort(1).values, dim=0)
    print(f"\n[표면] 경계면 {sf.shape[0]}, 표면 모서리 {se.shape[0]}, 표면 꼭짓점 {torch.unique(sf).numel()}", flush=True)
    print(f"[저장] {a.nodes}", flush=True)
    sys.exit(0)

# ============================================================== render
sys.path.append(a.pg)
sys.path.append(os.path.join(a.pg, "gaussian-splatting"))
os.chdir(a.pg)
import json                                                          # noqa: E402
import imageio.v2 as imageio                                         # noqa: E402
from PIL import Image, ImageDraw, ImageFont                          # noqa: E402
from scene.gaussian_model import GaussianModel                       # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402
from diff_gaussian_rasterization import (                             # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer)

X = AX
model = str(X["model"])
gs = GaussianModel(3)
gs.load_ply(f"{model}/point_cloud/iteration_30000/point_cloud.ply")
gi = torch.as_tensor(X["gidx"], device=dev)                          # rep_render (--sub 없이) 와 같음
shs = gs.get_features[gi].detach()
op = gs.get_opacity[gi].detach()
lo = torch.as_tensor(X["lo"], device=dev, dtype=torch.float32)
s, so_ = float(X["s"]), float(X["scale_origin"])
off = torch.as_tensor(X["off"], device=dev, dtype=torch.float32)
mean = torch.as_tensor(X["mean"], device=dev, dtype=torch.float32)


def to_model(P):
    return ((P - off) * s + lo - 1.0) / so_ + mean


_sm = float(gs.get_scaling[gi].detach().median())
print(f"[등방] 축 길이 중앙값 {_sm:.5f} (모델 좌표) x {a.iso} -> 표준편차 {a.iso * _sm:.5f}", flush=True)

# 카메라: rep_render.py 와 같은 코드 (학습 카메라 a.cam)
cams = json.load(open(f"{model}/cameras.json"))
c = cams[a.cam]
Rw = np.array(c["rotation"]); pos = np.array(c["position"])
W2C = np.linalg.inv(np.block([[Rw, pos[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
Rm, Tm = W2C[:3, :3].T, W2C[:3, 3]
fovx = 2 * math.atan(c["width"] / (2 * c["fx"]))
fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
wv = torch.tensor(getWorld2View2(Rm, Tm)).transpose(0, 1).to(dev).float()
pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(dev).float()
fp = (wv.unsqueeze(0).bmm(pj.unsqueeze(0))).squeeze(0)
campos = wv.inverse()[3, :3]
Wd, Hd = int(c["width"]), int(c["height"])
print(f"[카메라] cameras.json[{a.cam}] {c.get('img_name', '')} {Wd}x{Hd} fx {c['fx']:.2f} fy {c['fy']:.2f} "
      f"위치 {np.round(pos, 4).tolist()}", flush=True)
st = GaussianRasterizationSettings(
    image_height=Hd, image_width=Wd,
    tanfovx=math.tan(fovx * 0.5), tanfovy=math.tan(fovy * 0.5),
    bg=torch.ones(3, device=dev), scale_modifier=1.0, viewmatrix=wv,
    projmatrix=fp, sh_degree=3, campos=campos, prefiltered=False, debug=False)
rast = GaussianRasterizer(raster_settings=st)


def render(Pn):
    P = to_model(Pn)
    cov = torch.zeros(P.shape[0], 6, device=dev)
    cov[:, 0] = cov[:, 3] = cov[:, 5] = (a.iso * _sm) ** 2
    img = rast(means3D=P, means2D=torch.zeros_like(P), shs=shs, colors_precomp=None, opacities=op,
               scales=None, rotations=None, cov3D_precomp=cov)[0]
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def project(Pn):
    """3DGS 래스터라이저와 같은 투영: p_hom = [x,1]·full_proj, 화소 = ((ndc+1)·W - 1)/2."""
    P = to_model(Pn)
    ph = torch.cat([P, torch.ones_like(P[:, :1])], 1) @ fp
    pw = 1.0 / (ph[:, 3:4] + 1e-7)
    nd = ph[:, :3] * pw
    px = ((nd[:, 0] + 1.0) * Wd - 1.0) * 0.5
    py = ((nd[:, 1] + 1.0) * Hd - 1.0) * 0.5
    return torch.stack([px, py], 1)


Z = np.load(a.nodes)
NODES = torch.as_tensor(Z["nodes"], device=dev)                      # [T+1,M,3] (정규화)
SF = torch.as_tensor(Z["sface"], device=dev).long()
SO = torch.as_tensor(Z["sopp"], device=dev).long()
R = np.load(a.res)
AY = np.concatenate([AX["G"][None].astype(np.float32), R["traj"].astype(np.float32)], 0)   # rep_render 와 같음
TR = D["traj"]
T1 = AY.shape[0]
assert TR.shape[0] == T1 == NODES.shape[0], (TR.shape, T1, NODES.shape)
font_big = ImageFont.truetype(a.font, 44)
font_mid = ImageFont.truetype(a.font, 28)
REF = None
if a.ref_video and os.path.exists(a.ref_video):
    REF = [f for f in imageio.get_reader(a.ref_video)]
wr = imageio.get_writer(a.video, fps=a.fps, codec="libx264", quality=8)
eye = campos
diffs, hits, nvis = [], [], []
with torch.no_grad():
    for t in range(T1):
        gt_img = render(torch.as_tensor(TR[t], device=dev, dtype=torch.float32))
        Pm = torch.as_tensor(AY[t], device=dev)
        tm_img = render(Pm)
        if REF is not None and t < len(REF):
            diffs.append(float(np.abs(REF[t].astype(np.float32) - tm_img.astype(np.float32)).mean()))
        # 카메라를 향한 경계면 (바깥 법선 · (눈 - 면 중심) > 0)
        Vm = to_model(NODES[t])
        p0, p1, p2 = Vm[SF[:, 0]], Vm[SF[:, 1]], Vm[SF[:, 2]]
        n = torch.cross(p1 - p0, p2 - p0, dim=-1)
        ctr = (p0 + p1 + p2) / 3
        n = torch.where(((Vm[SO] - ctr) * n).sum(1, keepdim=True) > 0, -n, n)   # 맞은편 꼭짓점 반대쪽 = 바깥
        vis = ((eye[None] - ctr) * n).sum(1) > 0
        fv = SF[vis]
        ed = torch.unique(torch.cat([fv[:, [0, 1]], fv[:, [1, 2]], fv[:, [0, 2]]]).sort(1).values, dim=0)
        vv = torch.unique(fv)
        nvis.append((int(vis.sum()), ed.shape[0], vv.numel()))
        uv = project(NODES[t]).cpu().numpy()
        # 투영 확인: 입자 투영이 렌더의 흰 배경이 아닌 곳에 떨어지는 비율
        pp = project(Pm).round().long().cpu().numpy()
        ok = (pp[:, 0] >= 0) & (pp[:, 0] < Wd) & (pp[:, 1] >= 0) & (pp[:, 1] < Hd)
        q = pp[ok][:: max(1, ok.sum() // 20000)]
        hits.append(float((tm_img[q[:, 1], q[:, 0]].min(1) < 240).mean()))
        im = Image.fromarray(tm_img).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        en = ed.cpu().numpy()
        for i, j in en:
            dr.line([tuple(uv[i]), tuple(uv[j])], fill=(0, 70, 230, 110), width=1)
        for k in vv.cpu().numpy():
            x_, y_ = uv[k]
            dr.ellipse([x_ - 1.3, y_ - 1.3, x_ + 1.3, y_ + 1.3], fill=(0, 30, 140, 200))
        right = Image.alpha_composite(im, ov).convert("RGB")
        left = Image.fromarray(gt_img)
        canvas = Image.new("RGB", (2 * Wd, Hd), (255, 255, 255))
        canvas.paste(left, (0, 0)); canvas.paste(right, (Wd, 0))
        d = ImageDraw.Draw(canvas)
        d.text((12, 6), f"프레임 {t}", fill=(220, 0, 0), font=font_big)
        d.text((Wd + 12, 6), f"프레임 {t}", fill=(220, 0, 0), font=font_big)
        d.text((12, Hd - 40), "정답 (PG 모래)", fill=(0, 0, 0), font=font_mid)
        d.text((Wd + 12, Hd - 40), "사면체 메쉬 (riem) + 꼭짓점·모서리", fill=(0, 0, 0), font=font_mid)
        if a.stall >= 0 and t >= a.stall:
            d.text((2 * Wd - 150, 6), "멈춤", fill=(220, 0, 0), font=font_big)
        fr = np.asarray(canvas)
        fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
        wr.append_data(fr)
        if t in (0, a.stall, T1 - 1):
            Image.fromarray(fr).save(a.video.replace(".mp4", f"_f{t:02d}.png"))
wr.close()
print(f"[확인] 기존 영상과 사면체 메쉬 칸 화소 차 평균 (0-255): "
      + (f"전체 평균 {np.mean(diffs):.2f}, 최대 {np.max(diffs):.2f}" if diffs else "비교 없음"), flush=True)
print(f"[확인] 입자 투영이 렌더 물체 위 (화소 <240) 비율: 최소 {min(hits):.3f}, 평균 {np.mean(hits):.3f}", flush=True)
print(f"[겹침] 카메라 쪽 경계면·모서리·꼭짓점 (프레임 0 / 끝): {nvis[0]} / {nvis[-1]}", flush=True)
print(f"[영상] {a.video}  {T1} 프레임  {2 * Wd}x{Hd}", flush=True)
