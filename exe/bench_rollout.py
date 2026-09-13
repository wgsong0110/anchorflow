"""학생 롤아웃이 실시간으로 돌 수 있는 속도인지 잰다.

한 코어스 프레임에 필요한 것을 단계별로 나눠 잰다:
  1. 학생 순전파 -- chunk 개 스텝을 한 번에 내므로 프레임당 1/chunk 회
  2. 가우시안 위치 복호 -- N=203,930 개
  3. 렌더링용 회전·신축 -- 프레임 상태는 블렌드에서 곧바로 나오고,
     형상 매칭은 F 를 극분해해야 한다
  4. (참고) MPM 이 같은 한 프레임(40 서브스텝)에 걸리는 시간

속도는 학습 여부와 무관하므로 가중치는 아무거나 좋다 -- 구조와 앵커 수가 정한다.
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True)
ap.add_argument("--frame_state", action="store_true")
ap.add_argument("--ckpt", default=None, help="학생 체크포인트 (없으면 무작위 초기화)")
ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--frames", type=int, default=120)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--mpm", action="store_true", help="MPM 기준선도 잰다")
ap.add_argument("--render", action="store_true", help="래스터화까지 포함해 잰다")
ap.add_argument("--width", type=int, default=800); ap.add_argument("--height", type=int, default=800)
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import load_fitted
from anchorflow.nextstate import NextStep, apply_step, apply_step_frame
from anchorflow.frame_encode import FrameState
from anchorflow.anchor_fit import closest_rotation

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare()
w, rc = cache[0], cache[1]
Yg = fit.Xc - rc
M, N = fit.M, fit.N
FS = FrameState(fit.pair_g, fit.pair_a, N, M, sc.volume[sc.keep].clone()) if args.frame_state else None
dt_c = args.dt_mult * sc.sub_dt
print(f"[setup] 앵커 {M}, 물질 가우시안 {N}, 전체 가우시안 {sc.pos.shape[0]}, "
      f"코어스 dt {dt_c:.4g}s ({args.dt_mult} 서브스텝), chunk {args.chunk}", flush=True)

net = NextStep(args.hidden, args.depth, args.heads, 1.0, 1.0, 1.0,
               use_accel=False, chunk=args.chunk, frame=args.frame_state,
               u_scale=1.0, s_scale=1.0, du_scale=1.0, ds_scale=1.0).to(dev).eval()
if args.ckpt and os.path.exists(args.ckpt):
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    try:
        net.load_state_dict(ck["model"]); print(f"[setup] 가중치 {args.ckpt}")
    except Exception as e:
        print(f"[setup] 가중치 로드 실패({e}) -- 구조만으로 잰다")
print(f"[setup] 파라미터 {sum(q.numel() for q in net.parameters())/1e3:.1f}k", flush=True)

p = fit.pos.clone(); v = torch.zeros_like(p)
u = torch.zeros(M, 3, device=dev); s = torch.zeros(M, 3, device=dev)
fixed = fit.fixed


def sync(): torch.cuda.synchronize()


def bench(fn, n):
    for _ in range(args.warmup): fn()
    sync(); t0 = time.time()
    for _ in range(n): fn()
    sync()
    return (time.time() - t0) / n * 1e3          # ms


# 1) 학생 순전파 (한 번에 chunk 스텝)
def step_call():
    if args.frame_state:
        apply_step_frame(net, p, v, u, s, None, dt_c, fixed)
    else:
        apply_step(net, p, v, None, dt_c, fixed)


ms_net = bench(step_call, args.frames) / args.chunk

# 2) 위치 복호
if args.frame_state:
    def dec_pos():
        FS.decode_x(p, u, s, w, Yg)
else:
    def dec_pos():
        fit.gaussian_pos(p, cache)
ms_pos = bench(dec_pos, max(20, args.frames // 4))

# 3) 렌더링용 회전·신축
if args.frame_state:
    def dec_rs():
        FS.blend(u, s, w)          # (u_g, s_g) -> 그대로 3DGS 의 (회전, 스케일)
else:
    def dec_rs():
        F, _ = fit.deformation(p, w, rc, cache[3], cache[3], cache[4])
        closest_rotation(F, 8, 1e-6)
ms_rs = bench(dec_rs, max(10, args.frames // 8))

tot = ms_net + ms_pos + ms_rs
print(f"\n{'단계':<28}{'ms/프레임':>12}")
print(f"{'1. 학생 순전파 (/chunk)':<28}{ms_net:12.2f}")
print(f"{'2. 가우시안 위치 복호':<28}{ms_pos:12.2f}")
print(f"{'3. 렌더용 회전·신축':<28}{ms_rs:12.2f}")
print(f"{'합계 (렌더 제외)':<28}{tot:12.2f}   -> {1000/tot:.1f} fps")
print(f"실시간(30fps=33.3ms) 대비: {tot/33.3:.2f}x", flush=True)

# 4) 래스터화 -- 전체 가우시안(비물질 포함)을 한 장 그린다
ms_ren = 0.0
if args.render:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    import math
    NG = sc.pos.shape[0]
    fov = 0.8
    tanx = tany = math.tan(fov * 0.5)
    Rt = torch.eye(4, device=dev); Rt[2, 3] = 4.0
    proj = torch.zeros(4, 4, device=dev)
    proj[0, 0] = 1 / tanx; proj[1, 1] = 1 / tany
    proj[2, 2] = proj[3, 2] = 1.0; proj[2, 3] = -0.01
    full = (Rt.t() @ proj)
    rs = GaussianRasterizationSettings(
        image_height=args.height, image_width=args.width, tanfovx=tanx, tanfovy=tany,
        bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=Rt.t(),
        projmatrix=full, sh_degree=0, campos=torch.zeros(3, device=dev),
        prefiltered=False, debug=False)
    raster = GaussianRasterizer(raster_settings=rs)
    means3D = sc.pos.contiguous()
    means2D = torch.zeros_like(means3D)
    shs = torch.zeros(NG, 1, 3, device=dev)
    opac = torch.full((NG, 1), 0.8, device=dev)
    scl = torch.full((NG, 3), 0.005, device=dev)
    rot = torch.zeros(NG, 4, device=dev); rot[:, 0] = 1.0

    def render_once():
        raster(means3D=means3D, means2D=means2D, shs=shs, colors_precomp=None,
               opacities=opac, scales=scl, rotations=rot, cov3D_precomp=None)
    ms_ren = bench(render_once, max(20, args.frames // 4))
    print(f"{'4. 래스터화 (' + str(args.width) + 'x' + str(args.height) + ', ' + str(NG) + '개)':<28}"
          f"{ms_ren:12.2f}")
    tot2 = tot + ms_ren
    print(f"{'합계 (렌더 포함)':<28}{tot2:12.2f}   -> {1000/tot2:.1f} fps")
    print(f"실시간 대비: {tot2/33.3:.2f}x", flush=True)

if args.mpm:
    from anchorflow.mpm_teacher import MPMTeacher
    T = MPMTeacher(sc)
    T._set(T.pos_m.clone(), torch.zeros_like(T.pos_m), T.eye.clone(), torch.zeros_like(T.eye))
    def mpm_frame():
        for _ in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
    ms_mpm = bench(mpm_frame, 20)
    # MPM 도 렌더링하려면 F 를 극분해해 (회전, 스케일) 을 만들어야 한다 -- 학생의
    # 3 단계와 같은 일이다. 위치는 이미 입자로 나와 있으므로 복호 비용이 없다.
    def mpm_rs():
        F = T.solver.export_particle_F_to_torch().reshape(-1, 3, 3)
        closest_rotation(F, 8, 1e-6)
    ms_mrs = bench(mpm_rs, 10)
    mtot = ms_mpm + ms_mrs + ms_ren
    print(f"\n[MPM 기준선]")
    print(f"{'  40 서브스텝':<28}{ms_mpm:12.2f}")
    print(f"{'  렌더용 회전·신축(극분해)':<28}{ms_mrs:12.2f}")
    if args.render:
        print(f"{'  래스터화':<28}{ms_ren:12.2f}")
    print(f"{'  합계':<28}{mtot:12.2f}   -> {1000/mtot:.1f} fps")
    print(f"  학생 대비: {mtot/(tot+ms_ren):.1f}x 느림")
print("\nBENCH_DONE")
