"""정답 MPM (PG 공식 warp 솔버, lb_gen.py 와 같은 경로) 의 민감도 -- lego L000 을 조건 하나씩 바꿔 다시 굴려 정답과의 차를 잰다 (doc/fem_realtime.md 절 N.1).

정답 L000 = lb_gen.py (i-physgaussian MPM_Simulator_WARP, jelly = 고정 FCR, APIC rpic_damping 0, n_grid 100 · grid_lim 2 (dx 0.02),
substep_dt 2.0008e-5 (833/프레임), 바닥 surface_collider sticky (격자 노드 z < 0.48 -> 격자 속도 0), bounding_box, 감쇠 없음 (scale 1.1)).
초기 상태는 정답 0 프레임 h5 (x, v) 와 같은 값, 입자 부피 = dx0³ / (dx0 = 0.02 셀의 입자 수) -- **격자를 바꿔도 질량·부피는 정답 그대로** (격자 효과만 분리).
오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 80 프레임 평균 (fem_ex_sim 과 같은 지표, EXT = L000.pt _ext).

  cd i-physgaussian && python <anchorflow>/exe/gt_sens.py --out DIR [--nsub 833] [--rpic 0|-1] [--n_grid 100] [--floor_z 0.48] [--surface sticky|slip]
출력 DIR/: x.pt (float32 [80, N, 3]), err.json (프레임별 오차, 요약)
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--gt", default="/home/dkta/work/lbdata/test_low/L000", help="정답 h5 폴더 (sim_XXXXXXXXXX.h5, meta.json)")
ap.add_argument("--pt", default="/home/dkta/work/lbpt/test_low/L000.pt", help="EXT (_ext) 를 읽을 곳")
ap.add_argument("--config", default="/home/dkta/work/lbdata/config.json")
ap.add_argument("--pg", default="/home/dkta/work/i-physgaussian")
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--nsub", type=int, default=0, help="프레임당 서브스텝 (0 = config 그대로 833)")
ap.add_argument("--rpic", type=float, default=0.0, help="PG rpic_damping: 0 = APIC (정답), -1 = PIC")
ap.add_argument("--n_grid", type=int, default=0, help="0 = config 그대로 (100)")
ap.add_argument("--floor_z", type=float, default=None, help="바닥 평면 높이 (정답 0.48)")
ap.add_argument("--surface", default=None, help="바닥 종류 (정답 sticky)")
ap.add_argument("--no_dump", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.pg)
os.chdir(a.pg)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import warp as wp                                                # noqa: E402
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP   # noqa: E402

cfg = json.load(open(a.config))
meta = json.load(open(os.path.join(a.gt, "meta.json")))
GL = float(cfg["grid_lim"])
n_grid0 = int(cfg["n_grid"])
dx0 = GL / n_grid0
n_grid = a.n_grid or n_grid0
frame_dt = float(cfg["frame_dt"])
nsub = a.nsub or int(round(frame_dt / float(cfg["substep_dt"])))
dt = frame_dt / nsub
G = [float(q) for q in cfg["g"]]
floor = dict([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0])
if a.floor_z is not None:
    floor["point"] = [floor["point"][0], floor["point"][1], float(a.floor_z)]
if a.surface is not None:
    floor["surface"] = a.surface
d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
EXT = float(d["_ext"])
del d


def h5x(t, key="x"):
    with h5py.File(os.path.join(a.gt, f"sim_{t:010d}.h5")) as h:
        return np.array(h[key])


x0 = h5x(0).astype(np.float32)
v0 = h5x(0, "v").astype(np.float32)
N = x0.shape[0]
L = min(a.frames, int(meta["frames"]))
X = torch.as_tensor(x0).cuda().contiguous()
cell = (X / dx0).floor().long()
key = (cell[:, 0] * n_grid0 + cell[:, 1]) * n_grid0 + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx0 ** 3 / cnt[inv].float()).contiguous()
print(f"[정답 민감도] 입자 {N}, n_grid {n_grid} (dx {GL / n_grid:.5f}), 서브스텝 {dt:.4e} x {nsub}, rpic {a.rpic}, 바닥 {floor}, EXT {EXT:.6f}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)
wp.init()
sol = MPM_Simulator_WARP(10)
sol.load_initial_data_from_torch(X, VOL, None, n_grid=n_grid, grid_lim=GL)
sol.set_parameters_dict({"material": "jelly", "E": float(cfg["E"]), "nu": float(cfg["nu"]), "density": float(cfg["density"]),
                         "g": G, "n_grid": n_grid, "grid_lim": GL, "grid_v_damping_scale": 1.1, "rpic_damping": float(a.rpic)})
sol.add_bounding_box()
sol.add_surface_collider(tuple(floor["point"]), tuple(floor["normal"]), floor["surface"], float(floor.get("friction", 0.0)))
sol.finalize_mu_lam()
sol.import_particle_v_from_torch(torch.as_tensor(v0).cuda().contiguous())
os.makedirs(a.out, exist_ok=True)
XD = None if a.no_dump else torch.empty(L, N, 3, dtype=torch.float32)
errs = []
t0 = time.time()
step = 0
for f in range(1, L + 1):
    for s in range(nsub):
        sol.p2g2p(step, dt)
        step += 1
    xt = sol.export_particle_x_to_torch()
    gt = torch.as_tensor(h5x(f)).cuda()
    e = float((xt - gt).norm(dim=-1).mean()) / EXT
    if not torch.isfinite(xt).all():
        e = float("nan")
    errs.append(e)
    if XD is not None:
        XD[f - 1] = xt.cpu()
    print(f"  프레임 {f}/{L} 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) 최저 {float(xt[:, 2].min()):.5f} {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(e):
        print(f"[발산] 프레임 {f}", flush=True)
        break
summ = dict(args=vars(a), nsub=nsub, dt=dt, n_grid=n_grid, floor=floor, EXT=EXT, frames_done=len(errs), err_mean=float(np.mean(errs)),
            err=errs, sec=time.time() - t0)
json.dump(summ, open(os.path.join(a.out, "err.json"), "w"), ensure_ascii=False, indent=1)
if XD is not None:
    torch.save(XD, os.path.join(a.out, "x.pt"))
print(f"[정답 민감도 결과] {os.path.basename(a.out)}: 평균 오차 {100 * summ['err_mean']:.3f}% ({len(errs)}/{L} 프레임), {summ['sec']:.0f}s", flush=True)
