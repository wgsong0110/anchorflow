"""학습 비교 공통 데이터셋(lbdata) 궤적을 PGND(RSS 2025) 공식 학습 입력 `sub_episodes_v/episode_XXXX/` 으로.

PGND 의 공식 사슬(experiments/real_world/postprocess.py)은 RGB-D -> 추적 -> pcd_clean(점이 10,000 개를 넘으면
무작위 10,000 개, L827-829) -> get_sub_episodes(traj.npz 등) 이다. 우리는 MPM 입자 트랙이 이미 있으므로
마지막 산출물만 같은 형식으로 쓴다.

  episode_XXXX/traj.npz   xyz (81, 10000, 3) float32, v (81, 10000, 3) float32  -- 60 fps 원본 81 프레임 전부
                          (학습은 train.dataset_load_skip_frame=6 으로 0,6,..,78 을 읽어 0.1 s 스텝)
  episode_XXXX/sel.npy    10,000 입자의 원본 인덱스 (313,750 중, 궤적마다 시드 고정 무작위 = 공식 10k 상한)
  episode_XXXX/eef_traj.txt     (81, 3) 가짜 그리퍼 -- 영역 밖 아주 먼 곳 (변환 뒤 y = -100*scale 근처)
  episode_XXXX/eef_gripper.txt  (81, 1) 1.0 = 열림 (dataset.py: 1 open, 0 close)
  episode_XXXX/meta.txt   [k, 0, 81]
  metadata.json           [{"path": 원본 궤적 디렉토리, "split": train|test, "name": NNNN}] x 120 (index = episode 번호)

축: PGND preprocess 가 R = [[1,0,0],[0,0,-1],[0,1,0]] (x->x, y->-z, z->y) 을 곱하므로, 우리 z-up 을 y-up 으로
만들려고 파일에는 (x, -y, -z) 로 쓴다 (x 축 180도 회전, 고유 회전). R 뒤에는 (x, z, -y) -> 중력이 -y.

확인 (summary.json):
  * 벽 접촉: 어느 프레임이든 입자(전체 313,750) x 또는 y 가 [3dx, GL-3dx] 밖이면 그 궤적을 센다
  * 배율: PGND preprocess 와 같은 식(궤적별 가운데 맞추기, 최저점 -> 0.04)으로 sim.preprocess_scale 에서
    10k 부분집합과 전체 입자 모두 [0.04, 0.96] 안인지, 허용 최대 배율

  python exe/lb_to_pgnd.py --src /home/dkta/work/lbdata --out /home/dkta/work/lb/pgnd_data/lego_lb/sub_episodes_v --scale 0.45
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="lbdata 루트 (train/, test/ 아래 NNNN/)")
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=0.45, help="PGND sim.preprocess_scale (확인용)")
ap.add_argument("--n_pts", type=int, default=10000, help="공식 pcd_clean 상한 (postprocess.py L827)")
ap.add_argument("--only", type=int, default=-1, help="이 episode 번호 하나만 (검증용)")
ap.add_argument("--workers", type=int, default=12)
a = ap.parse_args()

DXP = 0.02 * (1.5 + 0.5)          # PGND 영역 여유 dx*(clip_bound+0.5), dx = num_grids[-1] = 0.02 (sloth 기본)
FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float32)
R_PGND = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def pgnd_bounds(xyz, s):
    """dataset.py preprocess(preprocess_with_table=True) 와 같은 변환 뒤 (min, max) 와 통과 여부."""
    p = np.einsum("nij,jk->nik", xyz.astype(np.float64), R_PGND.T) * s
    t = np.array([0.5 - (p[:, :, 0].max() + p[:, :, 0].min()) / 2,
                  DXP + 1e-5 - p[:, :, 1].min(),
                  0.5 - (p[:, :, 2].max() + p[:, :, 2].min()) / 2])
    lo = (p.min((0, 1)) + t); hi = (p.max((0, 1)) + t)
    ok = bool((lo >= DXP - 1e-6).all() and (hi <= 1 - DXP + 1e-6).all())
    ext = p.max((0, 1)) - p.min((0, 1))
    smax = min((1 - 2 * DXP) / (ext[0] / s), (1 - 2 * DXP) / (ext[2] / s), (1 - 2 * DXP - 1e-5) / (ext[1] / s))
    return lo.tolist(), hi.tolist(), ok, smax


def job(item):
    k, split, name = item
    d = f"{a.src}/{split}/{name}"
    meta = json.load(open(f"{d}/meta.json"))
    cfg = json.load(open(meta["config"]))
    GL = float(cfg["grid_lim"]); dx = GL / int(cfg["n_grid"])
    fs = sorted(glob.glob(f"{d}/sim_*.h5"))
    assert len(fs) == 81, (d, len(fs))
    X, V = [], []
    for f in fs:
        with h5py.File(f) as h:
            X.append(np.asarray(h["x"], dtype=np.float32)); V.append(np.asarray(h["v"], dtype=np.float32))
    X = np.stack(X); V = np.stack(V)                          # (81, N, 3) 우리 좌표
    N = X.shape[1]
    assert np.isfinite(X).all(), d
    # 벽 접촉 (상자 측벽, x·y)
    lo_xy = X[:, :, :2].min((0, 1)); hi_xy = X[:, :, :2].max((0, 1))
    wall = bool((lo_xy < 3 * dx).any() or (hi_xy > GL - 3 * dx).any())
    rng = np.random.default_rng(k)
    sel = np.sort(rng.choice(N, a.n_pts, replace=False))
    xs = X[:, sel] * FLIP; vs = V[:, sel] * FLIP
    lo_s, hi_s, ok_s, smax_s = pgnd_bounds(xs, a.scale)
    lo_f, hi_f, ok_f, smax_f = pgnd_bounds(X[:, ::4] * FLIP, a.scale)   # 전체 입자 (4 개 중 하나 프레임 -- 범위 확인용)
    od = f"{a.out}/episode_{k:04d}"
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(f"{od}/traj.npz", xyz=xs.astype(np.float32), v=vs.astype(np.float32))
    np.save(f"{od}/sel.npy", sel.astype(np.int64))
    far = np.tile(np.array([[0.0, 0.0, 100.0]]), (81, 1))     # 파일 좌표 z=+100 -> R 뒤 y=-100 (영역 한참 밖)
    np.savetxt(f"{od}/eef_traj.txt", far)
    np.savetxt(f"{od}/eef_gripper.txt", np.ones((81, 1)))
    np.savetxt(f"{od}/meta.txt", np.array([k, 0, 81]))
    r = dict(k=k, split=split, name=name, N=int(N), wall=wall, lo_xy=lo_xy.tolist(), hi_xy=hi_xy.tolist(),
             GL=GL, dx=dx, sub_ok=ok_s, sub_lo=lo_s, sub_hi=hi_s, sub_smax=smax_s,
             full_ok=ok_f, full_lo=lo_f, full_hi=hi_f, full_smax=smax_f,
             zmin=float(X[:, :, 2].min()), speed=meta.get("speed"))
    print(f"[{k:04d}] {split}/{name} N={N} wall={wall} scale {a.scale}: sub_ok={ok_s} full_ok={ok_f} "
          f"smax sub {smax_s:.3f} full {smax_f:.3f}", flush=True)
    return r


items = [(i, "train", f"{i:04d}") for i in range(100)] + [(100 + i, "test", f"{i:04d}") for i in range(20)]
if a.only >= 0:
    items = [it for it in items if it[0] == a.only]
os.makedirs(a.out, exist_ok=True)
with Pool(a.workers) as pool:
    res = pool.map(job, items)
md = [dict(path=f"{a.src}/{s}/{n}", split=s, name=n) for _, s, n in
      [(i, "train", f"{i:04d}") for i in range(100)] + [(100 + i, "test", f"{i:04d}") for i in range(20)]]
json.dump(md, open(f"{a.out}/metadata.json", "w"), indent=1)
summ = dict(scale=a.scale, n=len(res), n_wall=sum(r["wall"] for r in res),
            wall_eps=[r["k"] for r in res if r["wall"]],
            n_sub_out=sum(not r["sub_ok"] for r in res), n_full_out=sum(not r["full_ok"] for r in res),
            smax_sub=min(r["sub_smax"] for r in res), smax_full=min(r["full_smax"] for r in res), per_episode=res)
tag = f"_only{a.only}" if a.only >= 0 else ""
json.dump(summ, open(f"{a.out}/../convert_summary{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "per_episode"}), flush=True)
