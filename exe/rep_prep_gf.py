"""repflow A: GaussianFluent (GF) 장면 정답 궤적 -> 추종 목표 (사용자 11-15).
GF 정답 = GF 시뮬 (CD-MPM) 입자 궤적 (R2 datasets/anchorflow/gf_scenes/gt/<장면>/simulation_ply/sim_*.h5, x [3,N]).
추적할 점 = 그 입자 전부 (부분표본 없음), 목표 = 프레임별 정답 위치. 정규화는 gauss_flow 와 같음 (프레임 0 의 bbox -> [0,1]^3 긴 축 1, 짧은 축 가운데; 이후 프레임은 같은 상수).
출력: gflow_<이름>.npz (X0, traj [T+1,N,3] float32, idx, lo, s, field 빈 것) + aux_<이름>.npz (G, lo, s, off, 나머지는 영상용이라 빈 값).
  python exe/rep_prep_gf.py --h5_dir DIR --name kiwi --out_dir OUT [--frames 0]"""
import argparse, glob, os
import h5py, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--frames", type=int, default=0, help="0 이면 있는 프레임 전부")
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.h5_dir, "sim_*.h5")))
if a.frames: fs = fs[:a.frames + 1]
def rd(f):
    with h5py.File(f, "r") as h: return np.asarray(h["x"], dtype=np.float64).T          # [N,3]
x0 = rd(fs[0]); N = x0.shape[0]
assert np.isfinite(x0).all(), "프레임 0 에 비유한 입자"
lo = x0.min(0); s = float((x0.max(0) - lo).max()); off = (1.0 - ((x0.max(0) - lo) / s)) / 2.0
tr = np.empty((len(fs), N, 3), dtype=np.float32); bad = 0
for t, f in enumerate(fs):
    x = rd(f); assert x.shape[0] == N, f"{f}: 입자 수 {x.shape[0]} != {N}"
    bad += int((~np.isfinite(x)).any(1).sum()); tr[t] = ((x - lo) / s + off).astype(np.float32)
os.makedirs(a.out_dir, exist_ok=True)
G = tr[0].copy()
np.savez(f"{a.out_dir}/gflow_{a.name}.npz", X0=G, traj=tr, idx=np.arange(N), lo=lo, s=s, field=np.zeros((0, 9)))
np.savez(f"{a.out_dir}/aux_{a.name}.npz", G=G, gidx=np.arange(N), lo=lo, s=s, off=off, scale_origin=1.0, mean=np.zeros(3), model="", opacity_threshold=0.0)
print(f"[gf] {a.name}: 입자 {N}  프레임 {len(fs) - 1}  비유한 값 {bad}  bbox 긴 축 {s:.4f}  이동 최대 {float(np.abs(tr[-1] - tr[0]).max()):.3f}", flush=True)
