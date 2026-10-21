"""manytraj (사용자 10-21): 정답 MPM 없이 출력만 최적화 롤아웃만 만들 새 lego 궤적의 **초기 상태 .pt** 를 만든다.

초기 조건 분포 = 풀 학습(train_deform --pool_drop h0,h1,v0,v1)의 낙하 모드 그대로 -- lib/anchorflow/scene_pool.StatePool._drop_init
을 **그 함수 그대로** 부른다 (균일 회전(정규분포 사원수), xy 가운데, 바닥 위 여유 U[h0,h1], 강체 병진 속도 크기 U[v0,v1]·방향 구 균일,
상자 밖·최고점 초과면 다시 뽑음). 기준 형상 = 풀의 채우기 lbpool/fill/pgfill_lego.npy (lbdata 의 init h5 와 같은지 대조해 기록),
cfg = 기존 정답 .pt(L000) 의 cfg 그대로 (n_grid·grid_lim·바닥·g 가 거기서 온다). 궤적마다 시드 고정 (CPU 생성기).

.pt 형식 = exe/lb_to_pt.py 와 같은 키 (x, v, F half, sel, cfg, n_full, meta, gaussians_first). 정답 프레임이 없으므로 x·v·F 는
프레임 0 하나를 frames 개로 **expand** 해 둔다 (저장은 한 프레임 분량; 롤아웃의 오차 줄은 정지 기준이라 의미 없음 -- 학습은 덤프만 쓴다).

  python exe/manytraj_init.py --ref /home/dkta/work/lbpt/test_low/L000.pt --fill /home/dkta/work/lbpool/fill/pgfill_lego.npy \
      --drop 0.0,0.02,0.0,1.5 --n 95 --seed0 7001 --out /home/dkta/work/lbours/manytraj/one
"""
import argparse
import json
import math
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow.scene_pool import StatePool, load_scenes      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="cfg·입자 수를 가져올 기존 정답 .pt (lbpt/test_low/L000.pt)")
ap.add_argument("--fill", required=True, help="풀의 채우기 디렉토리 (--pool_fill, lbpool/fill)")
ap.add_argument("--pool_cfg", required=True, help="풀의 cfg 디렉토리 (--pool_cfg, lbpool/cfg)")
ap.add_argument("--combo", default="lego_jelly", help="--pool_combos")
ap.add_argument("--drop", default="0.0,0.02,0.0,1.5", help="--pool_drop 과 같은 h0,h1,v0,v1")
ap.add_argument("--n", type=int, default=95)
ap.add_argument("--seed0", type=int, default=7001, help="궤적 k (1..n) 의 시드 = seed0 + k - 1")
ap.add_argument("--frames", type=int, default=81, help="정답 .pt 와 같은 프레임 수 (0..80)")
ap.add_argument("--prefix", default="N")
ap.add_argument("--out", required=True, help="OUT/<태그>/<태그>.pt 로 쓴다 (train_deform --data OUT/<태그>)")
a = ap.parse_args()

ref = torch.load(a.ref, map_location="cpu", weights_only=False, mmap=True)
cfg = ref["cfg"]
N = int(ref["x"].shape[1])
# 풀과 같은 경로로 씬을 읽는다 (train_deform: load_scenes(pool_fill, pool_cfg, combos, 0, dev, seed))
_sc = load_scenes(a.fill, a.pool_cfg, [a.combo], 0, "cpu", seed=0)
x_fill = _sc[0][1]["x0"].float()
assert x_fill.shape == (N, 3), (x_fill.shape, N)
_pc = _sc[0][1]["cfg"]
_diffk = sorted(k for k in set(_pc) | set(cfg) if _pc.get(k) != cfg.get(k))
print(f"[cfg 대조] 풀 cfg vs 정답 .pt cfg 다른 키: {_diffk or '없음'}", flush=True)
if _diffk:
    raise SystemExit(f"[오류] 풀 cfg 와 정답 cfg 가 다르다: {_diffk}")
# 기준 형상 대조: lbdata 의 init h5 (정답 궤적들이 회전시킨 원래 형상)
h5p = (ref.get("meta") or {}).get("h5")
cmp = {}
if h5p and os.path.exists(h5p):
    import h5py
    with h5py.File(h5p) as h:
        xh = np.array(h["x"]); xh = xh.T if xh.shape[0] == 3 else xh
    xh = torch.as_tensor(xh).float()
    cmp = dict(h5=h5p, max_abs_diff=float((xh - x_fill).abs().max()))
    print(f"[기준 형상] 풀 채우기 vs lbdata init h5 ({h5p}) 최대 차 {cmp['max_abs_diff']:.3e}", flush=True)
    if cmp["max_abs_diff"] > 1e-5:
        raise SystemExit(f"[오류] 풀 채우기와 lbdata init h5 의 입자가 다르다 (최대 차 {cmp['max_abs_diff']:.3e}) -- 멈춤")
h0, h1, v0, v1 = (float(q) for q in a.drop.split(","))
GL = float(cfg.get("grid_lim", 2.0))
fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]
# train_deform 의 _DROP 와 같은 사전 (margin = 3·grid_lim/n_grid, zf = 바닥 점 z, gz = g_z)
drop = dict(h=(h0, h1), v=(v0, v1), zf=float(fl["point"][2]), gz=float(cfg["g"][2]), margin=3 * GL / int(cfg["n_grid"]))
print(f"[분포] 높이 여유 U{drop['h']}, 속도 크기 U{drop['v']}, 바닥 z {drop['zf']}, 여유 {drop['margin']:.3f}, 상자 {GL}", flush=True)
os.makedirs(a.out, exist_ok=True)
summ = []
for k in range(1, a.n + 1):
    tag = f"{a.prefix}{k:03d}"
    seed = a.seed0 + k - 1
    dst = f"{a.out}/{tag}/{tag}.pt"
    # StatePool._drop_init 을 그대로 부른다: self 에 필요한 것은 drop·dev·domain·gen 뿐
    me = types.SimpleNamespace(drop=drop, dev="cpu", domain=GL, gen=torch.Generator().manual_seed(seed))
    xr, vr = StatePool._drop_init(me, x_fill.clone())
    # 기록용 (분포 확인): 회전 R (프로크루스테스로 복원), 높이 여유, 속도
    c0 = x_fill.mean(0).double(); A = (x_fill.double() - c0); B = (xr.double() - xr.double().mean(0))
    U_, _, Vt_ = torch.linalg.svd(A.T @ B)
    D_ = torch.eye(3, dtype=torch.float64); D_[2, 2] = torch.sign(torch.linalg.det(U_ @ Vt_))
    R = (U_ @ D_ @ Vt_).T                                       # B ≈ A Rᵀ
    rres = float((A @ R.T - B).abs().max())
    lift = float(xr[:, 2].min()) - drop["zf"]
    vv = vr[0].double(); spd = float(vv.norm())
    rec = dict(tag=tag, seed=seed, lift=lift, speed=spd, dir=(vv / max(spd, 1e-30)).tolist(), v0=vv.tolist(), R=R.tolist(),
               R_fit_resid=rres, lo=xr.min(0).values.tolist(), hi=xr.max(0).values.tolist(),
               rigid_v=bool((vr - vr[0]).abs().max() == 0), range=dict(h=[h0, h1], v=[v0, v1]))
    summ.append(rec)
    if os.path.exists(dst):
        print(f"[있음] {dst}", flush=True)
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    X = xr.half().unsqueeze(0).expand(a.frames, N, 3)
    V = vr.half().unsqueeze(0).expand(a.frames, N, 3)
    F = torch.eye(3).half().view(1, 1, 3, 3).expand(a.frames, N, 3, 3)
    meta = dict(rec, frames=a.frames - 1, N=N, config_from=a.ref, fill=a.fill, fill_vs_h5=cmp, gt="없음 (초기 상태만, x·v·F 는 프레임 0 을 expand)",
                gaussians_first=int(ref.get("gaussians_first", N)), h5=h5p, frame_dt=float(cfg["frame_dt"]))
    torch.save({"x": X, "v": V, "F": F, "sel": torch.arange(N), "cfg": cfg, "n_full": N, "meta": meta,
                "gaussians_first": int(ref.get("gaussians_first", N))}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print(f"[저장] {tag} 시드 {seed}  높이 여유 {lift:.4f}  속도 {spd:.3f} 방향 {np.round(rec['dir'], 3).tolist()}  "
          f"R 잔차 {rres:.1e}  {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
json.dump(dict(drop=drop, fill=a.fill, fill_vs_h5=cmp, ref=a.ref, seed0=a.seed0, trajs=summ), open(f"{a.out}/ics.json", "w"), indent=1)
L = np.array([r["lift"] for r in summ]); S = np.array([r["speed"] for r in summ])
print(f"[분포 요약] {len(summ)} 궤적  높이 여유 최소 {L.min():.4f} 최대 {L.max():.4f} 평균 {L.mean():.4f}  "
      f"속도 최소 {S.min():.3f} 최대 {S.max():.3f} 평균 {S.mean():.3f}  강체 속도 {all(r['rigid_v'] for r in summ)}", flush=True)
ok = (L.min() >= h0 - 1e-5) and (L.max() <= h1 + 1e-5) and (S.min() >= v0 - 1e-5) and (S.max() <= v1 + 1e-5)
print("MANYTRAJ_INIT_OK" if ok else "MANYTRAJ_INIT_BAD", flush=True)
