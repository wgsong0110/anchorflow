"""manytraj: 새 궤적 출력만 최적화 덤프 확인 (사용자 10-21: 대량 생성 전 한 개 확인).

프레임별: 입자 평균 이동량 |x_t - x_{t-1}|, 무게중심 z, 바닥 위 최저 높이, 바닥 접촉 입자 수 (높이 < --touch), det F <= 0 입자 수.
초기 조건: 높이 여유·속도 크기가 분포 범위 안인지, 속도가 강체 병진인지.
판정 (PASS): 유한, 뒤집힘(det F <= 0) 0, 바닥에 닿음(최저 높이 < --touch 인 프레임 있음), 처음 10 프레임 평균 이동량 > --move_min, 초기 조건 범위 안.

  python exe/manytraj_check.py --dump roll_ov_N001.pt --init one/N001/N001.pt --out check_N001.json
"""
import argparse
import json

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--touch", type=float, default=0.01, help="바닥 접촉 판정 높이 (stick_diag 와 같은 0.01)")
ap.add_argument("--move_min", type=float, default=1e-4, help="처음 10 프레임 평균 이동량 하한 (m/프레임)")
a = ap.parse_args()

ini = torch.load(a.init, map_location="cpu", weights_only=False)
cfg, meta = ini["cfg"], ini["meta"]
zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
D = torch.load(a.dump, map_location="cpu", weights_only=False)
P = D["pred"].float()                                    # [L,N,3] 프레임 1..L
x0 = ini["x"][0].float()
X = torch.cat([x0.unsqueeze(0), P], 0)                   # [L+1,N,3]
L = P.shape[0]
fin = bool(torch.isfinite(P).all())
mv = (X[1:] - X[:-1]).norm(dim=-1).mean(1)               # [L]
comz = X[:, :, 2].mean(1)
hmin = X[:, :, 2].min(1).values - zf
ncont = (X[:, :, 2] - zf < a.touch).sum(1)
Fk = None
for k in ("F", "Fj", "F32"):
    if k in D and torch.is_tensor(D[k]):
        Fk = k
        break
if Fk is None:
    raise SystemExit(f"덤프에 F 가 없다: 키 {list(D.keys())}")
Fd = D[Fk]
ndet = []
for t in range(Fd.shape[0]):
    ft = Fd[t].float().reshape(-1, 3, 3)
    ndet.append(int((torch.linalg.det(ft.double()) <= 0).sum()))
h0, h1 = meta["range"]["h"]; v0, v1 = meta["range"]["v"]
ic_ok = (h0 - 1e-5 <= meta["lift"] <= h1 + 1e-5) and (v0 - 1e-5 <= meta["speed"] <= v1 + 1e-5) and bool(meta["rigid_v"])
v_ic = ini["v"][0].float()
lift_meas = float(x0[:, 2].min()) - zf
touched = bool((hmin[1:] < a.touch).any())
t_touch = int(torch.nonzero(hmin[1:] < a.touch)[0][0]) + 1 if touched else None
moves = float(mv[:10].mean()) > a.move_min
ok = fin and sum(ndet) == 0 and touched and moves and ic_ok
res = dict(dump=a.dump, init=a.init, frames=L, finite=fin, det_le0_total=sum(ndet), det_le0_per_frame=ndet,
           mean_move_per_frame=[float(q) for q in mv], com_z=[float(q) for q in comz], min_height=[float(q) for q in hmin],
           contact_count=[int(q) for q in ncont], touched=touched, first_touch_frame=t_touch,
           ic=dict(seed=meta["seed"], lift=meta["lift"], lift_measured=lift_meas, speed=meta["speed"],
                   speed_measured=float(v_ic.norm(dim=-1).mean()), dir=meta["dir"], R=meta["R"], range=meta["range"],
                   rigid_v=meta["rigid_v"], ok=ic_ok),
           com_drop=float(comz[0] - comz.min()), total_com_path=float((X[1:].mean(1) - X[:-1].mean(1)).norm(dim=-1).sum()),
           PASS=ok)
json.dump(res, open(a.out, "w"), indent=1)
pf = lambda arr, f="{:.4f}": " ".join(f.format(q) for q in arr)       # noqa: E731
print(f"[확인] {a.dump}", flush=True)
print(f"  초기 조건: 시드 {meta['seed']}  높이 여유 {meta['lift']:.4f} (잰 값 {lift_meas:.4f}, 범위 {h0}~{h1})  속도 {meta['speed']:.3f} "
      f"(범위 {v0}~{v1}) 방향 {np.round(meta['dir'], 3).tolist()}  강체 속도 {meta['rigid_v']}  -> {'범위 안' if ic_ok else '범위 밖'}", flush=True)
print(f"  유한 {fin}  뒤집힘(det F<=0) 합 {sum(ndet)}  첫 바닥 접촉 프레임 {t_touch}  무게중심 낙하 {res['com_drop']:.4f} m  "
      f"무게중심 경로 {res['total_com_path']:.4f} m", flush=True)
print(f"  프레임별 평균 이동량 (m, 1..{L}): {pf(mv.tolist())}", flush=True)
print(f"  프레임별 바닥 위 최저 높이 (0..{L}): {pf(hmin.tolist())}", flush=True)
print(f"  프레임별 접촉 입자 수 (0..{L}): {' '.join(str(int(q)) for q in ncont)}", flush=True)
print("MANYTRAJ_CHECK_PASS" if ok else "MANYTRAJ_CHECK_FAIL", flush=True)
