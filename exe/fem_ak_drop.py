"""절 AK.29 단위 시험 자료: 바닥에 떨어지는 균일 정육면체 블록 (lego 재질·바닥 정의 그대로, config.json).
- 충격 속력: lego 미달 5 궤적 (0001·0003·0016·0017·0018) 의 정답 첫 착지 직전 질량 중심 연직 속력의 평균 (첫 착지 = 최저 입자가 처음 바닥 + 0.5 dx_g 아래로 내려간 프레임).
- 입자 밀도: 같은 다섯 궤적 0 프레임의 점유 정답 칸 (dx_g 0.02) 당 입자 수 평균 -> 입자 간격 = dx_g / (칸당 수)^(1/3).
- 크기: 같은 다섯 궤적과 같은 입자 수·밀도 (= 같은 질량) 의 정육면체 -- 한 변 = dx_g (점유 칸 수)^(1/3) 의 평균 (바운딩 박스 기하 평균 0.93 m 는 속이 빈 lego 보다 8 배 무거워 버림).
- 처음 상태: 바닥 위 간격 g0 = 0.05 m 에 정지 회전 없이, 연직 속도 v0 = −sqrt(v_imp² − 2 g g0) (착지 순간 속력 v_imp), F = I.
출력 <out>/init.pt (fem_ex_sim 형식, x 는 0 프레임만) + info.json. 사용: python exe/fem_ak_drop.py <out>"""
import json
import math
import os
import sys

import torch

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cfg = json.load(open("/home/dkta/work/lbdata/config.json"))
fz = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2]
dxg = float(cfg["grid_lim"]) / int(cfg["n_grid"])
dt = float(cfg["frame_dt"])
g = abs(float(cfg["g"][2]))
vi, dens, ext = [], [], []
for t in ("0001", "0003", "0016", "0017", "0018"):
    d = torch.load(f"/tmp/dkta/af_data/{t}.pt", map_location="cpu", weights_only=False, mmap=True)
    x = d["x"].float()
    zmin = x[:, :, 2].min(1).values
    k = int(torch.nonzero(zmin < fz + 0.5 * dxg)[0])
    cz = x[:, :, 2].mean(1)
    vi.append(float((cz[k - 1] - cz[k]) / dt) if k >= 1 else float("nan"))
    c = (x[0] / dxg).floor().long()
    _, cnt = torch.unique((c[:, 0] * 1000 + c[:, 1]) * 1000 + c[:, 2], return_counts=True)
    dens.append(float(cnt.float().mean()))
    ext.append(dxg * float(cnt.numel()) ** (1 / 3))
    print(t, "착지 프레임", k, "속력", round(vi[-1], 3), "칸당 입자", round(dens[-1], 2), "크기", round(ext[-1], 3), "입자", x.shape[1], flush=True)
v_imp = sum(vi) / len(vi)
nc = sum(dens) / len(dens)
sp = dxg / nc ** (1 / 3)
Lb = sum(ext) / len(ext)
n = max(2, int(round(Lb / sp)))
g1 = (torch.arange(n, dtype=torch.float64) + 0.5) * sp - n * sp / 2
X = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
g0 = 0.05
X[:, 0] += 1.0; X[:, 1] += 1.0
X[:, 2] += fz + g0 + n * sp / 2
v0 = -math.sqrt(max(v_imp ** 2 - 2 * g * g0, 0.0))
N = X.shape[0]
V = torch.zeros(1, N, 3); V[..., 2] = v0
info = dict(v_imp=v_imp, v0=v0, cell_density=nc, spacing=sp, side=n * sp, n_side=n, N=N, floor_z=fz, gap=g0, per_traj=dict(v=vi, dens=dens, ext=ext))
json.dump(info, open(os.path.join(out, "info.json"), "w"), ensure_ascii=False, indent=1)
torch.save(dict(cfg=cfg, x=X.float()[None], v=V, F=torch.eye(3)[None, None].expand(1, N, 3, 3).contiguous(), F_is_true=True), os.path.join(out, "init.pt"))
print("[블록]", json.dumps({k: v for k, v in info.items() if k != "per_traj"}, ensure_ascii=False), flush=True)
