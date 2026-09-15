"""학생 롤아웃을 MPM 정답과 Chamfer / Earth Mover 거리로 잰다.

지금까지 쓰던 지표는 **같은 입자끼리의 평균 위치 오차**였다. 대응이 이미 알려져
있다는 전제가 깔려 있어서, 물체가 뭉개지거나 찢어져 대응이 흐려지는 경우에는
그 전제가 약해진다. CD 와 EMD 는 대응을 가정하지 않고 점 구름 자체를 비교한다.

  Chamfer -- 각 점에서 상대 구름의 최근접까지 거리, 양방향 평균. 국소적이라
             한쪽이 뭉쳐 있어도 값이 작게 나올 수 있다.
  EMD     -- 일대일 대응을 최적으로 맺었을 때의 평균 이동량. 질량 분포까지 보므로
             뭉침에 속지 않지만 O(n^3) 이라 부분표본으로 잰다.

둘 다 **고정 길이**(물체 크기)로 나눈다. 자기 변위로 나누면 덜 움직인 쪽이
유리해진다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True, help="학생 체크포인트")
ap.add_argument("--fit", default=None, help="앵커 기하 (stu2 계열은 별도 파일)")
ap.add_argument("--dreamphysics", required=True)
ap.add_argument("--anchorflow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_traj", type=int, default=5, help="홀드아웃 궤적 수")
ap.add_argument("--emd_sample", type=int, default=2048,
                help="EMD 부분표본 크기. O(n^3) 이라 전체로는 못 푼다")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, a.anchorflow)
sys.path.insert(0, a.dreamphysics)
dev = "cuda"
torch.set_grad_enabled(False)

import warp as wp  # noqa: E402

wp.init()
from anchorflow import scene_setup  # noqa: E402
from anchorflow.mpm_teacher import MPMTeacher  # noqa: E402

# 앵커 수는 기하 파일이 정한다. 씬을 다른 수로 세우면 fixed_mask 길이가 어긋난다.
_n_anch = a.n_anchors
if a.fit and os.path.exists(a.fit):
    _f = torch.load(a.fit, map_location="cpu", weights_only=False)
    for _k in ("pos", "ac", "anchor"):
        if _k in _f:
            _n_anch = int(_f[_k].shape[0])
            break
    del _f
    print(f"[앵커] 기하 파일 기준 {_n_anch} 개", flush=True)
sc = scene_setup.build(a.ply, a.config, _n_anch, a.K, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
FRAME_DT = float(sc.sub_dt) * a.dt_mult
EXT = float(sc.extent)
print(f"[씬] 물질 {int(sc.keep.sum())}, 물체 크기 {EXT:.4f}, "
      f"frame_dt {FRAME_DT:.4g}", flush=True)


def chamfer(p, q):
    """양방향 평균 최근접 거리. p,q: [N,3], [M,3]"""
    d = torch.cdist(p, q)                       # [N, M]
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())


def emd(p, q, n):
    """부분표본 위에서 최적 일대일 대응의 평균 이동량 (헝가리안, 정확해)."""
    from scipy.optimize import linear_sum_assignment
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    ip = torch.randperm(p.shape[0], generator=g)[:n]
    iq = torch.randperm(q.shape[0], generator=g)[:n]
    d = torch.cdist(p[ip.to(p.device)], q[iq.to(q.device)]).double().cpu().numpy()
    r, c = linear_sum_assignment(d)
    return float(d[r, c].mean())


# --- 학생 ---
st = torch.load(a.ckpt, map_location=dev, weights_only=False)
from anchorflow.nextstate import NextStep, apply_step  # noqa: E402

# 구조는 체크포인트에 저장된 학습 인자를 그대로 따른다. 임의로 고르면
# state_dict 가 안 맞는다 (stu2 계열은 hidden/depth/heads/chunk 가 다를 수 있다).
_ta = st.get("args", {}) or {}
if not isinstance(_ta, dict):
    _ta = vars(_ta)
net = NextStep(hidden=int(_ta.get("hidden", 128)),
               depth=int(_ta.get("depth", 4)),
               heads=int(_ta.get("heads", 4)),
               use_accel=not bool(_ta.get("no_accel", True)),
               scale=EXT, vel_scale=EXT / max(FRAME_DT, 1e-6),
               chunk=int(_ta.get("chunk", 1)), zero_init=True).to(dev)
key = "net" if "net" in st else "model"
net.load_state_dict(st[key])
net.eval()
print(f"[구조] hidden {_ta.get('hidden')}, depth {_ta.get('depth')}, "
      f"heads {_ta.get('heads')}, chunk {_ta.get('chunk')}, "
      f"no_accel {_ta.get('no_accel')}", flush=True)

if "ac" in st:
    AC0 = st["ac"].to(dev)
else:
    fit = torch.load(a.fit, map_location=dev, weights_only=False)
    # 기하 파일은 앵커 위치를 pos 로 들고 있다 (ac / anchor 가 아니다).
    for _k in ("pos", "ac", "anchor"):
        if _k in fit:
            AC0 = fit[_k].to(dev)
            print(f"[기하] {a.fit} 의 '{_k}' 사용, 앵커 {AC0.shape[0]}", flush=True)
            break
    else:
        raise KeyError(f"앵커 위치를 못 찾았다. 키: {sorted(fit.keys())}")
print(f"[학생] {key} 로드, 앵커 {AC0.shape[0]}, iter {st.get('iter')}", flush=True)

T = MPMTeacher(sc, horizon=a.frames * FRAME_DT * 1.1)
n_sub = a.dt_mult
rng = np.random.RandomState(a.seed)
rows = []
for ti in range(a.n_traj):
    # 홀드아웃 궤적: 같은 분포에서 다른 씨앗으로 뽑은 임펄스
    g = torch.Generator(device=dev).manual_seed(1000 + ti)
    d = torch.randn(3, device=dev, generator=g)
    v0p = (0.3 * EXT / (a.frames * FRAME_DT) * d / d.norm()).expand(T.n, 3)
    T._set(T.pos_m.clone(), v0p.contiguous(), T.eye.clone(),
           torch.zeros_like(T.eye))
    truth = [T.pos_m.clone()]
    bad = False
    for _ in range(a.frames - 1):
        for k in range(n_sub):
            T.solver.p2g2p(None, float(sc.sub_dt), device=T.wp_dev)
            if (k + 1) % 4 == 0 and (not T._in_domain() or not T._vel_safe(4)):
                bad = True
                break
        if bad:
            break
        truth.append(T.solver.export_particle_x_to_torch().clone())
    if bad or len(truth) < a.frames:
        print(f"  궤적 {ti}: 격자 이탈 -- 건너뜀", flush=True)
        continue
    TR = torch.stack(truth)

    # 학생 롤아웃: 같은 초기 속도
    p = AC0.clone()
    # 임펄스가 전 입자에 같은 속도를 주므로 앵커도 같은 속도로 시작한다.
    v = v0p[0].unsqueeze(0).expand(p.shape[0], 3).contiguous().clone()
    v[sc.fixed_mask] = 0
    gp = sc.pos.clone()
    pred = [gp[sc.keep].clone()]
    for _ in range(a.frames - 1):
        q, dd = apply_step(net, p, v, None, FRAME_DT, sc.fixed_mask)[-1]
        p, v = q, dd / FRAME_DT
        gp = sc.skin(p, gp)
        pred.append(gp[sc.keep].clone())
    PR = torch.stack(pred)

    cds, emds = [], []
    for t in range(TR.shape[0]):
        cds.append(float(chamfer(PR[t], TR[t])) / EXT)
        emds.append(emd(PR[t], TR[t], a.emd_sample) / EXT)
    rows.append(dict(traj=ti, cd_last=cds[-1], cd_mean=float(np.mean(cds)),
                     emd_last=emds[-1], emd_mean=float(np.mean(emds))))
    print(f"  궤적 {ti}: CD 마지막 {100*cds[-1]:.3f}% 평균 {100*np.mean(cds):.3f}% | "
          f"EMD 마지막 {100*emds[-1]:.3f}% 평균 {100*np.mean(emds):.3f}%", flush=True)

os.makedirs(a.out, exist_ok=True)
summ = dict(n_traj=len(rows), extent=EXT, frames=a.frames,
            emd_sample=a.emd_sample, rows=rows)
if rows:
    summ["cd_mean"] = float(np.mean([r["cd_mean"] for r in rows]))
    summ["emd_mean"] = float(np.mean([r["emd_mean"] for r in rows]))
    print(f"\n[요약] {len(rows)} 궤적 | CD {100*summ['cd_mean']:.3f}% | "
          f"EMD {100*summ['emd_mean']:.3f}%  (둘 다 물체 크기 대비)", flush=True)
json.dump(summ, open(os.path.join(a.out, "cd_emd.json"), "w"), indent=1)
print(f"[저장] {os.path.join(a.out, 'cd_emd.json')}", flush=True)
print("CDEMD_OK")
