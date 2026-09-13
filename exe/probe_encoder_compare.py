"""세 인코더를 같은 자리에서 견준다: 회전벡터+신축 / 자유 F 선형 / 로그-유클리드.

재는 것 넷:
  1. F 잔차 -- 얼마나 정확한가
  2. 블렌드된 F_g 의 행렬식 분포 -- 부피가 붕괴하는가 (F 잔차는 이걸 못 잡는다.
     납작해진 가우시안도 프로베니우스 오차는 작을 수 있는데 렌더링에선 얇은 조각이다)
  3. 앵커 상태의 크기와 스텝 변화 꼬리 -- 학생이 배울 수 있는가
  4. 복호 비용
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from anchorflow import scene_setup

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--fit", required=True); ap.add_argument("--traj_cache", required=True)
ap.add_argument("--n_win", type=int, default=6); ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--n_pair", type=int, default=4, help="스텝 변화용 연속 프레임 쌍 수")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp
dev = "cuda"; torch.set_grad_enabled(False); wp.init()
from anchorflow.anchor_sparse import Traj, load_fitted
from anchorflow.frame_encode import FrameState, polar_target
from anchorflow.frame_logeuc import LogEucState, logm_target
sys.modules["__main__"].Traj = Traj

sc = scene_setup.build(args.ply, args.config, 512, 8, device=dev,
                       frozen_weights=True, rot_fallback=True, eig_floor=0.02)
fit = load_fitted(sc, args.fit, dev)[0].fit
cache = fit.prepare(); w = cache[0]; Yg = fit.Xc - cache[1]
MASS = sc.volume[sc.keep].clone()
FS = FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS)
LE = LogEucState(FS)
L = FS.gram(w)
FIT = torch.load(args.traj_cache, map_location=dev, weights_only=False)["fit"]
g = torch.Generator(device="cpu"); g.manual_seed(20260908)
WIN = [(int(torch.randint(len(FIT), (1,), generator=g).item()),
        int(torch.randint(args.frames - 1, (1,), generator=g).item()))
       for _ in range(args.n_win)]
print(f"[setup] 앵커 {fit.M}, 가우시안 {fit.N}, 평가창 {len(WIN)}", flush=True)


def mw(e2): return float(((MASS * e2).sum() / MASS.sum()).sqrt())
def q(v, qs=(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)):
    s = v.reshape(-1)
    s = s[torch.randperm(s.numel(), device=s.device)[:200000]]
    return [float(x) for x in torch.quantile(s, torch.tensor(qs, device=s.device))]


ENC = {}
def free_F(F0):
    return FS._wls(F0.reshape(-1, 9), w, L).view(-1, 3, 3)

for name in ("회전+신축", "자유 F", "로그-유클리드"):
    ef = detq = None
    dets, sts, dsts, efs = [], [], [], []
    t0 = time.time()
    for i, t in WIN:
        F0 = FIT[i][2][t].to(dev, torch.float32).view(-1, 3, 3)
        F1 = FIT[i][2][t + 1].to(dev, torch.float32).view(-1, 3, 3)
        if name == "회전+신축":
            a0 = torch.cat(polar_target(F0), -1)                 # [N,6] 목표(참고용)
            _, u0, s0 = FS.encode_closed(FIT[i][0][t].to(dev, torch.float32), F0, w, Yg,
                                          fixed=fit.fixed, p_fix=fit.pos)
            _, u1, s1 = FS.encode_closed(FIT[i][0][t + 1].to(dev, torch.float32), F1, w, Yg,
                                          fixed=fit.fixed, p_fix=fit.pos)
            Fh = FS.decode(u0, s0, w)
            st = torch.cat([u0, s0], -1); st1 = torch.cat([u1, s1], -1)
        elif name == "자유 F":
            X0 = free_F(F0); X1 = free_F(F1)
            Fh = FS.blend(X0.reshape(-1, 3)[:, :3] * 0, X0.reshape(-1, 3)[:, :3] * 0, w)[0] \
                 if False else None
            # 선형 블렌드
            Fh = torch.zeros(fit.N, 3, 3, device=dev).index_add_(
                0, fit.pair_g, w.reshape(-1, 1, 1) * X0[fit.pair_a])
            st = X0.reshape(fit.M, 9); st1 = X1.reshape(fit.M, 9)
        else:
            X0 = LE.encode(F0, w, L=L); X1 = LE.encode(F1, w, L=L)
            Fh = LE.decode(X0, w)
            st = X0.reshape(fit.M, 9); st1 = X1.reshape(fit.M, 9)
        efs.append(mw((Fh - F0).pow(2).sum((-1, -2))))
        dets.append(torch.linalg.det(Fh))
        sts.append(st.norm(dim=-1)); dsts.append((st1 - st).norm(dim=-1))
    dt = (time.time() - t0) / len(WIN)
    ENC[name] = dict(F=sum(efs) / len(efs), det=q(torch.cat(dets)),
                     st=q(torch.cat(sts)), dst=q(torch.cat(dsts)), ms=dt * 1e3)
    print(f"  {name}: F {ENC[name]['F']:.4f}  ({dt*1e3:.0f} ms/창)", flush=True)

QL = ("p0.1", "p1", "p10", "p50", "p90", "p99", "p100")
print(f"\n== 1. F 잔차 ==")
for k, v in ENC.items(): print(f"  {k:<14}{v['F']:.4f}")
print(f"\n== 2. 블렌드된 F_g 의 행렬식 (1 이 부피 보존, 0 이면 납작) ==")
print(f"{'인코더':<14}" + "".join(f"{a:>10}" for a in QL))
for k, v in ENC.items(): print(f"{k:<14}" + "".join(f"{a:>10.4f}" for a in v["det"]))
print(f"\n== 3. 앵커 상태 크기 |X_a| ==")
print(f"{'인코더':<14}" + "".join(f"{a:>10}" for a in QL))
for k, v in ENC.items(): print(f"{k:<14}" + "".join(f"{a:>10.3f}" for a in v["st"]))
print(f"\n== 4. 스텝 변화 |dX_a| (한 코어스 프레임) ==")
print(f"{'인코더':<14}" + "".join(f"{a:>10}" for a in QL) + f"{'꼬리비':>10}")
for k, v in ENC.items():
    d = v["dst"]
    print(f"{k:<14}" + "".join(f"{a:>10.4f}" for a in d)
          + f"{d[-1]/max(d[3],1e-9):>10.0f}x")
print(f"\n== 5. 인코드+복호 비용 ==")
for k, v in ENC.items(): print(f"  {k:<14}{v['ms']:.0f} ms/창")
print("\n꼬리비 = p100/p50. 클수록 학생이 예측할 양의 분포가 험하다.")
print("COMPARE_DONE")
