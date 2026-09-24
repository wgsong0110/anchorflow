"""두 궤적 파일을 같은 자로 비교한다 (교사 대 i-PhysGaussian 등).

학생 평가와 **같은 규약**을 쓴다: t0 마다 L 프레임, 위치 오차는 물체 크기로 나누고,
CD 는 Spring-Gaus 의 양방향 평균 최근접 제곱거리, EMD 는 두 구름에서 같은 인덱스를
뽑아 최적 대응을 푼다. 정지 기준선도 함께 내어 "아무것도 안 했을 때" 와 견준다.

입자 순서가 같아야 한다 -- 같은 씨앗으로 생성하면 채우기와 부분표본이 같으므로
그대로 대응된다. 다르면 여기서 멈춘다 (조용히 섞이면 수치가 거짓말을 한다).
"""
import argparse
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="기준 궤적 (교사)")
ap.add_argument("--b", required=True, help="견줄 궤적 (i-PG 등)")
ap.add_argument("--t0", type=int, nargs="*", default=[3, 10, 20])
ap.add_argument("--len", type=int, default=15)
ap.add_argument("--cd_pts", type=int, default=2048)
ap.add_argument("--dump", default=None, help="영상용 덤프 (t0 하나, --dump_len 프레임)")
ap.add_argument("--dump_t0", type=int, default=3)
ap.add_argument("--dump_len", type=int, default=40)
ap.add_argument("--dev", default="cuda")
a = ap.parse_args()

A = torch.load(a.a, map_location="cpu", weights_only=False)
B = torch.load(a.b, map_location="cpu", weights_only=False)
xa, xb = A["x"].float(), B["x"].float()
if xa.shape != xb.shape:
    raise SystemExit(f"모양이 다르다: {tuple(xa.shape)} vs {tuple(xb.shape)}")
if "sel" in A and "sel" in B and not torch.equal(A["sel"], B["sel"]):
    raise SystemExit("부분표본 색인이 다르다 -- 같은 입자가 아니다")
dev = a.dev
xa, xb = xa.to(dev), xb.to(dev)
EXT = float((xa[0].max(0).values - xa[0].min(0).values).norm())
print(f"[비교] {os.path.basename(a.a)} vs {os.path.basename(a.b)}  "
      f"{tuple(xa.shape)}  물체 {EXT:.4f}", flush=True)


def chamfer(p_, q_, chunk=4096):
    def one(u, w_):
        s_, n_ = 0.0, u.shape[0]
        for i_ in range(0, n_, chunk):
            dd = torch.cdist(u[i_:i_ + chunk], w_)
            s_ += float((dd.min(1).values ** 2).sum())
        return s_ / n_
    return one(p_, q_) + one(q_, p_)


def emd(p_, q_, seed):
    from scipy.optimize import linear_sum_assignment
    g_ = torch.Generator().manual_seed(int(seed))
    i_ = torch.randperm(p_.shape[0], generator=g_)[:a.cd_pts].to(p_.device)
    dd = torch.cdist(p_[i_], q_[i_]).double().cpu().numpy()
    r_, c_ = linear_sum_assignment(dd)
    return float(dd[r_, c_].mean())


pos = pos0 = cd = cd0 = em = em0 = 0.0
n = 0
for t0 in a.t0:
    for i in range(a.len):
        t = t0 + i + 1
        if t >= xa.shape[0]:
            break
        gt, pr, st = xa[t], xb[t], xa[t0]
        pos += float((pr - gt).norm(dim=-1).mean()) / EXT
        pos0 += float((st - gt).norm(dim=-1).mean()) / EXT
        cd += chamfer(pr, gt)
        cd0 += chamfer(st, gt)
        em += emd(pr, gt, t) / EXT
        em0 += emd(st, gt, t) / EXT
        n += 1
print(f"  프레임 {n} 개")
print(f"  위치 {100*pos/n:.3f}% (정지 {100*pos0/n:.3f}%) 비 {pos/pos0:.3f}")
print(f"  CD   {cd/n:.3e} (정지 {cd0/n:.3e}) 비 {cd/cd0:.3f}")
print(f"  EMD  {100*em/n:.3f}% (정지 {100*em0/n:.3f}%) 비 {em/em0:.3f}", flush=True)

if a.dump:
    t0 = a.dump_t0
    L = min(a.dump_len, xa.shape[0] - t0 - 1)
    torch.save({"pred": xb[t0 + 1:t0 + 1 + L].cpu(),
                "gt": xa[t0 + 1:t0 + 1 + L].cpu(),
                "ctrl_pos": A.get("ctrl_pos"), "t0": t0,
                "tag": A.get("tag", "cmp")}, a.dump)
    print(f"[덤프] {a.dump}  {L} 프레임", flush=True)
