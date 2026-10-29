"""절 Z 검산 (GPU, 측정 밖): A_T 특이값 비율 σ1/σ3 의 (2) 로그 장벽 힘 (ex_tet_force_SR, 해석 미분) vs float64 에너지 중앙 차분,
(1) 클램프 (ex_lim_sr_tet) 가 준 상자 [0,s_T]⁴ 안 무작위 배치의 실제 비율 위반률.
서로 독립인 사면체 T 개 (사면체마다 노드 4 개) 로 만든다.
사용: python exe/fem_z_check.py [--T 400] [--c 2] [--seed 0]  -> 표준 출력 (요약 json 한 줄 포함)"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
from anchorflow import fem_ex_cuda as EX                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=400)
ap.add_argument("--c", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
T = a.T
M = 4 * T
c = a.c
dh = 0.5 * (c - 1.0)
mu, lam, kap = 3.0e5, 5.0e5, 1.0


def rand_rot(n):
    q = torch.randn(n, 4, dtype=torch.float64, device=dev)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(n, 3, 3)


# 기준 사면체 (Kuhn 꼴 + 잡음) -> ∇λ
X = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float64, device=dev) * 0.03
X = X.unsqueeze(0) + 0.003 * torch.randn(T, 4, 3, dtype=torch.float64, device=dev)
D = (X[:, 1:] - X[:, :1]).transpose(1, 2)                                            # 열 = 변
Di = torch.linalg.inv(D)                                                             # 행 1..3 = ∇λ_1..3
gl = torch.cat([-Di.sum(1, keepdim=True), Di], 1)                                    # [T,4,3]


def A_of(U):                                                                         # U [T,4,3] -> A [T,3,3] = I + Σ U_i ∇λ_iᵀ
    return torch.eye(3, dtype=U.dtype, device=dev) + torch.einsum("tic,tib->tcb", U, gl.to(U.dtype))


def ratio(A):
    sv = torch.linalg.svdvals(A)
    r = sv[:, 0] / sv[:, -1]
    return torch.where(torch.linalg.det(A) > 0, r, torch.full_like(r, float("inf")))


def bar(d, jh):
    eps = 1e-3 * jh
    dc = torch.clamp(d, min=eps)
    b = -(dc - jh) ** 2 * torch.log(dc / jh)
    db = -2 * (dc - jh) * torch.log(dc / jh) - (dc - jh) ** 2 / dc
    b = torch.where(d < eps, b + db * (d - eps), b)
    return torch.where(d < jh, b, torch.zeros_like(d))


def kforce(U, k):
    """커널 힘 [T,4,3] (float32)."""
    sz = torch.tensor([M, T, 0, 0], dtype=torch.int32, device=dev)
    tets = torch.arange(M, device=dev, dtype=torch.long).view(T, 4)
    f = torch.zeros(M, 3, device=dev)
    rst = torch.zeros(12, device=dev)
    Uf, glf, FTf, VTf = U.float().reshape(M, 3).contiguous(), gl.float().reshape(T, 12).contiguous(), torch.eye(3, device=dev).reshape(1, 9).repeat(T, 1).contiguous(), VT.float().contiguous()
    EX.launch("ex_tet_force_SR", T, (EX.P(sz), EX.P(tets), EX.P(Uf), EX.P(glf),     # 임시 텐서를 이름에 묶어 둔다 (EX.P 는 주소만 -- 풀린 메모리를 읽지 않게)
                                     EX.P(FTf), EX.P(VTf),
                                     np.float32(mu), np.float32(lam), EX.P(f), np.float32(0.0), np.float32(0.3), np.int32(0), EX.P(rst),
                                     np.float32(k), np.float32(c), np.float32(dh)))
    torch.cuda.synchronize()
    return f.view(T, 4, 3).double(), rst


# ---------- (2) 장벽: 비율이 활성 구간 (c - d̂, c) 에 오도록 A = R1 diag(s) R2
VT = (0.5 + torch.rand(T, dtype=torch.float64, device=dev)) * 4.5e-6
r_t = (c - dh) + dh * (0.05 + 0.9 * torch.rand(T, dtype=torch.float64, device=dev))
s2 = 1.0 + (r_t - 1.0) * torch.rand(T, dtype=torch.float64, device=dev)              # 가운데 특이값 (겹침 없게 무작위)
S = torch.stack([r_t, s2, torch.ones_like(r_t)], -1) * (0.8 + 0.4 * torch.rand(T, 1, dtype=torch.float64, device=dev))
At = rand_rot(T) @ torch.diag_embed(S) @ rand_rot(T)
At = At * torch.sign(torch.linalg.det(At)).view(T, 1, 1)
U = torch.einsum("tcb,tib->tic", At - torch.eye(3, dtype=torch.float64, device=dev), X - X.mean(1, keepdim=True))
r0 = ratio(A_of(U))
f1, rst = kforce(U, kap)
f0, _ = kforce(U, 0.0)
fk = f1 - f0                                                                         # 장벽 몫 (커널, 해석 미분)


def Etet(U_):
    return VT * kap * mu * bar(c - ratio(A_of(U_)), dh)


hh = 1e-7
ffd = torch.zeros(T, 4, 3, dtype=torch.float64, device=dev)
for i in range(4):
    for q in range(3):
        Up, Um = U.clone(), U.clone()
        Up[:, i, q] += hh
        Um[:, i, q] -= hh
        ffd[:, i, q] = -(Etet(Up) - Etet(Um)) / (2 * hh)
Ua = U.clone().requires_grad_(True)
fag = -torch.autograd.grad(Etet(Ua).sum(), Ua)[0]
rel = (fk - ffd).flatten(1).norm(dim=-1) / ffd.flatten(1).norm(dim=-1).clamp_min(1e-30)
rel_ag = (fk - fag).flatten(1).norm(dim=-1) / fag.flatten(1).norm(dim=-1).clamp_min(1e-30)
rel_fa = (ffd - fag).flatten(1).norm(dim=-1) / fag.flatten(1).norm(dim=-1).clamp_min(1e-30)
inact = (fk.flatten(1).norm(dim=-1) == 0).sum().item()
mom = fk.sum(1).norm(dim=-1) / fk.flatten(1).norm(dim=-1).clamp_min(1e-30)          # 사면체 힘 합 (병진 불변 -> 0)
print(f"[장벽] 사면체 {T}, c {c} d̂ {dh}, 비율 범위 {float(r0.min()):.4f}~{float(r0.max()):.4f}, 활성 계수 {int(rst[10])}, 힘 0 인 사면체 {inact}")
print(f"  커널 vs 중앙 차분 (h {hh:g}, float64 에너지) 상대 차: 중앙 {float(rel.median()):.2e}, 99 % {float(rel.quantile(0.99)):.2e}, 최대 {float(rel.max()):.2e}")
print(f"  커널 vs autograd: 중앙 {float(rel_ag.median()):.2e}, 최대 {float(rel_ag.max()):.2e};  중앙 차분 vs autograd: 최대 {float(rel_fa.max()):.2e}")
print(f"  사면체 힘 합 / |힘| (병진 불변): 최대 {float(mom.max()):.2e}")
# 비활성 쪽: 비율 < c - d̂ 이면 장벽 힘 0
U2 = U * 0.3
r2 = ratio(A_of(U2))
f1b, _ = kforce(U2, kap)
f0b, _ = kforce(U2, 0.0)
nz = ((f1b - f0b).flatten(1).abs().max(-1).values > 0)
print(f"  비활성 확인 (U x 0.3, 비율 < c - d̂ 인 사면체 {int((r2 < c - dh).sum())}): 그중 힘이 바뀐 것 {int((nz & (r2 < c - dh)).sum())}")

# ---------- (1) 클램프: 지금 비율 1~c, 무작위 이동 -> 커널 s_T -> 상자 안 무작위 배치 위반률
r_t = 1.0 + (c - 1.0) * torch.rand(T, dtype=torch.float64, device=dev) * 0.95
s2 = 1.0 + (r_t - 1.0) * torch.rand(T, dtype=torch.float64, device=dev)
S = torch.stack([r_t, s2, torch.ones_like(r_t)], -1)
At = rand_rot(T) @ torch.diag_embed(S) @ rand_rot(T)
At = At * torch.sign(torch.linalg.det(At)).view(T, 1, 1)
U = torch.einsum("tcb,tib->tic", At - torch.eye(3, dtype=torch.float64, device=dev), X - X.mean(1, keepdim=True))
hs = 1e-4
vn = torch.randn(T, 4, 3, dtype=torch.float64, device=dev) * (0.03 / hs) * torch.rand(T, 1, 1, dtype=torch.float64, device=dev)   # 이동 |d| ~ 0~3 cm (셀 3 cm)
res = {}
for jon, jmin in ((0, -1.0), (1, 0.3)):
    sz = torch.tensor([M, T, 0, 0], dtype=torch.int32, device=dev)
    tets = torch.arange(M, device=dev, dtype=torch.long).view(T, 4)
    alph = torch.full((M,), 1065353216, dtype=torch.int32, device=dev)
    rst = torch.zeros(12, device=dev)
    glf, VTo, vnf, Uf = gl.float().reshape(T, 12).contiguous(), torch.ones(T, device=dev), vn.float().reshape(M, 3).contiguous(), U.float().reshape(M, 3).contiguous()
    EX.launch("ex_lim_sr_tet", T, (EX.P(sz), EX.P(tets), EX.P(glf), EX.P(VTo), EX.P(vnf), EX.P(Uf), np.float32(hs),
                                   np.float32(c), np.int32(jon), np.float32(jmin), EX.P(alph), EX.P(rst)))
    torch.cuda.synchronize()
    sT = alph.view(torch.float32).view(T, 4).double().min(1).values
    d = hs * vn
    K = 64
    al = torch.rand(K, T, 4, dtype=torch.float64, device=dev) * sT.view(1, T, 1)
    al[0] = sT.view(T, 1).expand(T, 4)                                               # 실제로 쓰는 꼭짓점 (모두 s_T)
    Ak = A_of(U).unsqueeze(0) + torch.einsum("ktj,tjc,tjb->ktcb", al, d, gl)
    rk = ratio(Ak.reshape(-1, 3, 3)).view(K, T)
    thrR = torch.maximum(torch.full_like(r_t, c), ratio(A_of(U)))
    viol = rk > thrR.view(1, T) * (1 + 1e-4)
    dk = torch.linalg.det(Ak.reshape(-1, 3, 3)).view(K, T)
    thrJ = torch.minimum(torch.full_like(r_t, jmin), torch.linalg.det(A_of(U)))
    violJ = (dk < thrJ.view(1, T) - 1e-5) if jon else torch.zeros_like(viol)
    # 꼭 필요했나: s_T 가 1 보다 작을 때, s_T 의 1.05 배 꼭짓점 (모두 이동) 이 실제로 위반하는 비율 (보수성)
    lim = sT < 1
    A105 = A_of(U) + torch.einsum("tjc,tjb->tcb", (sT * 1.05).clamp(max=1).view(T, 1, 1) * d, gl)
    tight = (ratio(A105) > thrR * (1 + 1e-4)) | ((torch.linalg.det(A105) < thrJ) if jon else torch.zeros_like(lim))
    res[jon] = dict(limited=int(lim.sum()), weyl_or_v1=int((~lim).sum()), viol_frac=float(viol.float().mean()), viol_tets=int(viol.any(0).sum()),
                    violJ=int(violJ.any(0).sum()), vertex_ok=int((~viol[0]).sum()), tight_frac=float(tight[lim].float().mean()) if bool(lim.any()) else None)
    print(f"[클램프 {'비율 + det ≥ 0.3' if jon else '비율만'}] c {c}: 제한된 사면체 {res[jon]['limited']}/{T}, 상자 안 무작위 {K} 배치 위반률 {res[jon]['viol_frac']:.2e} "
          f"(위반 사면체 {res[jon]['viol_tets']}, det 위반 {res[jon]['violJ']}), 쓰는 배치 (모두 s_T) 만족 {res[jon]['vertex_ok']}/{T}, "
          f"s_T x 1.05 가 위반하는 비율 (판정이 빡빡한 정도) {res[jon]['tight_frac']}")
print("[요약]", json.dumps(dict(c=c, T=T, bar_rel_med=float(rel.median()), bar_rel_max=float(rel.max()), bar_rel_ag_max=float(rel_ag.max()), clamp=res)))
