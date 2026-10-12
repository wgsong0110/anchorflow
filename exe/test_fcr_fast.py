"""_FcrPsiFast 가 기존 _FcrPsi 와 같은 값·기울기를 내는지 (float64 기준해와도) 비교한다."""
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow import phys_resid as pr

torch.manual_seed(0)
dev = "cuda"
mu, lam = 1e6 / 2.6, 1e6 * 0.3 / (1.3 * 0.4)


def rot(n):
    q = torch.randn(n, 4, device=dev); q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(n, 3, 3)


def mk(n, lo, hi, flip=False):
    s = torch.exp(torch.empty(n, 3, device=dev).uniform_(lo, hi))
    if flip:
        s[:, 0] = -s[:, 0]
    return rot(n) @ torch.diag_embed(s) @ rot(n)


cases = {"정지 근처 (1e-4)": torch.eye(3, device=dev) + 1e-4 * torch.randn(200000, 3, 3, device=dev),
         "작은 변형 (1e-2)": torch.eye(3, device=dev) + 1e-2 * torch.randn(200000, 3, 3, device=dev),
         "보통 신축 (±30%)": mk(200000, -0.3, 0.3),
         "큰 신축 (σ 0.05~5)": mk(200000, -3.0, 1.6),
         "뒤집힘 (det<0)": mk(200000, -0.5, 0.5, flip=True),
         "하한 걸림 섞임": torch.cat([mk(1000, -6.0, -5.0), mk(199000, -0.3, 0.3)])}
for name, F in cases.items():
    out = {}
    for tag, fn in (("old", pr._FcrPsi.apply), ("new", pr._FcrPsiFast.apply)):
        Fi = F.clone().requires_grad_(True)
        p = fn(Fi, mu, lam, 0.01); p.sum().backward()
        out[tag] = (p.detach(), Fi.grad)
    F64 = F.double().requires_grad_(True)
    p64 = pr._FcrPsi.apply(F64, mu, lam, 0.01); p64.sum().backward()
    sc_p = p64.abs().max().clamp_min(1e-30); sc_g = F64.grad.abs().max().clamp_min(1e-30)
    for tag in ("old", "new"):
        ep = float((out[tag][0].double() - p64).abs().max() / sc_p)
        eg = float((out[tag][1].double() - F64.grad).abs().max() / sc_g)
        print(f"{name:16s} {tag}: ψ 최대 상대오차 {ep:.2e}  기울기 최대 상대오차 {eg:.2e}")
F = mk(313750, -0.3, 0.3)
for tag, fn in (("old", pr._FcrPsi.apply), ("new", pr._FcrPsiFast.apply)):
    for k in range(3):
        Fi = F.clone().requires_grad_(True); torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            fn(Fi, mu, lam, 0.01).sum().backward()
        torch.cuda.synchronize()
    print(f"속도 {tag}: 313750 입자 순전파+역전파 {1e3 * (time.time() - t) / 20:.2f} ms")
