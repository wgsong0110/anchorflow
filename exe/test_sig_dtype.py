"""특이값 분해를 float32 로 해도 되는지 측정한다.

`_sig`/`_sig_vec` 는 float64 + 지터(tr*diag(0,1e-9,2e-9)) 로 돈다. 축퇴 고유값에서
타이치 f32 가 터진 전례가 있어 보수적으로 잡은 것인데, 여기서는 **torch 의**
eigh 를 f32 로 돌 때의 오차를 실제 F 분포에서 재서 판단한다.

판정 기준: 에너지밀도 Psi 와 소성 보정 dlog 가 f64 기준 대비 상대오차 1e-5
아래이고 NaN/inf 가 없어야 한다 (학습은 fp32 로 돈다).
"""
from __future__ import annotations
import argparse, os, sys, time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import phys_resid as PR

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=251001)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
cfg = {"E": 2e6, "nu": 0.3, "material": "metal", "yield_stress": 4e4,
       "density": 1000}


def sig_vec(F, dt):
    C = (F.transpose(-1, -2) @ F).to(dt)
    jit = torch.diag(torch.tensor([0.0, 1e-9, 2e-9], dtype=dt, device=C.device))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(
        *C.shape[:-2], 1, 1).clamp_min(1e-12)
    w, V = torch.linalg.eigh(C + tr * jit)
    return w.clamp_min(1e-12).sqrt(), V


def timed(fn, rep=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize() if dev == "cuda" else None
    return (time.time() - t0) / rep * 1000


for name, scale in (("정지 근처(항등+1e-3)", 1e-3), ("보통(0.1)", 0.1),
                    ("큰 변형(0.5)", 0.5)):
    I3 = torch.eye(3, device=dev).expand(a.n, 3, 3).contiguous()
    F = (I3 + scale * torch.randn(a.n, 3, 3, device=dev)).contiguous()
    s64, V64 = sig_vec(F, torch.float64)
    s32, V32 = sig_vec(F, torch.float32)
    bad32 = int((~torch.isfinite(s32)).sum())
    rel = ((s32.double() - s64).abs() / s64.abs().clamp_min(1e-12)).max()
    # Psi 와 dlog 까지 비교 (실제로 쓰이는 양)
    mu, lam = PR.lame(cfg["E"], cfg["nu"])
    e64 = s64.clamp_min(0.01).log()
    e32 = s32.clamp_min(0.01).log().double()
    p64 = PR._vm_project(e64, mu, lam, cfg["yield_stress"])
    p32 = PR._vm_project(e32, mu, lam, cfg["yield_stress"])
    dl = ((p32 - e32) - (p64 - e64)).abs().max()
    ps64 = PR._psi_hencky(p64, mu, lam)
    ps32 = PR._psi_hencky(p32, mu, lam)
    # 실제로 쓰이는 양: 탄성 에너지 **합** 과 그 F 기울기. 개별 Psi 의 상대오차는
    # Psi->0 인 정지 근처에서 뜻이 없다.
    vol = torch.full((a.n,), 1.0 / a.n, device=dev, dtype=torch.float64)
    E64, E32 = (vol * ps64).sum(), (vol * ps32).sum()
    rE = float((E32 - E64).abs() / E64.abs().clamp_min(1e-30))

    def e_of(dt):
        Fv = F.clone().requires_grad_(True)
        sg, _ = sig_vec(Fv, dt)
        ep = sg.clamp_min(0.01).log().double()
        pp = PR._vm_project(ep, mu, lam, cfg["yield_stress"])
        E = (vol * PR._psi_hencky(pp, mu, lam)).sum()
        gr, = torch.autograd.grad(E, Fv)
        return E, gr

    _, g64 = e_of(torch.float64)
    _, g32 = e_of(torch.float32)
    rg = float((g32 - g64).norm() / g64.norm().clamp_min(1e-30))
    # sig 상대오차가 큰 것은 sigma~0 인 입자다 -- 절대차와 함께 본다
    absd = float((s32.double() - s64).abs().max())
    t64 = timed(lambda: sig_vec(F, torch.float64))
    t32 = timed(lambda: sig_vec(F, torch.float32))
    print(f"{name}: f32 비유한 {bad32}  sig 상대 {float(rel):.1e} 절대 {absd:.1e}  "
          f"dlog 절대 {float(dl):.1e}")
    print(f"    **탄성에너지 합 상대오차 {rE:.2e}   기울기 상대오차 {rg:.2e}**")
    print(f"    시간  f64 {t64:.1f}ms   f32 {t32:.1f}ms   ({t64/max(t32,1e-9):.1f}배)")
