"""tri_spline(단조 RQS 셀 재배열) + SITReg 합성의 정합성 검사.

  python exe/test_tri_spline.py            # CPU 로도 돈다
"""
import torch

from anchorflow import tri_spline as TS
from anchorflow.conv_stepper import ConvStepper
from anchorflow.sitreg_warp import (SITRegWarp, FALLBACK_BOUND_444,
                                    cubic_bspline_g2p)

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = 8
P = TS.n_params(K)
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


# 1) 원시 파라미터 0 -> 정확히 항등
u = torch.rand(5000, 3, device=dev)
th0 = torch.zeros(5000, 3, 3 * K + 1, device=dev)
err = (TS.rqs(u, th0, K) - u).abs().max()
chk("항등 초기화", err < 2e-3, f"max|T(u)-u|={float(err):.2e}")

# 2) 무작위 파라미터: 단조 + 치역 [0,1]
th = torch.randn(1, 3, 3 * K + 1, device=dev) * 3.0
us = torch.linspace(0, 1, 4096, device=dev).reshape(-1, 1).expand(-1, 3)
tv = TS.rqs(us, th.expand(4096, -1, -1), K)
mono = bool((tv.diff(dim=0) > -1e-9).all())
rng = bool((tv.min() >= -1e-6) and (tv.max() <= 1 + 1e-6))
chk("단조성", mono)
chk("치역 [0,1]", rng, f"[{float(tv.min()):.4f},{float(tv.max()):.4f}]")

# 3) 준불연속: 기울기 파라미터를 키우면 슬로프가 커진다
th_st = torch.zeros(1, 3, 3 * K + 1, device=dev)
th_st[..., 2 * K + 4] = 12.0                     # 가운데 매듭 기울기만 크게
tv2 = TS.rqs(us, th_st.expand(4096, -1, -1), K)
slope = float((tv2.diff(dim=0).max() / (1.0 / 4095)))
chk("가파른 슬로프", slope > 50, f"max slope={slope:.0f}x")

# 4) remap: 점이 자기 셀을 못 벗어난다 + 기울기가 θ 와 x 로 흐른다
n3 = torch.tensor([9, 9, 9])
lo, h = torch.zeros(3, device=dev), 0.125
x = (torch.rand(3000, 3, device=dev) * 0.999).requires_grad_(True)
thc = (torch.randn(512, P, device=dev) * 2.0).requires_grad_(True)
xr = TS.remap(x, lo, h, n3, thc, K)
ci = ((x.detach() - lo) / h).floor().clamp(0, 7)
cr = ((xr.detach() - lo) / h).floor().clamp(0, 7)
chk("셀 유지", bool((ci == cr).all() or
                  ((xr.detach() - lo) / h - ci).max() <= 1.0 + 1e-5))
xr.sum().backward()
chk("기울기 흐름", thc.grad is not None and x.grad is not None
    and bool(torch.isfinite(thc.grad).all()) and float(
        thc.grad.abs().sum()) > 0)

# 5) 연속성 벌점: 상수장 = 0, 요동장 > 0
tg = torch.zeros(8, 8, 8, P, device=dev)
chk("벌점 상수장=0", float(TS.cont_penalty(tg)) == 0.0)
chk("벌점 요동장>0", float(TS.cont_penalty(torch.randn_like(tg))) > 0)

# 6) RQS + SITReg 합성이 접히지 않는다: 표본점의 det J > 0
bnd = FALLBACK_BOUND_444 * h
dp = (torch.randn(9 ** 3, 3, device=dev) * 10.0)     # squash 가 상한을 지킨다
w = SITRegWarp(bnd, 2)


def full(q):
    qr = TS.remap(q, lo, h, n3, thc.detach(), K)
    return w.apply(qr, dp, lambda z, c: z + cubic_bspline_g2p(z, lo, h, n3, c))


# 셀 면에서 떨어진 내부점만 뽑는다 -- 면을 가로지르는 유한차분은 (허용된)
# 접선 불연속을 밟아 미분이 아니게 된다
_ci = torch.randint(0, 8, (200, 3), device=dev).float()
xs = (_ci + 0.05 + 0.9 * torch.rand(200, 3, device=dev)) * h
eps = 1e-4
J = torch.stack([(full(xs + eps * torch.eye(3, device=dev)[k])
                  - full(xs - eps * torch.eye(3, device=dev)[k]))
                 / (2 * eps) for k in range(3)], -1)
det = torch.linalg.det(J)
chk("합성 det J > 0", bool((det > 0).all()),
    f"min det={float(det.min()):.4f}")

# 7) ConvStepper rqs 헤드: 모양과 초기 근사 항등
net = ConvStepper(n_feat=16, hidden=32, depth=2, rqs_dim=P).to(dev)
cells = (7, 7, 7)
feat = torch.randn(343, 16, device=dev)
out = net(None, feat, 0.01, (8, 8, 8), cells=cells)
chk("헤드 모양", out[-1].shape == (343, P), f"{tuple(out[-1].shape)}")
u9 = torch.rand(1000, 3, device=dev)
th9 = out[-1][torch.randint(0, 343, (1000,), device=dev)].reshape(1000, 3, -1)
ide = (TS.rqs(u9, th9, K) - u9).abs().max()
chk("초기 근사 항등", ide < 0.05, f"max|T(u)-u|={float(ide):.2e}")

print("ALL-OK" if ok else "SOME-FAIL", flush=True)
raise SystemExit(0 if ok else 1)
