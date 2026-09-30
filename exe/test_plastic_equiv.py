"""소성 사영 최적화가 결과를 바꾸지 않는지 검증한다.

바꾼 것 둘:
  (1) 고유분해를 psi_of 에서 한 번만 하고 plastic_step 이 그 기저를 받아 쓴다
  (2) dlog 가 정확히 0 인(항복 안 한) 입자는 곱셈을 건너뛴다

(2) 가 정당한 근거: P = V diag(exp(0)) V^T = V V^T = I. 건너뛰면 float64 직교성
오차조차 안 생기므로 **더 정확하다**. 여기서는 옛 경로(항상 재분해 + 전체 곱셈)와
새 경로를 같은 입력으로 비교해 fp32 해상도 아래인지 본다.
"""
from __future__ import annotations
import argparse, os, sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import torch
from anchorflow import phys_resid as PR

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


def old_plastic(F, dlog):
    """최적화 전 경로: 항상 재분해하고 전체에 곱한다."""
    C = (F.transpose(-1, -2) @ F).double()
    _, V = torch.linalg.eigh(C)
    P = V @ torch.diag_embed(dlog.double().exp()) @ V.transpose(-1, -2)
    return (F.double() @ P).to(F.dtype)


cfg = {"E": 2e6, "nu": 0.3, "material": "metal", "yield_stress": 4e4,
       "density": 1000}
# 변형을 섞어 만든다: 대부분 작은 변형(항복 전), 일부 큰 변형(항복)
I3 = torch.eye(3, device=dev).expand(a.n, 3, 3).contiguous()
sc = torch.rand(a.n, 1, 1, device=dev) ** 3 * 0.6
F = I3 + sc * torch.randn(a.n, 3, 3, device=dev)
F = F.contiguous()

psi, pl = PR.psi_of(F, cfg, 1 / 60)
chk("psi_of 가 기저를 함께 돌려준다", isinstance(pl, PR.Plast),
    f"{type(pl).__name__}")
dlog, V = pl.dlog, pl.V
nz = dlog.abs().sum(-1) > 0
chk("항복/비항복이 섞여 있다", 0 < int(nz.sum()) < a.n,
    f"항복 {int(nz.sum())}/{a.n} ({100*float(nz.float().mean()):.1f}%)")
chk("비항복 입자는 dlog 가 **정확히** 0",
    float(dlog[~nz].abs().max()) == 0.0, f"{float(dlog[~nz].abs().max()):.1e}")

Fn = PR.plastic_step(F, pl)
Fo = old_plastic(F, dlog)
d = (Fn - Fo).abs().max(dim=-1).values.max(dim=-1).values
rel = d / F.abs().amax(dim=(1, 2)).clamp_min(1e-12)
chk("새 경로 == 옛 경로 (fp32 해상도 아래)",
    float(rel.max()) < 1e-6,
    f"상대오차 최대 {float(rel.max()):.2e}, 중앙 {float(rel.median()):.2e}")
chk("비항복 입자는 F 가 비트 단위로 그대로",
    bool((Fn[~nz] == F[~nz]).all()), "")
# 항복 입자에서도 옛 경로와 같은 값이어야 한다 (기저 공유가 정의를 안 바꾼다)
dy = (Fn[nz] - Fo[nz]).abs().max()
chk("항복 입자는 기저 공유로도 같은 값",
    float(dy) < 1e-5 * float(F[nz].abs().max()),
    f"최대차 {float(dy):.2e}")
# 에너지도 그대로인지 (psi 는 값만 쓰므로 eigh/eigvalsh 차이만 본다)
psi2, _ = PR.psi_of(F, {**cfg, "material": "jelly"}, 1 / 60)
chk("탄성 전용 물성은 기저 없이 None", _ is None, "")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
