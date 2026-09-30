"""CD/EMD 구현을 알려진 답으로 검사한다. 값이 나온다는 것만으로는 모자란다.

  같은 점집합      -> CD = 0, EMD = 0
  평행이동 t       -> CD = t^2 (제곱거리 평균), EMD = t
  부분표본은 같은 색인으로 뽑아야 한다 (따로 뽑으면 표본 바닥이 오차를 덮는다)
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, types

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = a.dev
torch.manual_seed(0)
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


# train_deform 의 chamfer/emd 를 argparse 를 태우지 않고 꺼낸다
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "train_deform.py")).read()
i = src.index("def chamfer(")
j = src.index("\ndef ", src.index("def emd(") + 5)
body = src[i:j]
mod = types.ModuleType("m")
mod.__dict__.update(torch=torch, dev=dev, a=argparse.Namespace(cd_pts=2048))
exec(compile(body, "metrics", "exec"), mod.__dict__)
chamfer, emd = mod.chamfer, mod.emd

p = torch.rand(a.n, 3, device=dev)
chk("같은 점집합 CD = 0", abs(float(chamfer(p, p.clone()))) < 1e-10,
    f"{float(chamfer(p, p.clone())):.2e}")
chk("같은 점집합 EMD = 0", abs(float(emd(p, p.clone(), 0))) < 1e-6,
    f"{float(emd(p, p.clone(), 0)):.2e}")

for t in (0.01, 0.1):
    sh = torch.tensor([t, 0.0, 0.0], device=dev)
    cd = float(chamfer(p, p + sh))
    em = float(emd(p, p + sh, 1))
    # CD 는 제곱거리 평균이라 t^2 이 상한이다 (양쪽 최근접이 더 가까울 수 있어
    # 조밀한 구름에서는 그보다 작다). EMD 는 최적수송이라 t 에 가깝다.
    chk(f"평행이동 {t}: CD <= t^2", cd <= t * t * 1.05 + 1e-12,
        f"CD {cd:.3e} vs t^2 {t*t:.3e}")
    chk(f"평행이동 {t}: EMD ~ t", abs(em - t) / t < 0.25,
        f"EMD {em:.4f} vs t {t:.4f}")

# 대칭성과 단조성
q = torch.rand(a.n, 3, device=dev)
chk("CD 대칭", abs(float(chamfer(p, q)) - float(chamfer(q, p))) < 1e-8,
    f"{float(chamfer(p,q)):.4e} vs {float(chamfer(q,p)):.4e}")
c1 = float(chamfer(p, p + torch.tensor([0.01, 0, 0], device=dev)))
c2 = float(chamfer(p, p + torch.tensor([0.05, 0, 0], device=dev)))
chk("CD 가 거리에 단조", c2 > c1, f"{c1:.3e} -> {c2:.3e}")
e1 = float(emd(p, p + torch.tensor([0.01, 0, 0], device=dev), 2))
e2 = float(emd(p, p + torch.tensor([0.05, 0, 0], device=dev), 2))
chk("EMD 가 거리에 단조", e2 > e1, f"{e1:.4f} -> {e2:.4f}")
# 같은 씨앗이면 같은 부분표본 -> 재현된다
chk("EMD 가 같은 씨앗에서 재현된다",
    abs(float(emd(p, q, 5)) - float(emd(p, q, 5))) < 1e-9, "")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
