"""dt 조건화(ConvStepper + DtFiLM)의 정합성 검사.

  python exe/test_dt_cond.py
"""
import math

import torch

from anchorflow.conv_stepper import ConvStepper

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DTR = 1.0 / 24.0
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}", flush=True)


def mk(**kw):
    torch.manual_seed(1)
    n = ConvStepper(n_feat=12, hidden=32, depth=3, h=0.05, scale=0.02,
                    dt_ref=DTR, **kw).to(dev)
    # 출력층이 0 초기화라 dp 가 항상 0 이면 무엇을 바꿔도 안 보인다. 조건화가
    # 실제로 흐르는지 보려면 출력층을 깨워야 한다.
    with torch.no_grad():
        n.out.weight.normal_(0, 0.1)
    return n


feat = torch.randn(343, 12, device=dev)
GRID, CELLS = (8, 8, 8), (7, 7, 7)


def run(n, dt):
    return n(None, feat, dt, GRID, cells=CELLS)[0]


# 1) dt_cond 끄면 dt 를 바꿔도 (기존 film 의 미미한 기여를 빼면) 거의 그대로
n0 = mk(dt_cond=False)
d0 = (run(n0, DTR) - run(n0, DTR / 8)).abs().max()
chk("조건화 off: dt 영향 거의 없음", d0 < 1e-3, f"max|Δdp|={float(d0):.2e}")

# 2) dt_cond 켜면 -- 0 초기화라 처음에는 **항등**이어야 한다 (옛 체크포인트 보호)
n1 = mk(dt_cond=True)
d1 = (run(n1, DTR) - run(n1, DTR / 8)).abs().max()
chk("조건화 on, 0 초기화: 아직 항등", d1 < 1e-6, f"max|Δdp|={float(d1):.2e}")
chk("켜기 전후 출력 동일", bool(torch.allclose(run(n0, DTR), run(n1, DTR),
                                          atol=1e-6)))

# 3) DtFiLM 을 학습된 상태로 흔들면 dt 가 출력을 바꾼다
with torch.no_grad():
    last = [m for m in n1.dtfilm.mlp.modules()
            if isinstance(m, torch.nn.Linear)][-1]
    last.weight.normal_(0, 0.05)
    last.bias.normal_(0, 0.05)
    n1.dtfilm._cache.clear()
d2 = (run(n1, DTR) - run(n1, DTR / 8)).abs().max()
r2 = float(d2) / float(run(n1, DTR).abs().max())
chk("학습 후: dt 가 출력을 바꾼다", r2 > 0.05, f"상대차 {100*r2:.1f}%")
# 같은 dt 는 늘 같은 값 (캐시가 값을 섞지 않는다)
chk("같은 dt 는 재현", bool(torch.allclose(run(n1, DTR / 8),
                                       run(n1, DTR / 8), atol=0)))

# 4) 단조로운 두 자리 범위에서 전부 유한
vals = [float(run(n1, DTR / k).abs().max()) for k in (1, 2, 4, 8, 16, 104)]
chk("두 자리 dt 범위에서 유한", all(math.isfinite(q) for q in vals),
    " ".join(f"{q:.3f}" for q in vals))

# 5) dt_scale: 변위가 dt 에 정확히 비례
n2 = mk(dt_cond=True, dt_scale=True)
with torch.no_grad():
    last = [m for m in n2.dtfilm.mlp.modules()
            if isinstance(m, torch.nn.Linear)][-1]
    last.weight.zero_(); last.bias.zero_()      # FiLM 은 항등으로 두고 비례만 본다
    n2.dtfilm._cache.clear()
r = run(n2, DTR / 4) / run(n2, DTR).clamp(min=1e-12)
chk("dt_scale: dp ∝ dt", bool(torch.allclose(run(n2, DTR / 4),
                                             run(n2, DTR) / 4, atol=1e-7)),
    f"비 중앙값 {float(r.median()):.4f} (기대 0.25)")

# 6) state_dict 왕복: 조건화 모듈이 저장·복원된다
sd = n1.state_dict()
n3 = mk(dt_cond=True)
miss, unex = n3.load_state_dict(sd, strict=False)
chk("state_dict 왕복", not miss and not unex, f"miss {len(miss)} unex {len(unex)}")
# 조건화 없는 옛 체크포인트를 조건화 켠 망에 얹으면 dtfilm 만 빠진다
miss2, unex2 = mk(dt_cond=True).load_state_dict(n0.state_dict(), strict=False)
chk("옛 체크포인트 호환", (not unex2)
    and all(k.startswith("dtfilm.") for k in miss2),
    f"빠진 키 {len(miss2)} 개 전부 dtfilm.*")

print("ALL-OK" if ok else "SOME-FAIL", flush=True)
raise SystemExit(0 if ok else 1)
