"""서브스텝을 바꿔도 지켜져야 하는 것과, 바뀌는 게 정상인 것을 가른다.

학습 안 된 망은 변위 출력이 정확히 0 이라 손잡이 혼합만 남는다:
    x2 = x + w(x) * dt * v_cmd,    w = (1-q^2)^2,  q = |x - 중심| / R

- **손잡이 입자** (q=0, w=1) 는 한 프레임에 정확히 frame_dt * v_cmd 를 간다.
  서브스텝 수와 무관해야 한다 -- dt 가 손잡이 명령까지 제대로 흐르는지는
  여기서 갈린다 (예전에 _CTRL_SCALE 을 빼먹어 틀렸던 자리다).
- **반경 안쪽 입자** 는 중심이 프레임 안에서 움직이므로 w 가 스텝마다 달라진다.
  서브스텝을 잘게 쪼갤수록 그 비선형을 더 정확히 푸는 것이라, 값이 달라지는
  것이 정상이다. 여기에 불변을 요구하면 안 된다.

  python exe/test_dt_invariant.py --dumps a.pt b.pt c.pt
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", nargs="+", required=True,
                help="AF_ROLL_DUMP 결과들. 첫 번째가 기준(서브스텝 1)")
ap.add_argument("--tol", type=float, default=1e-3,
                help="손잡이 입자 상대 어긋남 허용치 (기본 0.1%)")
ap.add_argument("--n_ctrl", type=int, default=2)
a = ap.parse_args()

base = torch.load(a.dumps[0], map_location="cpu", weights_only=False)
x0 = base["x0"].float()
P0 = base["pred"].float()
# 손잡이 입자는 w=1 이라 가장 많이 움직인다. 명령 크기와 정확히 같은 거리를
# 간 입자들이 그것이다 -- 색인이 덤프에 없으므로 이동량으로 집는다.
mv = (P0[-1] - x0).norm(dim=-1)
hid = mv.argsort(descending=True)[:a.n_ctrl]
ok = True
print(f"기준 {a.dumps[0]}: 입자 {mv.numel()}, 손잡이 이동 "
      f"{float(mv[hid].min()):.5f}~{float(mv[hid].max()):.5f}")
for p in a.dumps[1:]:
    P = torch.load(p, map_location="cpu", weights_only=False)["pred"].float()
    df = (P - P0).norm(dim=-1)[-1]
    rel_h = float(df[hid].max()) / float(mv[hid].max())
    rel_a = float(df.max()) / float(mv.max())
    good = rel_h < a.tol
    ok = ok and good
    print(f"  {p}: 손잡이 상대 어긋남 {100*rel_h:.4f}% "
          f"[{'OK' if good else 'FAIL'}]  |  전체 최대 {100*rel_a:.3f}% "
          f"(반경 가장자리의 가중치 비선형 -- 달라지는 것이 정상)")
print("ALL-OK" if ok else "SOME-FAIL")
raise SystemExit(0 if ok else 1)
