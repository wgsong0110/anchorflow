"""[final_pa_bl] 시간 구간 cuboid 풀림 확인: 덤프 (pred.pt {pred [L,P,3]} 또는 텐서) 에서 처음 위치가 구간 상자 안인 입자의 평균 z 를 프레임별로 정답과 나란히.

  python exe/fpabl_check_release.py --data pillow2sofa.pt --pred RUN/pred.pt [--pred ...]
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--pred", nargs="+", required=True)
a = ap.parse_args()
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
x0 = d["x"][0].float()
boxes = [(b, float(b.get("start_time", 0)), float(b.get("end_time", 1e3))) for b in cfg["boundary_conditions"] if b.get("type") == "cuboid" and float(b.get("end_time", 1e3)) < 100]
for p in a.pred:
    q = torch.load(p, map_location="cpu", weights_only=False)
    P = q["pred"] if isinstance(q, dict) else q
    L = P.shape[0]
    print(f"== {p} 프레임 {L}")
    for (b, t0, t1) in boxes:
        m = ((x0 - torch.tensor(b["point"])).abs() < torch.tensor(b["size"])).all(-1)
        fe = round(t1 / h)
        fr = sorted({1, max(1, fe - 2), fe, fe + 3, fe + 6, fe + 10, L} & set(range(1, L + 1)))
        print(f"  상자 z {b['point'][2]} 구간 [{t0}, {t1}) s = 프레임 {fe} 에서 풀림, 입자 {int(m.sum())}, 처음 평균 z {float(x0[m, 2].mean()):.4f}")
        for f in fr:
            print(f"    프레임 {f:4d} (t={f * h:.2f}s): 예측 z {float(P[f - 1][m, 2].mean()):.4f}  정답 z {float(d['x'][f][m, 2].mean()):.4f}")
