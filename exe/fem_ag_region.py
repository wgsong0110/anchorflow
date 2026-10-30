"""절 AG 진단: 롤아웃 오차를 영역별로 (입자 속도 구속 구역 / 나머지) + 구속 구역 평균 이동 (우리 vs 정답).
사용: python exe/fem_ag_region.py --gt bread.pt --dump roll_fem.pt --out region.json"""
import argparse
import json
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = torch.load(a.gt, map_location="cpu", weights_only=False)
r = torch.load(a.dump, map_location="cpu", weights_only=False)
x0 = d["x"][0].float(); EXT = float(r["EXT"])
P, G = r["pred"].float(), r["gt"].float()
regs = {}
for k, bc in enumerate([b for b in d["cfg"].get("boundary_conditions") or [] if b.get("type") == "enforce_particle_translation"]):
    regs[f"구속{k}"] = ((x0 - torch.tensor(bc["point"])).abs() < torch.tensor(bc["size"])).all(-1)
anyc = torch.zeros(x0.shape[0], dtype=torch.bool)
for m in regs.values():
    anyc |= m
regs["나머지"] = ~anyc
out = dict(frames=[])
for t in list(range(0, P.shape[0], max(1, P.shape[0] // 15))) + [P.shape[0] - 1]:
    e = (P[t] - G[t]).norm(dim=-1) / EXT
    rec = dict(t=t + 1, err=float(e.mean()))
    for nm, m in regs.items():
        rec[nm] = float(e[m].mean())
        rec[nm + "_dy_ours"] = [round(q, 4) for q in (P[t][m] - x0[m]).mean(0).tolist()]
        rec[nm + "_dy_gt"] = [round(q, 4) for q in (G[t][m] - x0[m]).mean(0).tolist()]
    out["frames"].append(rec)
    print(rec, flush=True)
out["n"] = {k: int(v.sum()) for k, v in regs.items()}
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
