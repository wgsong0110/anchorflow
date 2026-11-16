"""[final_fix_mpmbl] i-PG 원 구현 두 번 (비결정성 기준) vs 빠른 구현: 프레임별 입자 위치 차 (EXT 로 나눈 평균·최대) 와 오차.
  python exe/ipgfast_cmp.py --ref DIR_origA --rep DIR_origB --new DIR_fast"""
import argparse
import json
import os
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--ref"); ap.add_argument("--rep"); ap.add_argument("--new"); a = ap.parse_args()
L = {k: torch.load(os.path.join(d, "pred.pt"), weights_only=False)["pred"].double() for k, d in (("ref", a.ref), ("rep", a.rep), ("new", a.new))}
J = {k: json.load(open(os.path.join(d, "err.json"))) for k, d in (("ref", a.ref), ("rep", a.rep), ("new", a.new))}
EXT = J["ref"]["EXT"]
for f in range(min(v.shape[0] for v in L.values())):
    r = []
    for k in ("rep", "new"):
        dd = (L[k][f] - L["ref"][f]).norm(dim=-1) / EXT
        r.append(f"{k}: 평균 {dd.mean():.3e} 최대 {dd.max():.3e}")
    print(f"프레임 {f + 1}: " + " | ".join(r) + " | 오차 " + " ".join(f"{k} {100 * J[k]['err'][f]:.4f}%" for k in ("ref", "rep", "new")))
print("ms/프레임 " + " ".join(f"{k} {J[k]['ms']}" for k in ("ref", "rep", "new")))
