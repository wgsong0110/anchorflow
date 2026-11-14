"""절 CM172 집계 (CPU): 한 판의 cm170.jsonl (먼 쪽 잔차·선탐색 칸 sc1) 와 cm167.jsonl (점 상태 바뀜 pl) 을 프레임·반복별로 붙여 출력.
사용: python3 exe/cm172_an.py <판 디렉토리> [프레임 ...]   (R2 에서 받은 두 jsonl 이 든 디렉토리)
칸: SC[9] g·δ, SC[10] 탄성 E (현재), SC[14] 받은 보폭, SC[20] 관성 E (현재), SC[21+i] 후보 i (c = 2^-i) 관성 E, SC[40+i] 후보 탄성 E, SC[63] c=1 받음 표시. pl = [상태0 상태1 상태2 바뀜 탄성<->소성 꼬리<->꼭짓점] (cm167)."""
import json
import os
import sys

d = sys.argv[1]
want = {int(t) for t in sys.argv[2:]}
a = {json.loads(l)["f"]: json.loads(l) for l in open(os.path.join(d, "cm170.jsonl"))}
b = {}
if os.path.exists(os.path.join(d, "cm167.jsonl")):
    b = {json.loads(l)["f"]: json.loads(l) for l in open(os.path.join(d, "cm167.jsonl"))}
for f in sorted(a):
    if want and f not in want:
        continue
    print(f"== f{f}  it | 먼 쪽 잔차 | 관성 | g·δ | 받은 보폭 | E0 (관성+탄성) | E(c=1) | E(1/2) | c=1 받음 | 점 st0 st1 st2 | 바뀜 전체 (탄성<->소성 / 꼬리<->꼭짓점)")
    pl = {o["it"]: o for o in b.get(f, {}).get("its", [])}
    for o in a[f]["its"]:
        sc = o.get("scL") or [0.0] * 64
        e0 = sc[20] + sc[10]; e1 = sc[21] + sc[40]; e2 = sc[22] + sc[41]
        p = pl.get(o["it"], {}).get("pl")
        ps = f"{p[0]:7d} {p[1]:6d} {p[2]:6d} | {p[3]:6d} ({p[4]} / {p[5]})" if p else "-"
        print(f"  {o['it']:3d} | {o['resid']:+9.2f} | {o['inert_def']:+7.3f} | {sc[9]:+.3e} | {sc[14]:.3f} | {e0:.4e} | {e1:.4e} | {e2:.4e} | {sc[63]:.0f} | {ps}")
