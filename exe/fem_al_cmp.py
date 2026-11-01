"""절 AL: 두 실행의 프레임별 오차 비교 (fem.jsonl err) + 구간 시간 (sec_*) 평균. 사용: python exe/fem_al_cmp.py <기준 디렉토리> <비교 디렉토리> ..."""
import json
import sys


def load(d):
    return [json.loads(l) for l in open(f"{d}/fem.jsonl")]


ref = load(sys.argv[1])
for d in sys.argv[1:]:
    r = load(d)
    n = min(len(r), len(ref))
    e = [q["err"] for q in r]
    dif = [abs(r[i]["err"] - ref[i]["err"]) for i in range(n)]
    rel = max(dif[i] / max(ref[i]["err"], 1e-12) for i in range(n) if ref[i]["err"] > 1e-4)
    ms = [q["sec"] * 1e3 for q in r[2:]]
    sec = {k: sum(q[k] for q in r[2:] if k in q) / max(1, sum(1 for q in r[2:] if k in q)) for k in r[-1] if k.startswith("sec_")}
    print(f"{d.rstrip('/').split('/')[-1]}: 프레임 {len(r)} 평균 오차 {100 * sum(e) / len(e):.4f}% 마지막 {100 * e[-1]:.4f}% | 기준 대비 프레임별 |Δ오차| 최대 {100 * max(dif):.4f}%p, 상대 최대 {100 * rel:.3f}% | "
          f"프레임 {sum(ms) / len(ms):.1f} ms (중앙 {sorted(ms)[len(ms) // 2]:.1f}) " + " ".join(f"{k[4:]} {v:.1f}" for k, v in sec.items()))
