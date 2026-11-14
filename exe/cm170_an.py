"""절 CM170 집계 (CPU): cm170.jsonl (먼 쪽 노드 뉴턴 반복별 평형 잔차) 를 프레임·반복별 표로. 사용: python3 exe/cm170_an.py <cm170.jsonl> [프레임 ...]
잔차 (m/s², 위 +) = 관성 (y−Ũ)/τ² 합 + Σ f_z (f = 에너지 기울기, 마지막 f 건드리는 커널 뒤 누적) 를 M 으로 나눈 것. 평형이면 0."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
want = {int(t) for t in sys.argv[2:]}
for r in rows:
    if want and r["f"] not in want:
        continue
    print(f"== f{r['f']} K={r['K']}  (먼 쪽 층 자유 노드, m/s², 위 +)")
    nm = r["its"][0]["names"] if r["its"] else []
    print("  it |  노드 | 관성 정의 | 관성 풀이 | " + " | ".join(f"f_z {i}:{n[:10]}" for i, n in enumerate(nm)) + " | 잔차 | 갱신 뒤 관성 | |g|²")
    for o in r["its"]:
        print(f"  {o['it']:3d} | {o['n']:6.0f} | {o['inert_def']:+9.3f} | {o['inert_slv']:+9.3f} | " + " | ".join(f"{v:+9.3f}" for v in o["fz"]) +
              f" | {o['resid']:+8.3f} | " + (f"{o['post_inert']:+9.3f}" if o["post_inert"] is not None else "   -   ") + f" | {o['g2']:.3e}")

    print("  it | PCG 횟수 | rz 처음 -> 끝 | 이 반복 PCG 해 δ 의 관성 합 (m/s²) | 잔차 | δ/잔차")
    for o in r["its"]:
        c = o.get("cg") or {}
        if o.get("dl_slv") is not None:
            print(f"  {o['it']:3d} | {c.get('n')} | {c.get('rz0', 0):.2e} -> {c.get('rz1', 0):.2e} | {o['dl_slv']:+9.4f} | {o['resid']:+8.3f} | {o['dl_slv'] / o['resid'] if o['resid'] else 0:+.4f}")
