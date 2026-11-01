"""절 AN 진단 표: fem.jsonl 의 diag -> 프레임·걸음마다 뉴턴 반복 열 (|g|/|g0| 반복 시작, 보폭, PCG 반복, PCG 최종 상대 잔차, 란초스 λmin·λmax) 과 걸음 끝 |g|/|g0|."""
import json
import os
import sys

R = sys.argv[1]
for l in open(os.path.join(R, "fem.jsonl")):
    r = json.loads(l)
    for k, st in enumerate(r.get("diag", [])):
        print(f"프레임 {r['t'] + 1} 걸음 {k + 1} 오차 {100 * r['err']:.3f}% 고착 노드 {r.get('stuck')} 걸음 끝 |g|/|g0| {st.get('grf')} 음의 곡률 중단 {st.get('negc')}")
        for i, it in enumerate(st["it"]):
            res = it.get("res")
            print(f"  뉴턴 {i + 1}: |g|/|g0| {it['g']:.3e}, 보폭 {it.get('st')}, PCG {it['cg']} (잔차 {res:.1e}), 란초스 λ {it['lmin']:.3g} 에서 {it['lmax']:.3g} (비 {it['lmax'] / max(it['lmin'], 1e-300):.3g}), ΔE {it.get('dE')}")
