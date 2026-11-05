"""절 S.8 (1) 표: exe/stab_k4diag.sh 출력 폴더 -> 판마다 발산 프레임·오차·끝 해시, 처음 이상 프레임 (nbad·J), 그 프레임 이상 입자 위치 (바닥 위 높이·칸·관통 칸),
처음 이상 프레임 앞뒤 뉴턴 반복 진단 (선탐색 보폭 < 1 비율·g·δ ≥ 0·에너지 증가), 시작점 섞기 k (y0k) 분포.

  python exe/stab_k4diag_tab.py OUT > OUT/k4diag_table.md
"""
import glob
import json
import math
import os
import sys

O = sys.argv[1]


def f(x, k=2):
    return "" if x is None else (f"%.{k}f" % x if isinstance(x, float) else str(x))


rows = []
for d in sorted(glob.glob(f"{O}/*/summary.json"), key=os.path.getmtime):
    n = d.split("/")[-2]
    s = json.load(open(d))
    L = [json.loads(q) for q in open(d.replace("summary.json", "fem.jsonl")) if q.strip()]
    on_nb = next((q["t"] + 1 for q in L if q.get("nbad", 0) > 0), None)
    on_pd = next((q["t"] + 1 for q in L if (q.get("pd") or {}).get("nbad", 0) > 0), None)
    on_jn = next((q["t"] + 1 for q in L if (q.get("pd") or {}).get("nJneg", 0) > 0), None)
    pd0 = next((q["pd"] for q in L if (q.get("pd") or {}).get("nbad", 0) > 0), None)
    ls = dict(n=0, st_lt1=0, gd_pos=0, e_up=0)
    y0 = []
    for q in L:
        y0 += q.get("y0k") or []
        if on_pd is None or not (on_pd - 6 <= q["t"] + 1 <= on_pd + 2):
            continue
        for it in (q.get("diag") or {}).get("it", []):
            ls["n"] += 1
            st = it.get("st")
            if st is not None and st < 1.0:
                ls["st_lt1"] += 1
            if it.get("gd") is not None and it["gd"] >= 0:
                ls["gd_pos"] += 1
            Ec = it.get("Ec") or []
            if st is not None and Ec and st > 0:
                i = int(round(-math.log2(st))) if st > 0 else None
                if i is not None and i < len(Ec) and Ec[i] > it["E0"]:
                    ls["e_up"] += 1
    rows.append((n, s, on_nb, on_pd, on_jn, pd0, ls, y0, L))

print("# 절 S.8 (1) 우리 (B) lego L000 K 4 발산 진단 (exe/stab_k4diag.sh)\n")
print("| 판 | K | 발산 프레임 | 오차 % (끝난 프레임 평균) | 끝 해시 | 처음 유클리드 평균 점 (nbad) 프레임 | 처음 이상 입자 프레임 (J≤0.05·J>5) | 처음 J≤0 | 그 프레임 이상 입자 수 · 높이 (최소/중앙/최대, m) · 칸 (관통/아래) · 장벽 띠 안 | 앞 6 프레임 뉴턴 반복 수 · 보폭<1 · g·δ≥0 · 에너지 증가 | y0k 평균 (최소~최대) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n, s, on_nb, on_pd, on_jn, pd0, ls, y0, L in rows:
    a = s["args"]
    pdt = "" if not pd0 else f"{pd0['nbad']} · {'/'.join(f'{q:.4f}' for q in pd0['bad_h'])} · {pd0['bad_cells']} ({pd0['bad_cells_cross']}/{pd0['bad_cells_below']}) · {pd0.get('bad_in_band')}"
    lst = f"{ls['n']} · {ls['st_lt1']} · {ls['gd_pos']} · {ls['e_up']}" if ls["n"] else ""
    y0t = f"{sum(y0) / len(y0):.1f} ({min(y0)}~{max(y0)})" if y0 else ""
    print(f"| {n} | {a['K']} | {f(s.get('diverged'))} | {f(100 * s['err_mean_done'])} | {s.get('x_sha1')} | {f(on_nb)} | {f(on_pd)} | {f(on_jn)} | {pdt} | {lst} | {y0t} |")
print("\n## 처음 이상 프레임의 최악 입자 (J 최소)\n")
for n, s, on_nb, on_pd, on_jn, pd0, ls, y0, L in rows:
    if pd0:
        print(f"- {n}: 프레임 {on_pd} 최악 {pd0['worst']}, 이상 입자 중심 {[round(q, 4) for q in pd0['bad_ctr']]}")
print("\n## 프레임별 J 최소 · 이상 입자 수 · nbad (진단 판, 처음 이상 프레임 -5 .. 발산)\n")
for n, s, on_nb, on_pd, on_jn, pd0, ls, y0, L in rows:
    if not any("pd" in q for q in L):
        continue
    st = (on_pd or len(L)) - 5
    seq = [f"{q['t'] + 1}: {q['pd']['Jmin']:.3g}/{q['pd']['nbad']}/{q.get('nbad')}" for q in L if q["t"] + 1 >= st and "pd" in q]
    print(f"- {n}: " + ", ".join(seq[:30]))
print("\n## 바닥 위 최저 높이 (hmin, m) 와 장벽 걸린 표본점 수 (bar_cnt), 처음 이상 프레임 앞뒤\n")
for n, s, on_nb, on_pd, on_jn, pd0, ls, y0, L in rows:
    if not any("pd" in q for q in L) or on_pd is None:
        continue
    seq = [f"{q['t'] + 1}: {q['pd']['hmin']:.4f}/{(q.get('bc') or {}).get('neng')}" for q in L if on_pd - 4 <= q["t"] + 1 <= on_pd + 4 and "pd" in q]
    print(f"- {n}: " + ", ".join(seq))
