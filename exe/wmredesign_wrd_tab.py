"""절 CM38 진단 표 (로컬 CPU): wrd.jsonl (lib/anchorflow/fem_wmrd.py) -> 무너진 걸음 원인 가르기 표.
사용: python exe/wmredesign_wrd_tab.py <판 디렉토리>...  (디렉토리마다 wrd.jsonl·summary.json)"""
import json
import math
import os
import sys

import numpy as np


def load(d):
    R, E = [], []
    for ln in open(os.path.join(d, "wrd.jsonl")):
        o = json.loads(ln)
        (E if o.get("frame_end") else R).append(o)
    return R, E


def g(o, *k, dflt=float("nan")):
    for q in k:
        if not isinstance(o, dict) or q not in o:
            return dflt
        o = o[q]
    return o if o is not None else dflt


def med(v):
    v = [q for q in v if isinstance(q, (int, float)) and math.isfinite(q)]
    return float(np.median(v)) if v else float("nan")


def f(x, p=3):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "-"
    return f"{x:.{p}g}"


def main():
    rows = []
    for d in sys.argv[1:]:
        nm = os.path.basename(d.rstrip("/"))
        try:
            R, E = load(d)
        except FileNotFoundError:
            print(f"| {nm} | wrd 없음 |"); continue
        s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
        col = {}                                                                         # (fr, ks) -> 걸음 끝 ‖g‖²/F² (판정 커널 ufg)
        for e in E:
            for ks, u in enumerate(e.get("ufg") or []):
                col[(e["fr"], ks)] = u
        st = {}
        for o in R:
            if "err" in o:
                continue
            st.setdefault((o["fr"], o["ks"]), []).append(o)
        big = [k for k, u in col.items() if not (u <= 1e6)]
        ok = [k for k, u in col.items() if (u <= 1e6)]
        def agg(keys, fn):
            return med([fn(it) for k in keys for it in st.get(k, [])])
        def agg0(keys, fn):                                                              # 걸음 첫 반복만
            return med([fn(st[k][0]) for k in keys if k in st and st[k]])
        rows.append((nm, s, R, E, col, st, big, ok))
        print(f"\n### {nm}: 결과 {'발산 f' + str(s.get('diverged')) if s.get('diverged') else '완주 ' + str(s.get('frames_done'))}, 진단 걸음 {len(col)}, 무너진 걸음 (‖g‖²/F² > 1e6) {len(big)}: {sorted(big)[:30]}")
        print("| 걸음 묶음 | 첫 반복 ‖g‖/F | 첫 반복 ‖g‖/W (관성 척도) | F/W | PCG ‖r‖/‖r0‖ | δ 최대 칸 | δ>1 칸 질량 몫 | δ 최대 노드 질량/중앙 | 그 노드 대각 강성/(m/τ²) | δ 방향 곡률 비 차분/접선 (ε 1e-3 칸) | 차분·접선 cos | δ 방향 할선 곡률 비 (온 걸음) | 온 걸음 비유한 힘 노드 | 실제 보폭 ‖Δy‖/‖δ‖ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab, keys in (("무너진 걸음", big), ("안 무너진 걸음", ok)):
            if not keys:
                continue
            print(f"| {lab} ({len(keys)}) | {f(agg0(keys, lambda o: o['r_F']))} | {f(agg0(keys, lambda o: o['r_W']))} | {f(agg0(keys, lambda o: g(o, 'F0') / max(g(o, 'W0'), 1e-300)))} | "
                  f"{f(agg(keys, lambda o: g(o, 'lin_rel')))} | {f(agg(keys, lambda o: g(o, 'd_max')))} | {f(agg(keys, lambda o: g(o, 'd_gt1_mass')))} | "
                  f"{f(agg(keys, lambda o: g(o, 'd_argmax', 'mrel')))} | {f(agg(keys, lambda o: g(o, 'd_argmax', 'Bdiag') / max(g(o, 'd_argmax', 'Mi'), 1e-300)))} | "
                  f"{f(agg(keys, lambda o: -g(o, 'fd_e3', 'dJd') / g(o, 'fd_e3', 'dKd')))} | {f(agg(keys, lambda o: -g(o, 'fd_e3', 'cos')))} | {f(agg(keys, lambda o: -g(o, 'sec', 'dJs') / g(o, 'sec', 'dKd')))} | "
                  f"{f(agg(keys, lambda o: g(o, 'sec', 'nonfin')))} | {f(agg(keys, lambda o: g(o, 'step_norm')))} |")
        fk = sorted(big)[:2]                                                             # 처음 무너진 걸음 둘: 반복별
        for k in fk:
            print(f"\n{nm} 프레임 {k[0]} 걸음 {k[1]} (끝 ‖g‖²/F² {f(col[k])}):")
            print("| 반복 | ‖g‖ N | 관성 몫 ‖Mη/τ²‖ | 힘 몫 ‖Tf‖ | ‖g‖/F | ‖g‖/W | PCG ‖r‖/‖r0‖ | δ 최대 칸 | δ>1 칸 노드 · 질량 몫 | δ 최대 노드 질량/중앙 · 대각/(m/τ²) | 곡률 비 · cos (ε 1e-3) / cos (1e-5) | 할선 곡률 비 · 비유한 | 보폭 ‖Δy‖/‖δ‖ · 중앙 · 줄인 노드 몫 |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for o in st.get(k, []):
                print(f"| {o['it']} | {f(o['g'])} | {f(o['gI'])} | {f(o['gF'])} | {f(o['r_F'])} | {f(o['r_W'])} | {f(g(o, 'lin_rel'))} | {f(g(o, 'd_max'))} | {g(o, 'd_gt1', dflt='-')} · {f(g(o, 'd_gt1_mass'))} | "
                      f"{f(g(o, 'd_argmax', 'mrel'))} · {f(g(o, 'd_argmax', 'Bdiag') / max(g(o, 'd_argmax', 'Mi'), 1e-300))} | {f(-g(o, 'fd_e3', 'dJd') / g(o, 'fd_e3', 'dKd'))} · {f(-g(o, 'fd_e3', 'cos'))} / {f(-g(o, 'fd_e5', 'cos'))} | "
                      f"{f(-g(o, 'sec', 'dJs') / g(o, 'sec', 'dKd'))} · {g(o, 'sec', 'nonfin', dflt='-')} | {f(g(o, 'step_norm'))} · {f(g(o, 'step_med'))} · {f(g(o, 'step_frac_lt1'))} |")


if __name__ == "__main__":
    main()
