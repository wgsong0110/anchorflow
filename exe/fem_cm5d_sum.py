"""절 CM5 진단 요약: --diag 판 (r_ 시작점 ŷ, u_ 시작점 U) 의 프레임별 걸음 첫 뉴턴 |g|²·둘째 뉴턴 |g|²·보폭, 첫 nbad·첫 PCG 상한·첫 y0k>0 프레임."""
import json, os, sys
import numpy as np
R = sys.argv[1]
for t in sys.argv[2:]:
    for v in ("r", "u"):
        p = f"{R}/{v}_{t}/fem.jsonl"
        if not os.path.exists(p): continue
        rows = [json.loads(l) for l in open(p)]
        f = lambda c: next((r["t"] for r in rows if c(r)), None)
        fy = f(lambda r: any(r.get("y0k", [0])))
        fb = f(lambda r: r["nbad"] > 0); fc = f(lambda r: r["it"]["capP"] >= r["it"]["solves"])
        fg = f(lambda r: r["diag"]["it"][0]["g2"] > 1e2)          # 첫 접촉 걸음 (자유 낙하 |g|² ~1e-3)
        ok = [r for r in rows if r["err"] == r["err"] and fg is not None and r["t"] >= fg and (fb is None or r["t"] < fb)]
        g1 = [max(r["diag"]["it"][0]["g2"], r["diag"]["it"][2]["g2"]) for r in ok]
        g2 = [max(r["diag"]["it"][1]["g2"], r["diag"]["it"][3]["g2"]) for r in ok]
        E0 = [r["diag"]["it"][0]["E0"] for r in ok]
        st = [min(x["st"] for x in r["diag"]["it"]) for r in ok]
        print(f"{v}_{t}: 프레임 {len(rows)} 첫 접촉 {fg} 첫 y0k>0 {fy} 첫 PCG 상한 전부 {fc} 첫 nbad {fb} | 접촉 뒤·nbad 전 {len(ok)} 프레임: "
              f"첫 뉴턴 |g|² 중앙 {np.median(g1):.1e}, 둘째 뉴턴 |g|² 중앙 {np.median(g2):.1e} (최대 {max(g2):.1e}), 걸음 1 시작 E 중앙 {np.median(E0):.1e}, 보폭 < 1 프레임 {sum(s < 1 for s in st)}" if ok else f"{v}_{t}: {len(rows)} 프레임 (접촉 전 자료 없음)")
