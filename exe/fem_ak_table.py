"""절 AK 표: R2 (로컬 마운트) lbours/<root>/<태그>/fem.jsonl 에서 오차·완주·몸 높이·잔류 변형·에너지 수지 요약.
사용: python exe/fem_ak_table.py <root> [태그 정규식] [--frames a,b,c]"""
import glob
import json
import math
import os
import re
import sys

R = os.path.expanduser("~/r2/result/anchorflow/lbours/")
root = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
fr = [int(q) for q in sys.argv[sys.argv.index("--frames") + 1].split(",")] if "--frames" in sys.argv else []
rows = []
for d in sorted(glob.glob(R + root + "/*/")):
    tg = os.path.basename(d.rstrip("/"))
    if pat and not pat.search(tg):
        continue
    f = d + "fem.jsonl"
    if not os.path.exists(f):
        continue
    rs = []
    for ln in open(f):
        try:
            rs.append(json.loads(ln))
        except Exception:
            pass
    if not rs:
        continue
    sm = json.load(open(d + "summary.json")) if os.path.exists(d + "summary.json") else {}
    errs = [q["err"] for q in rs if q.get("err") is not None and math.isfinite(q["err"])]
    div = sm.get("diverged")
    em = 100 * sum(errs) / max(len(errs), 1)
    rg = [q["rigid"] for q in rs if "rigid" in q]
    ak = [(q["t"] + 1, q["ak"]) for q in rs if isinstance(q.get("ak"), dict)]
    s = f"{tg:34s} {len(rs):3d}{'/발산' + str(div) if div else '     '} 오차 {em:6.2f}%"
    if rg:
        mn = min(r[0][1] for r in rg)
        s += f" | 몸높이 최저·끝 {mn:.3f}·{rg[-1][0][1]:.3f} / 정답 {min(r[1][1] for r in rg):.3f}·{rg[-1][1][1]:.3f} 잔류 {rg[-1][0][0]:.1f}/{rg[-1][1][0]:.1f}% 질량중심z {rg[-1][0][2]:.3f}/{rg[-1][1][2]:.3f}"
    if ak:
        lp2g = sum(a["ke_p0"] - a["ke_n0"] for _, a in ak)
        lg2p = sum(a["ke_n1"] - a["ke_p1"] for _, a in ak)
        dsub = sum((a["ke_n1"] + a["ec1"] + a["pe_n1"]) - (a["ke_n0"] + a["ec0"] + a["pe_n0"]) for _, a in ak)
        dre = sum(a["ec1"] for _, a in ak[:-1]) - sum(a["ec0"] for _, a in ak[1:])   # 셀 탄성: 프레임 끝 -> 다음 프레임 P2G 재평균
        lst = ak[-1][1]
        s += f" | 합: P2G KE손실 {lp2g:.3g} G2P {lg2p:.3g} 서브스텝 ΔE {dsub:.3g} 셀탄성 재평균 {dre:.3g} J | 끝 입자탄성 쓰는F {lst['ep1']:.3g} 호환F {lst['ep_c']:.3g} 정답 {lst['ep_gt']:.3g}"
        for t, a in ak:
            if t in fr:
                s += (f"\n    f{t}: KE p0 {a['ke_p0']:.3g} n0 {a['ke_n0']:.3g} n1 {a['ke_n1']:.3g} p1 {a['ke_p1']:.3g} 정답 {a['ke_gt']:.3g} | 셀탄성 {a['ec0']:.3g}->{a['ec1']:.3g}"
                      f" | 입자탄성 {a['ep0']:.3g}->{a['ep1']:.3g} (B0 {a['ep_b0']:.3g}, 호환 {a['ep_c']:.3g}, 정답 {a['ep_gt']:.3g}) | PE p {a['pe_p1']:.4g} 정답 {a['pe_gt']:.4g}"
                      f" | σmin 1/10/50% 쓰는 {a['smin']['used']} 호환 {a['smin']['compat']} 정답 {a['smin']['gt']} | |F-F_c| {a['dF_used_compat']}")
    rows.append(s)
print("\n".join(rows))
