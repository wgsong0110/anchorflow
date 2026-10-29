"""절 Z 표: 같은 설정 반복 (_r1.._r5) 을 한 줄로 (summary.json, period.json, NOT_TIMED).
사용: python exe/fem_z_table.py <dir>..."""
import collections
import json
import os
import re
import sys

G = collections.OrderedDict()
for d in sorted(sys.argv[1:]):
    if os.path.isdir(d):
        G.setdefault(re.sub(r"_r\d+$", "", os.path.basename(d)), []).append(d)


def how(s):
    p = []
    if s.get("lim_sr"):
        p.append(f"클램프 r ≤ {s['lim_sr']:g}")
    if s.get("lim") is not None and s["lim"] >= 0:
        p.append(f"det A ≥ {s['lim']:g}")
    if s.get("sb"):
        p.append(f"장벽 κ {s['sb']:g} c {s['sb_c']:g} d̂ {s['sb_dhat']:g}")
    return " + ".join(p) or "기준 (제한 없음)"


print("| 설정 | 방식 | N | 완주 | 발산 프레임 | 오차 (완주 평균; 회별, 발산이면 발산 전) | ms (참고/측정) | 제한 작동 프레임 / 노드x서브스텝 (회 평균) | 지운 KE 합 (J) / Σ|Δp| (kg m/s) | 비율 위반 / det≤0 (사면체x서브스텝) | 장벽 활성 (사면체x서브스텝) | 끝 σ1/σ3(A) 최대 | 뒤집힘 사면체x서브스텝 | 주기 (s) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, ds in G.items():
    S = []
    for d in ds:
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        s["_timed"] = not os.path.exists(os.path.join(d, "NOT_TIMED"))
        pj = os.path.join(d, "period.json")
        s["_per"] = json.load(open(pj))["T_zc"] if os.path.exists(pj) else None
        S.append(s)
    if not S:
        print(f"| {k} | 요약 없음 |")
        continue
    s0 = S[0]
    fin = [s for s in S if not s["diverged"]]
    divs = [s["diverged"] for s in S if s["diverged"]]
    err = (sum(s["err_mean"] for s in fin) / len(fin)) if fin else None
    errs = ", ".join(f"{100 * (s['err_mean'] if s['err_mean'] is not None else s['err_mean_done']):.2f}" for s in S)
    ms = sum(s["ms_mean"] for s in S) / len(S)
    tm = "단독" if all(s["_timed"] for s in S) else "참고"
    av = lambda key: sum((s.get(key) or 0) for s in S) / len(S)
    srm = [s.get("srA_max") for s in S if s.get("srA_max") is not None]
    per = ", ".join(f"{s['_per']:.3f}" for s in S if s["_per"] is not None) or "-"
    print(f"| {k} | {how(s0)} | {s0['N']} | {len(fin)}/{len(S)} | {', '.join(map(str, sorted(divs))) or '-'} | "
          f"{('%.2f %%' % (100 * err)) if err is not None else '-'} ({errs}) | {ms:.2f} ({tm}) | {av('lim_frames'):.1f} / {av('lim_n_tot'):.0f} | "
          f"{av('lim_dke_tot'):.3g} / {av('lim_dpabs_tot'):.3g} | {av('sr_viol_tot'):.0f} / {av('sr_inv_tot'):.0f} | {av('sb_act_tot'):.0f} | "
          f"{(min(srm) if srm else float('nan')):.3g}~{(max(srm) if srm else float('nan')):.3g} | {av('inv_sub_tot'):.0f} | {per} |")
