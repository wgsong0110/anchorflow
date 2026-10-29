"""절 R: 스윕 결과 표 -- 같은 설정 반복 (_r1·_r2·_r3) 을 한 줄로 묶는다 (summary.json + period.json, NOT_TIMED 표식).
사용: python exe/fem_r_table.py <dir>...      (반복 없는 태그는 그대로 한 줄)"""
import collections
import json
import os
import re
import sys

G = collections.OrderedDict()
for d in sys.argv[1:]:
    if not os.path.isdir(d):
        continue
    b = os.path.basename(d)
    k = re.sub(r"_r\d+$", "", b)
    G.setdefault(k, []).append(d)


def f3(x, fmt):
    return "-" if x is None else fmt % x


print("| 설정 | 장벽 κ / Ĵ / J | 제한 det A ≥ / 속도 / 방식 | N | 완주 | 발산 프레임 | 오차 (완주 평균; 없으면 발산 전) | ms 평균 (측정) | 뒤집힘 사면체x서브스텝 (회별) | 끝 뒤집힘 최대 | 입자 F 뒤집힘 최대 | det A 최소 | 제한 노드x서브스텝 / 줄인 KE (J) | 최저 입자 (m) / 아래 최대 | 주기 (s) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, ds in G.items():
    S = []
    for d in sorted(ds):
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            s = json.load(open(f))
            s["_timed"] = not os.path.exists(os.path.join(d, "NOT_TIMED"))
            pj = os.path.join(d, "period.json")
            s["_per"] = json.load(open(pj))["T_zc"] if os.path.exists(pj) else None
            S.append(s)
        else:
            S.append(None)
    ok = [s for s in S if s is not None]
    if not ok:
        print(f"| {k} | 요약 없음 (실패) |")
        continue
    s0 = ok[0]
    fin = [s for s in ok if not s["diverged"]]
    divs = [s["diverged"] for s in ok if s["diverged"]]
    err = (sum(s["err_mean"] for s in fin) / len(fin)) if fin else min(s["err_mean_done"] for s in ok)
    errs = ", ".join(f"{100 * (s['err_mean'] if s['err_mean'] is not None else s['err_mean_done']):.2f}" for s in ok)
    ms = sum(s["ms_mean"] for s in ok) / len(ok)
    tm = "단독" if all(s["_timed"] for s in ok) else "비측정"
    fb = f"{s0.get('fb', 0):g} / {s0.get('fb_jhat', 0.3):g} / {s0.get('fb_on', 'X')}" if s0.get("fb", 0) > 0 else "-"
    lm = f"{s0.get('lim'):g} / {s0.get('lim_v')} / {s0.get('lim_mode', 'gs')}" if (s0.get("lim") is not None and s0.get("lim") >= 0) else "-"
    inv = "/".join(str(s.get("inv_sub_tot")) for s in ok)
    ie = max((s.get("inv_end_max") or 0) for s in ok)
    fpi = max((s.get("Fp_inv_max") or 0) for s in ok)
    dmin = min((s["detA_min"] for s in ok if s.get("detA_min") is not None), default=None)
    lim = f"{sum(s.get('lim_n_tot') or 0 for s in ok) / len(ok):.0f} / {sum(s.get('lim_dke_tot') or 0 for s in ok) / len(ok):.3g}" if lm != "-" else "-"
    smin = f"{min(s['s_min'] for s in ok):+.5f} / {max(s['n_below_max'] for s in ok)}" if s0.get("s_min") is not None else "-"
    per = ", ".join(f"{s['_per']:.3f}" for s in ok if s["_per"] is not None) or "-"
    print(f"| {k} | {fb} | {lm} | {s0['N']} | {len(fin)}/{len(S)} | {', '.join(map(str, divs)) or '-'} | {100 * err:.3f} % ({errs}) | {ms:.2f} ({tm}) | {inv} | {ie} | {fpi} | "
          f"{f3(dmin, '%.3g')} | {lim} | {smin} | {per} |")
