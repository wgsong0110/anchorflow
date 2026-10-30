"""절 AD 표: 같은 설정 반복 (_r1.._rK) 을 한 줄로 (summary.json, fem.jsonl 의 sp_chk, period.json, NOT_TIMED/TIMED).
사용: python exe/fem_ad_table.py <dir>..."""
import collections
import json
import os
import re
import sys

G = collections.OrderedDict()
for d in sorted(sys.argv[1:]):
    if os.path.isdir(d):
        G.setdefault(re.sub(r"_r\d+$", "", os.path.basename(d)), []).append(d)
NM = {"vcap": "탄성 부피 상한", "none": "기준", "merge": "묶음 (면 이웃)", "cube": "묶음 (정육면체)", "blend": "F̄ 이웃 섞기", "stiff": "점유율 응력 가중", "mlow": "노드 관성 질량 하한"}


def how(s):
    a = s["args"]
    sp = a.get("sp", "none")
    p = [NM[sp] + ({"merge": f" k≤{a['sp_k']}", "cube": f" k≤{a['sp_k']}", "blend": f" k≤{a['sp_k']} β {a['sp_beta']:g}",
                    "stiff": f" k≤{a['sp_k']} w=min(점유율/{a['sp_wocc']:g},1)", "mlow": f" {a['sp_mlow']:g}x중앙", "vcap": f" min(V_T, {a.get('sp_vcap', 1):g} V_기하)"}.get(sp, ""))
         + (f" 점유율<{a['sp_occ']:g}" if a.get("sp_occ", 0) > 0 and sp in ("merge", "cube", "blend", "stiff") else "")]
    p.append(f"클램프 {s['lim_sr']:g}" if s.get("lim_sr") is not None else "클램프 끔")
    if a.get("fupd", "T") != "T":
        p.append(f"G2P F {a['fupd']} β {a.get('fupd_b', 1):g}")
    return " + ".join(p)


def chk(d):
    f = os.path.join(d, "fem.jsonl")
    out = []
    if os.path.exists(f):
        for ln in open(f):
            r = json.loads(ln)
            if isinstance(r.get("sp_chk"), dict):
                out.append(r["sp_chk"])
    return out


print("| 설정 | 방식 | N | 완주 | 발산 프레임 | 오차 (완주 평균; 회별, 발산이면 발산 전) | ms (참고/단독) | 폭주 노드 (프레임 끝 합 / 최대) | 제한 작동 프레임 / 노드x서브스텝 | 지운 KE 합 (J) | 희소 / 묶인 사면체 (중앙) | 검산 |Σf|/Σ|f| · |Σr×f|/Σ|r||f| 최대 | 주기 (s) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, ds in G.items():
    S = []
    for d in ds:
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        s["_timed"] = os.path.exists(os.path.join(d, "TIMED")) or not os.path.exists(os.path.join(d, "NOT_TIMED"))
        pj = os.path.join(d, "period.json")
        s["_per"] = json.load(open(pj))["T_zc"] if os.path.exists(pj) else None
        s["_chk"] = chk(d)
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
    ck = [c for s in S for c in s["_chk"]]
    cks = (f"{max(c['F_sum'] for c in ck):.1e} · {max(c['tq'] for c in ck):.1e}" + (f" (Σ(m/m')f {max(c['P_rate'] for c in ck):.1e})" if 'P_rate' in ck[0] else "")) if ck else "-"
    per = ", ".join(f"{s['_per']:.3f}" for s in S if s["_per"] is not None) or "-"
    spn = f"{av('sp_n_med'):.0f} / {av('sp_grouped_med'):.0f}" if s0.get("sp", "none") != "none" else "-"
    print(f"| {k} | {how(s0)} | {s0['N']} | {len(fin)}/{len(S)} | {', '.join(map(str, sorted(divs))) or '-'} | "
          f"{('%.2f %%' % (100 * err)) if err is not None else '-'} ({errs}) | {ms:.2f} ({tm}) | {av('rn_tot'):.0f} / {max((s.get('rn_max') or 0) for s in S)} | "
          f"{av('lim_frames'):.1f} / {av('lim_n_tot'):.0f} | {av('lim_dke_tot'):.3g} | {spn} | {cks} | {per} |")
