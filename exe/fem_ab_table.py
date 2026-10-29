"""절 AB 표: 같은 설정 반복 (_r1.._r5) 을 한 줄로 (summary.json, fem.jsonl 의 fspread, period.json, NOT_TIMED/TIMED).
사용: python exe/fem_ab_table.py <dir>..."""
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
    f = s.get("fupd") or s["args"].get("fupd", "T")
    nmf = {"T": "기존 A_T F", "bar": "셀 평균 이완", "nbar": "노드 평균 F̄ 이완", "navg": "노드 평균 A", "polar": "극분해 신장 이완"}[f]
    p = [nmf + ("" if f == "T" else f" β {s['args'].get('fupd_b', 1.0):g}")]
    if s.get("lim_sr") is not None:
        p.append(f"클램프 σ1/σ3 ≤ {s['lim_sr']:g}")
    if s.get("lim") is not None and s["lim"] >= 0:
        p.append(f"det A ≥ {s['lim']:g}")
    return " + ".join(p)


def sprd(d):
    """마지막 (발산 전) fspread 기록: 전체 중앙 우리/정답, 1~2 개 셀 중앙 우리/정답."""
    f = os.path.join(d, "fem.jsonl")
    last = None
    if os.path.exists(f):
        for ln in open(f):
            r = json.loads(ln)
            if isinstance(r.get("fspread"), dict):
                last = (r["t"] + 1, r["fspread"])
    return last


def cqa(d):
    f = os.path.join(d, "fem.jsonl")
    if not os.path.exists(f):
        return None
    v = [json.loads(ln).get("cq_act", 0) for ln in open(f)]
    return sum(v) / max(len(v), 1)


print("| 설정 | 방식 | N | 완주 | 발산 프레임 | 오차 (완주 평균; 회별, 발산이면 발산 전) | ms (참고/측정) | 입자 F 뒤집힘 최대 | 제한 작동 프레임 / 노드x서브스텝 | 지운 KE 합 (J) | 이웃 벌어짐 중앙 우리/정답 (전체 · 1~2 개 셀, 마지막 기록 프레임) | 주기 (s) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
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
        s["_sp"] = sprd(d)
        s["_cq"] = cqa(d)
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
    sp = []
    for s in S:
        if s["_sp"]:
            t, r = s["_sp"]
            a_ = r.get("all", {})
            b_ = r.get("sparse", {})
            sp.append(f"f{t}: {a_.get('ours', [float('nan')])[0]:.3g}/{a_.get('gt', [float('nan')])[0]:.3g} · {b_.get('ours', [float('nan')])[0]:.3g}/{b_.get('gt', [float('nan')])[0]:.3g}")
    cq = [s["_cq"] for s in S if s["_cq"] is not None]
    per = ", ".join(f"{s['_per']:.3f}" for s in S if s["_per"] is not None) or "-"
    print(f"| {k} | {how(s0)} | {s0['N']} | {len(fin)}/{len(S)} | {', '.join(map(str, sorted(divs))) or '-'} | "
          f"{('%.2f %%' % (100 * err)) if err is not None else '-'} ({errs}) | {ms:.2f} ({tm}) | {max((s.get('Fp_inv_max') or 0) for s in S)} | "
          f"{av('lim_frames'):.1f} / {av('lim_n_tot'):.0f} | {av('lim_dke_tot'):.3g} | {'; '.join(sp) or '-'} | {per} |")
