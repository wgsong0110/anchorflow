"""절 AI 표: R2 (또는 노드 /tmp) lbours/fem_ai_{lego,ficus}/<태그>/{summary.json, fem.jsonl} -> 마크다운.
태그 = <씬>_<A><B>_<θ>_<클램프>_r<회>. 사용: python exe/fem_ai_table.py <root 디렉토리> [태그 접두 필터...] > table.md
칸: 완주 / 발산 프레임 / 오차 (완주 평균) / ms (비측정 참고) / 클램프 작동 노드x서브스텝 (끔이면 0 이어야) /
    프레임 시작 셀 응력 ‖τ‖ (쓰는 셀 모델) 99.9 % 분위·최대 ÷ 같은 프레임 정답 입자 ‖τ‖ 최대 (기록 프레임 중 최대) / 입자 1~2 개 셀 최대 ÷ 정답 최대 /
    정답 최대 넘는 셀 수 (기록 프레임 중 최대) / 입자 F 이웃 퍼짐 중앙 우리÷정답 (전체 · 입자 1~2 개 셀, 프레임 40 과 마지막 기록).
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = sys.argv[1]
FILT = sys.argv[2:]


def load(dd):
    try:
        s = json.load(open(os.path.join(dd, "summary.json")))
    except Exception:
        return None
    recs = []
    for ln in open(os.path.join(dd, "fem.jsonl")):
        try:
            recs.append(json.loads(ln))
        except Exception:
            pass
    return s, recs


def stat(recs):
    r = dict(sc_q=0.0, sc_max=0.0, sc12=0.0, hot=0, hot2=0, gt=None, sp40=None, splast=None, ai_frames=0, tau_used_q999=[], euc_rs=None)
    for rc in recs:
        ai = rc.get("ai")
        if isinstance(ai, dict):
            g = ai["hot"]["gt_max"]
            if not g > 0:
                continue
            tc = ai["tau_cell"]["used"]
            q = tc["all"]
            r["sc_q"] = max(r["sc_q"], q[3] / g)
            r["sc_max"] = max(r["sc_max"], q[4] / g)
            m12 = max([tc[k][4] for k in ("c1", "c2") if tc.get(k)] or [0.0])
            r["sc12"] = max(r["sc12"], m12 / g)
            r["hot"] = max(r["hot"], ai["hot"]["n"])
            r["hot2"] = max(r["hot2"], ai["hot"]["n_2x"])
            r["ai_frames"] += 1
        fs = rc.get("fspread")
        if isinstance(fs, dict) and "all" in fs:
            v = (fs["all"]["ours"][0] / max(fs["all"]["gt"][0], 1e-12), fs["sparse"]["ours"][0] / max(fs["sparse"]["gt"][0], 1e-12) if "sparse" in fs else float("nan"))
            if rc["t"] + 1 == 40:
                r["sp40"] = v
            r["splast"] = (rc["t"] + 1, v)
    return r


rows = defaultdict(list)
for dd in sorted(glob.glob(os.path.join(ROOT, "*_r*"))):
    tg = os.path.basename(dd)
    if FILT and not any(tg.startswith(f) for f in FILT):
        continue
    L = load(dd)
    if L is None:
        continue
    s, recs = L
    key = tg.rsplit("_r", 1)[0]
    rows[key].append((tg, s, recs))


def f2(x, n=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


print("| 판 | 완주 | 발산 프레임 | 오차 % (완주 평균 / 회별) | ms (참고) | 클램프 작동 | 셀 ‖τ‖ 99.9 % · 최대 ÷ 정답 최대 | 1~2 개 셀 최대 ÷ 정답 | 정답 최대 넘는 셀 (2 배) | 퍼짐 우리÷정답 f40 (전체 · 희소) | 마지막 기록 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(rows):
    rr = rows[key]
    nd = sum(1 for _, s, _ in rr if s.get("diverged") is None)
    dv = [str(s["diverged"]) for _, s, _ in rr if s.get("diverged") is not None]
    ef = [s["err_mean"] for _, s, _ in rr if s.get("diverged") is None and s.get("err_mean") is not None]
    eall = [100 * (s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done", float("nan"))) for _, s, _ in rr]
    ms = [s.get("ms_mean") for _, s, _ in rr if s.get("ms_mean")]
    lim = [s.get("lim_n_tot") or 0 for _, s, _ in rr]
    st = [stat(rc) for _, _, rc in rr]
    sc_q = max(x["sc_q"] for x in st)
    sc_m = max(x["sc_max"] for x in st)
    sc12 = max(x["sc12"] for x in st)
    hot = max(x["hot"] for x in st)
    hot2 = max(x["hot2"] for x in st)
    sp40 = [x["sp40"] for x in st if x["sp40"]]
    spl = [x["splast"] for x in st if x["splast"]]
    sp40s = f"{sum(v[0] for v in sp40) / len(sp40):.1f} · {sum(v[1] for v in sp40) / len(sp40):.1f}" if sp40 else "-"
    spls = f"f{spl[0][0]} {spl[0][1][0]:.1f} · {spl[0][1][1]:.1f}" if spl else "-"
    emean = f"{100 * sum(ef) / len(ef):.2f}" if ef else "-"
    print(f"| {key} | {nd}/{len(rr)} | {'·'.join(dv) or '-'} | {emean} ({', '.join(f2(e) for e in eall)}) | {f2(sum(ms) / len(ms), 1) if ms else '-'} | {max(lim)} | "
          f"{sc_q:.2f} · {sc_m:.2f} | {sc12:.2f} | {hot} ({hot2}) | {sp40s} | {spls} |")
