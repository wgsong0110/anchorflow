"""절 Q: 스윕 결과 표 (summary.json + period.json, NOT_TIMED 표식) -- 사용: python exe/fem_q_table.py <dir>..."""
import json
import os
import sys

print("| 태그 | 질량 | 전달 가중 | APIC | N | K | 방식 | 결과 | 오차 | ms 평균 / 중앙 | 측정 | 최저 입자 (m) / 아래 최대 | 주기 (s) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    if not os.path.isdir(d):
        continue
    f = os.path.join(d, "summary.json")
    if not os.path.exists(f):
        print(f"| {os.path.basename(d)} | 요약 없음 (실패) |")
        continue
    s = json.load(open(f))
    pj = os.path.join(d, "period.json")
    per = json.load(open(pj))["T_zc"] if os.path.exists(pj) else None
    res = "완주" if not s["diverged"] else f"발산 {s['diverged']}"
    e = s["err_mean"] if s["err_mean"] is not None else s["err_mean_done"]
    smin = f"{s['s_min']:+.5f} / {s['n_below_max']}" if s.get("s_min") is not None else "-"
    tm = "비측정" if os.path.exists(os.path.join(d, "NOT_TIMED")) else "단독"
    print(f"| {os.path.basename(d)} | {s['mass']} | {s.get('rep_w', 'std')} | {'o' if s.get('affine') else '-'} | {s['N']} | {s.get('rep') or '-'} | {s.get('rep_mode') or '-'} | {res} | "
          f"{100 * e:.3f} %{'' if not s['diverged'] else ' (발산 전)'} | {s['ms_mean']:.2f} / {s['ms_med']:.2f} | {tm} | {smin} | {'' if per is None else '%.3f' % per} |")
