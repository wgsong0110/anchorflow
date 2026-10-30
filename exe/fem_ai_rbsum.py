"""절 AI: --ai_rigid 기록 (fem.jsonl 의 rigid = [[잔차 RMS/EXT %, 몸 좌표 높이 m, 질량중심 z], 정답 같은 것]) 요약 -- 충격 뒤 몸 높이 최저·끝, 끝 잔류 변형, 질량중심 z 끝 (우리 / 정답).
사용: python exe/fem_ai_rbsum.py <root> [태그 접두...]"""
import glob
import json
import os
import sys

R, FILT = sys.argv[1], sys.argv[2:]
print("| 판 | 완주 | 오차 % | 몸 높이 0 → 최저 → 끝 (m) 우리 | 정답 | 끝 잔류 변형 % 우리 / 정답 | 끝 질량중심 z 우리 / 정답 |")
print("|---|---|---|---|---|---|---|")
for dd in sorted(glob.glob(os.path.join(R, "*_rb*"))):
    tg = os.path.basename(dd)
    if FILT and not any(tg.startswith(f) for f in FILT):
        continue
    try:
        s = json.load(open(os.path.join(dd, "summary.json")))
        rr = [json.loads(q) for q in open(os.path.join(dd, "fem.jsonl"))]
    except Exception:
        continue
    rg = [(q["t"] + 1, q["rigid"]) for q in rr if "rigid" in q]
    if not rg:
        continue
    ho = [x[1][0][1] for x in rg]
    hg = [x[1][1][1] for x in rg]
    fin = s.get("diverged") is None
    e = s.get("err_mean") if fin else s.get("err_mean_done")
    print(f"| {tg} | {'완주' if fin else '발산 %s' % s.get('diverged')} | {100 * e:.2f} | {ho[0]:.3f} → {min(ho):.3f} → {ho[-1]:.3f} (f{rg[-1][0]}) | {hg[0]:.3f} → {min(hg):.3f} → {hg[-1]:.3f} | "
          f"{rg[-1][1][0][0]:.2f} / {rg[-1][1][1][0]:.2f} | {rg[-1][1][0][2]:.3f} / {rg[-1][1][1][2]:.3f} |")
