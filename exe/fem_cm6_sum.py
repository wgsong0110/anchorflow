"""절 CM6 표: 판별 평균·프레임 30·끝 오차, 정답·우리 높이 (질량 중심·최대, 프레임 30·50), 반복, 장벽 보폭 k 분포."""
import json, os, sys
import numpy as np
R = sys.argv[1]
print("# cm6 -- wolf 무너짐 진단 (절 CM4 공통 설정 wolf 2x5 에서 인자 하나씩, 결정적)\n")
print("| 판 | 평균 % | f30 % | 끝 % | 질량중심 z f30 (정답) | f50 (정답) | 최대 z f30 (정답) | f50 (정답) | N·P /프레임 | 보폭 k>0 비율 · 평균 k | 끝 해시 | 인자 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
curves = {}
for d in sorted(os.listdir(R)):
    p = os.path.join(R, d)
    if not os.path.isdir(p):
        continue
    try:
        L = [json.loads(l) for l in open(os.path.join(p, "fem.jsonl"))]
        S = json.load(open(os.path.join(p, "summary.json")))
    except Exception as e:
        print(f"| {d} | 실패 {e} |"); continue
    ri = open(os.path.join(p, "run_info.txt")).read().split("\n")[0]
    ri = ri.split("fem_an_rt_sim.py", 1)[-1].replace("|", "/")[-260:]
    er = [r["err"] * 100 for r in L]
    def at(i, k, j):
        return L[i][k][j] if i < len(L) and k in L[i] else float("nan")
    i30 = min(29, len(L) - 1); iE = len(L) - 1
    nN = np.mean([r["it"]["nN"] for r in L if "it" in r]) if "it" in L[0] else float("nan")
    nP = np.mean([r["it"]["nP"] for r in L if "it" in r]) if "it" in L[0] else float("nan")
    bk = [k for r in L for k in r.get("bk", [])]
    bks = f"{np.mean(np.array(bk) > 0):.2f} · {np.mean(bk):.2f}" if bk else "-"
    dv = S.get("diverged")
    print(f"| {d} | {np.mean(er):.2f}{' x%d' % dv if dv else ''} | {er[i30]:.2f} | {er[iE]:.2f} | {at(i30,'hp',0):.3f} ({at(i30,'hg',0):.3f}) | {at(iE,'hp',0):.3f} ({at(iE,'hg',0):.3f}) | "
          f"{at(i30,'hp',1):.3f} ({at(i30,'hg',1):.3f}) | {at(iE,'hp',1):.3f} ({at(iE,'hg',1):.3f}) | {nN:.1f}·{nP:.1f} | {bks} | {S.get('x_sha1')} | {ri} |")
    curves[d] = dict(err=er, hp=[r.get("hp") for r in L], hg=[r.get("hg") for r in L])
print("\n## 프레임별 (5 프레임마다): 오차 % / 질량중심 z (정답 질량중심 z)")
for d, c in curves.items():
    print(f"- {d}: " + " ".join(f"{i + 1}:{c['err'][i]:.1f}/{(c['hp'][i] or [float('nan')])[0]:.3f}({(c['hg'][i] or [float('nan')])[0]:.3f})" for i in range(4, len(c["err"]), 5)))
json.dump(curves, open(os.path.join(R, "cm6_curves.json"), "w"))
