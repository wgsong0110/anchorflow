"""파레토 표 (doc/pareto.md): exe/pareto.sh 출력 폴더 -> 표 (stdout, markdown) + 점 목록 points.json (그림은 exe/pareto_plot.py, 로컬).
점 = (씬, 방식, 설정, 오차 %, 프레임 평균 ms, 최대 ms, 발산 프레임, 카드 0 의 다른 프로세스 수)."""
import glob
import json
import os
import re
import sys

R = sys.argv[1]
P = []


def card_others(d):
    try:
        m = re.search(r"다른 계산 프로세스 (\d+) 개", open(os.path.join(d, "card.txt")).read())
        return int(m.group(1)) if m else None
    except OSError:
        return None


for d in sorted(glob.glob(os.path.join(R, "t_ours_*"))):
    m = re.match(r"t_ours_(\w+?)_K(\d+)c(\d+)$", os.path.basename(d))
    f = os.path.join(d, "summary.json")
    if not m or not os.path.exists(f):
        continue
    s = json.load(open(f))
    ms = [json.loads(q)["ms"] for q in open(os.path.join(d, "fem.jsonl"))][2:]
    ms = [q for q in ms if q == q]
    P.append(dict(scene=m.group(1), method="ours", setting=f"K{m.group(2)} x 뉴턴 2 x PCG {m.group(3)}", err=(100 * s["err_mean"] if s.get("err_mean") is not None else None),
                  err_done=100 * s["err_mean_done"], ms=s.get("ms_mean"), ms_max=max(ms) if ms else None, div=s.get("diverged"), others=card_others(d)))
for d in sorted(glob.glob(os.path.join(R, "t_mpm_*"))):
    m = re.match(r"t_mpm_(\w+?)_n(\d+)$", os.path.basename(d))
    f = os.path.join(d, "err.json")
    if not m:
        continue
    if not os.path.exists(f):
        P.append(dict(scene=m.group(1), method="mpm", setting=f"서브스텝 {m.group(2)}", err=None, err_done=None, ms=None, ms_max=None, div="중단 (로그)", others=card_others(d)))
        continue
    s = json.load(open(f))
    P.append(dict(scene=m.group(1), method="mpm", setting=f"서브스텝 {m.group(2)}", err=(100 * s["err_mean"] if s.get("div") is None else None),
                  err_done=(100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None), ms=s.get("ms_mean"), ms_max=s.get("ms_max"),
                  div=(s["div"]["frame"] if s.get("div") else None), others=card_others(d)))
for d in sorted(glob.glob(os.path.join(R, "t_vrgs_*"))):
    f = os.path.join(d, "run.json")
    if not os.path.exists(f):
        continue
    s = json.load(open(f))
    for r in s["res"]:
        P.append(dict(scene=s["tag"], method="vrgs", setting=f"서브스텝 {r['N']} x 반복 {r['iters']} (E {r['E']:.3g})", err=(100 * r["err"] if r["err"] is not None else None),
                      err_done=(100 * r["err_done"] if r.get("err_done") is not None else None), ms=r["ms_mean"], ms_max=r["ms_max"], div=r["div"], others=card_others(d),
                      N=r["N"], iters=r["iters"], E=r["E"], E_gt=s["E_gt"]))
json.dump(P, open(os.path.join(R, "points.json"), "w"), indent=1, ensure_ascii=False)
NAME = dict(ours="우리 (절 AN.22 채택 판)", mpm="정답 MPM (PG warp, 공식 config)", vrgs="VR-GS 직접 구현 (XPBD 케이지)")
print("# 파레토 표 -- 오차 % = 롤아웃 평균 |x−x_gt|/EXT, ms = 프레임 시간 (A6000 카드 0 단독, 첫 2 프레임 뺀 평균 / 최대), 발산 = 그 프레임\n")
for sc in ("L000", "ficus"):
    print(f"## {sc}\n\n| 방식 | 설정 | 오차 % | 평균 ms | 최대 ms | 발산 | 측정 중 카드 0 다른 프로세스 |\n|---|---|---|---|---|---|---|")
    for q in P:
        if q["scene"] != sc:
            continue
        e = "%.2f" % q["err"] if q["err"] is not None else ("x (완주 전 %.2f)" % q["err_done"] if q.get("err_done") is not None else "x")
        print(f"| {NAME[q['method']]} | {q['setting']} | {e} | {'' if q['ms'] is None else '%.2f' % q['ms']} | {'' if q['ms_max'] is None else '%.2f' % q['ms_max']} | {q['div'] or ''} | {q['others']} |")
    print()
for f in sorted(glob.glob(os.path.join(R, "fit_*", "fit.json"))):
    s = json.load(open(f))
    print(f"### VR-GS E 맞춤 {os.path.basename(os.path.dirname(f))} (정답 E {s['E_gt']:g}; 케이지 정점 {s['info']['nv']} 사면체 {s['info']['nt']} 색 {s['info']['ncolor']})\n")
    for r in s["res"]:
        print(f"- {r['N']}x{r['iters']}: E {r['E_best']:.4g} ({r['E_best'] / s['E_gt']:.3g} 배), 오차 {'x' if r['err'] is None else '%.2f %%' % (100 * r['err'])}")
    print()
f = os.path.join(R, "t_vrgs_scale", "scale.json")
if os.path.exists(f):
    print("### VR-GS 규모 대조 (논문 표 1, RTX 4090 렌더 포함 fps; 여기 A6000 시뮬레이션만)\n\n| 장면 | 정점 (논문) | 사면체 | 색 | 가우시안 | 서브스텝 1 회 ms (반복별) | 중심 갱신 ms | 논문 프레임 안 최대 서브스텝 |\n|---|---|---|---|---|---|---|---|")
    for r in json.load(open(f)):
        print(f"| {r['scene']} | {r['nv']} ({r['nv_paper']}) | {r['nt']} | {r['ncolor']} | {r['gauss']} | " + ", ".join(f"{k}: {v:.4f}" for k, v in r["sub_ms"].items())
              + f" | {r['skin_ms']:.4f} | {r['max_sub_in_budget']:.1f} (반복 {r['iters_paper']}, {r['fps_paper']} fps) |")
