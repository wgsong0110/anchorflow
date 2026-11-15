"""[final_pa_bl] 파레토 베이스라인 대표 점 모음 -> pareto_bl.json (R2 lbours/final_pa/bl/).

각 태그 폴더 (R2 에서 내려받은 로컬 사본) 의 g/<m>_<t>_n<N>_r<k>/res.json 을 모아 방법 x 장면 하나씩:
  {method, scene, N, setting, reps, ok, err, err_all, ms, ms_all, div, note}   (err·ms = 안정한 판들의 중앙값, 3 회)
해당 없음·실패 항목은 --na JSON ([{method, scene, note}]) 에서 그대로 더함.

  python exe/fpabl_collect.py --dirs /tmp/fpabl/T1 /tmp/fpabl/T2 --na na.json --out pareto_bl.json
"""
import argparse
import glob
import json
import os
import re
import statistics as S

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--na", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
SET = dict(mpm="PhysGaussian 공식 config 서브스텝 int(frame_dt / substep_dt) (pr2_grid HI)",
           vrgs="논문 PG 비교 설정 dt 1e-4 s x 반복 11, E = 논문 절차 (황금분할 10 회)",
           phystwin="공식 dt 5e-5 를 프레임 간격에 맞춘 N0 (준비 판 fit.json), 물성 = CMA-ES 20 세대 + 경사 학습 200 반복",
           springgaus="공식 3 단계 학습 끝 n_step (sg_train.json)")
runs = {}
for d in a.dirs:
    for f in glob.glob(f"{d}/g/*_r*/res.json"):
        m_ = re.match(r"(\w+?)_(\w+)_n(\d+)_r(\d+)$", os.path.basename(os.path.dirname(f)))
        if not m_:
            continue
        m, t, N, k = m_.group(1), m_.group(2), int(m_.group(3)), int(m_.group(4))
        runs.setdefault((m, t, N), {})[k] = json.load(open(f))
out = []
for (m, t, N), rr in sorted(runs.items()):
    rs = [rr[k] for k in sorted(rr)]
    ok = [r for r in rs if r.get("stable") and r.get("ms") is not None]
    out.append(dict(method=m, scene=t, N=N, setting=SET.get(m, ""), reps=len(rs), ok=len(ok),
                    err=S.median([r["err"] for r in ok]) if ok else None, err_all=[r["err"] for r in ok],
                    ms=S.median([r["ms"] for r in ok]) if ok else None, ms_all=[r["ms"] for r in ok],
                    div=[r.get("div") for r in rs if not r.get("stable")], others=[r.get("others") for r in rs],
                    E=(rs[0].get("E") if m == "vrgs" else None), note=""))
if a.na:
    for q in json.load(open(a.na)):
        out.append(dict(method=q["method"], scene=q["scene"], N=None, setting=q.get("setting", ""), reps=0, ok=0, err=None, err_all=[], ms=None, ms_all=[],
                        div=[], note=q["note"]))
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
for r in out:
    print(r["method"], r["scene"], r["N"], r["ok"], r["err"], r["ms"], r["ms_all"], r["note"][:60])
