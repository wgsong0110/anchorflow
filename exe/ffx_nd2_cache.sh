#!/bin/bash
# 절 CM192: 표 캐시 만들기 (로컬, R2 FUSE 에서 metrics.json 만 복사) -> <캐시>/runs/<장면>_<판>_r<반복>/metrics.json
#   레고·pillow2sofa = runs/<장면>_<판>.wel_r<반복> (새 판 --wrx_el 1), bread·wolf = runs_nd/ (CM191 판; --wrx_el 1 은 소성 재질에서 계산 불변)
#   영 에너지 모드 수 (eig, 강성 고유값 -- 접촉 처리와 무관) 는 레고·pillow2sofa 새 판 r1 에 분석이 없어 runs_nd r1 값을 옮김 (eig_src 표시)
set -e
C=$1; LB=~/r2/result/anchorflow/lbours; mkdir -p $C/runs
cp $LB/paper_remeasure2/final2/all.json $LB/final_pa/bl/pareto_bl.json $C/
for d in $LB/final_fix/runs_nd/*/; do n=$(basename $d); case $n in bread_*|wolf_*) mkdir -p $C/runs/$n; cp $d/metrics.json $C/runs/$n/ 2>/dev/null || true;; esac; done
for d in $LB/final_fix/runs/{L000,pillow2sofa}_*.wel_r[123]/; do n=$(basename $d); case $n in *lean*|*dump*) [[ $n == *_ss*.dump.wel_r1 ]] || continue;; esac
  m=${n/.wel_r/_r}; mkdir -p $C/runs/$m; cp $d/metrics.json $C/runs/$m/; done
python3 - "$C" <<'P'
import json, os, sys, glob
C = sys.argv[1]; LB = os.path.expanduser("~/r2/result/anchorflow/lbours")
for s in ("L000", "pillow2sofa"):
    for v in ("def", "q1", "q27"):
        f = f"{C}/runs/{s}_{v}_r1/metrics.json"; o = f"{LB}/final_fix/runs_nd/{s}_{v}_r1/metrics.json"
        if os.path.exists(f) and os.path.exists(o):
            m = json.load(open(f)); e = json.load(open(o)).get("eig")
            if not isinstance(m.get("eig"), dict) and isinstance(e, dict):
                m["eig"] = e; m["eig_src"] = "runs_nd r1 (CM191 판, 강성 고유값은 접촉 처리와 무관)"; json.dump(m, open(f, "w"))
print(len(glob.glob(f"{C}/runs/*")), "판")
P
# 베이스라인 덧칸: VR-GS 반복 1 (final_fix/vrgs_*.json) + 재질 확장 (final_pa/blx/pareto_blx.json) + i-PG·OmniPhysGS (final_fix/bl_mpm/pareto_mpmbl.json) + Spring-Gaus pillow2sofa (final_fix/sg_pillow2sofa.json, 있으면)
python3 - "$C" <<'P'
import json, os, sys
C = sys.argv[1]; LB = os.path.expanduser("~/r2/result/anchorflow/lbours")
rows = []
for f, src in ((f"{LB}/final_fix/vrgs_pillow2sofa.json", "final_fix/vrgs_pillow2sofa.json"), (f"{LB}/final_fix/vrgs_lego_it1.json", "final_fix/vrgs_lego_it1.json"),
               (f"{LB}/final_pa/blx/pareto_blx.json", "final_pa/blx/pareto_blx.json"), (f"{LB}/final_fix/bl_mpm/pareto_mpmbl.json", "final_fix/bl_mpm/pareto_mpmbl.json"),
               (f"{LB}/final_fix/sg_pillow2sofa.json", "final_fix/sg_pillow2sofa.json")):
    if os.path.exists(f):
        for r in json.load(open(f)):
            r = dict(r); r.setdefault("src", src); r.pop("fit_evals", None)
            rows = [q for q in rows if (q["method"], q["scene"]) != (r["method"], r["scene"])] + [r]
json.dump(rows, open(f"{C}/bl_extra.json", "w"), ensure_ascii=False, indent=1)
print("덧칸", [(r["method"], r["scene"]) for r in rows])
P
