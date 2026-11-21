#!/usr/bin/env python3
"""파레토 i-PG·OmniPhysGS 칸 (오차·ms·부피 변화) 을 R2 원자료에서 만든다 -> <논문 폴더>/pareto_mpmbl_cells.json.
논문 repo results_all/tools/add_mpmbl.py 가 이 파일을 읽는다 (예전 add_mpmbl.py 의 E=·DV= 하드코딩과 고정 평균 행 대체).
  LB=<R2 lbours 또는 repro/fetch_raw.sh 사본> python3 exe/pareto_mpmbl_cells.py <논문 폴더>
원자료: lbours/final_fix/bl_mpm/pareto_mpmbl.json (방법·장면별 3 회 중앙값, src 폴더) + repro/data/pareto_mpmbl_sources.json (값이 없는 칸의 위치)
부피 변화 = <대표 폴더>/phys.json 의 dV_absmax x 100 (파레토 표 '부피 변화' 열과 같은 정의, exe/ffx_phys.py)."""
import json, os, sys

LB = os.environ.get("LB") or os.path.expanduser("~/r2/result/anchorflow/lbours")
OUT = sys.argv[1]
SPEC = json.load(open(os.environ.get("PARETO_MPMBL_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro", "data", "pareto_mpmbl_sources.json")))
SN = dict(L000="Lego", pillow2sofa="pillow2sofa", bread="bread", wolf="wolf"); MN = dict(ipg="i-PG", omni="OmniPhysGS")
rows = {(r["method"], r["scene"]): r for r in json.load(open(f"{LB}/final_fix/bl_mpm/pareto_mpmbl.json"))}
cells = []
for m in ("ipg", "omni"):
    for s in SN:
        sp = SPEC.get(f"{m}:{s}")
        if sp:
            e = json.load(open(f"{LB}/{sp['err']}")); t = json.load(open(f"{LB}/{sp['ms']}"))
            assert e["frames_done"] == e["frames_total"] and t["frames_done"] == t["frames_total"], (m, s)
            err, ms, N, pdir = e["err_pct"], t["ms_mean"], sp["N"], sp["phys"]
            src = dict(err=sp["err"], ms=sp["ms"], phys=sp["phys"])
        else:
            r = rows[(m, s)]
            assert r.get("err") is not None and r.get("ok") == r.get("reps"), (m, s)
            err, ms, N, pdir = r["err"], r["ms"], r["N"], r["src"].replace("lbours/", "", 1).rstrip("/")
            src = dict(err_ms="final_fix/bl_mpm/pareto_mpmbl.json", phys=pdir)
        ph = json.load(open(f"{LB}/{pdir}/phys.json"))
        assert ph.get("method") == m and ph.get("scene") == s, (pdir, ph.get("method"), ph.get("scene"))
        cells.append(dict(scene=SN[s], method=MN[m], N=N, err=round(err, 2), ms=float(round(ms)), dV=round(100 * ph["dV_absmax"], 2),
                          err_raw=err, ms_raw=ms, dV_raw=100 * ph["dV_absmax"], src=src))
json.dump(dict(rule=SPEC.get("ms_round"), cells=cells), open(f"{OUT}/pareto_mpmbl_cells.json", "w"), ensure_ascii=False, indent=1)
for c in cells:
    print(f"{c['scene']:12s} {c['method']:11s} N{c['N']:<5} 오차 {c['err']:.2f} ms {c['ms']:.0f} 부피 {c['dV']:.2f}")
