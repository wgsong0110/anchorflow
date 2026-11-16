"""[final_fix_mpmbl] R2 lbours/final_fix/bl_mpm/<방법>_<장면>/ 결과 -> pareto_mpmbl.json (vrgs_pillow2sofa.json 행과 같은 키), phys_mpmbl.json, vis_mpmbl.json.
  python3 exe/mpmbl_collect.py ~/r2/result/anchorflow/lbours/final_fix/bl_mpm <커밋 표기>
"""
import json
import os
import statistics
import sys

RB, COMMIT = sys.argv[1], sys.argv[2]
NAME = dict(ipg="i-PhysGaussian", omni="OmniPhysGS")
rows, phys, vis = [], [], []
for m in ("ipg", "omni"):
    for s in ("L000", "pillow2sofa", "bread", "wolf"):
        O = f"{RB}/{m}_{s}"
        if not os.path.exists(f"{O}/chosen.txt"):
            rows.append(dict(method=m, scene=s, N=None, setting="", reps=0, ok=0, err=None, err_all=[], ms=None, ms_all=[], div=[], note="결과 없음", commit=COMMIT, gpu=""))
            continue
        ch = open(f"{O}/chosen.txt").read().split()
        N = None if ch[0] == "none" else int(ch[0]); Noff = int(ch[1]); fused = ch[2]
        tried = open(f"{O}/tried.txt").read().split() if os.path.exists(f"{O}/tried.txt") else []
        R = []
        for rep in "abc":
            p = f"{O}/N{N}_{rep}/err.json"
            if N is not None and os.path.exists(p):
                R.append(json.load(open(p)))
        ok = [r for r in R if r["div"] is None and r["frames_done"] == r["frames_total"]]
        ea = [100 * r["err_mean"] for r in ok]; ma = [r["ms_mean"] for r in ok]
        setting = (f"i-PG newton\\_gmres, 공식 dt 배수 (논문 최대 k=20) -> 서브스텝 {Noff}" if m == "ipg" else f"OmniPhysGS 공식 dt 3e-4 s -> 서브스텝 {Noff}")
        if N is not None and N != Noff:
            setting += f" 발산 -> {N}"
        note = []
        if tried:
            note.append("시도 (N:완주) " + " ".join(tried))
        if m == "ipg":
            note.append(f"AF_IPG_FUSED={fused} (통합 커널은 탄성 재질만: 레고·pillow2sofa 1, bread·wolf 0 = 공식 warp 경로)")
        if m == "omni":
            note.append("공식 MPMModel·물리 구성모델 (jelly Corotated, plasticine Corotated+VonMises, sand StVK+DruckerPrager), 길이 0.5 배 축소, 입자 질량 균일 (공식), clip_bound 상자")
            if len(ch) > 3:
                note.append("재질 값 " + " ".join(ch[3:]))
        r0 = R[0] if R else {}
        rows.append(dict(method=m, scene=s, N=N, setting=setting, reps=len(R), ok=len(ok),
                         err=statistics.median(ea) if ea else None, err_all=ea, ms=statistics.median(ma) if ma else None, ms_all=ma,
                         div=[r["div"] for r in R if r["div"] is not None], note="; ".join(note), commit=COMMIT,
                         gpu=f"{r0.get('gpu', '')} 1 장 (vast), 카드 단독 직렬 3 회, 프레임 시간 = 서브스텝 루프 앞뒤 동기화 벽시계, 첫 2 프레임 뺀 평균 (pareto_mpm --time 과 같음)",
                         frames_done=r0.get("frames_done"), frames_total=r0.get("frames_total"), n_particles=r0.get("n_particles")))
        for fn, L in (("phys.json", phys), ("vis.json", vis)):
            if os.path.exists(f"{O}/{fn}"):
                q = json.load(open(f"{O}/{fn}"))
                q = dict(dict(method=m, scene=s, N=N), **q) if isinstance(q, dict) else dict(method=m, scene=s, N=N, data=q)
                L.append(q)
for fn, L in (("pareto_mpmbl.json", rows), ("phys_mpmbl.json", phys), ("vis_mpmbl.json", vis)):
    json.dump(L, open(f"/tmp/{fn}", "w"), ensure_ascii=False, indent=1)
    print(fn, len(L))
for r in rows:
    print(r["method"], r["scene"], r["N"], r["ok"], "/", r["reps"], r["err"], r["ms"])
