"""[final_fix_mpmbl] R2 lbours/final_fix/bl_mpm 결과 -> pareto_mpmbl.json (vrgs_pillow2sofa.json 행과 같은 키), phys_mpmbl.json, vis_mpmbl.json (/tmp 에 씀).
  python3 exe/mpmbl_collect.py ~/r2/result/anchorflow/lbours/final_fix/bl_mpm
"""
import json
import os
import statistics
import sys

RB = sys.argv[1]
GPU = "RTX A6000 1 장 (vast), 카드 단독, 프레임 시간 = 서브스텝 루프 앞뒤 동기화 벽시계의 첫 2 프레임 뺀 평균 (exe/pareto_mpm.py --time 과 같음), 3 회"
# (방법, 장면) -> (폴더, N, 공식 N, 설정, 메모, 커밋)
CELLS = {
    ("ipg", "L000"): ("ipg_L000", 42, 42, "i-PG newton\\_gmres, 공식 dt 배수 k=20 (서브스텝 round(833/20)=42)",
                      "AF_IPG_FUSED=1 (탄성 통합 커널, 결과 불변 가속; 안정성 실험과 같음)", "ddfd74b"),
    ("ipg", "pillow2sofa"): ("ipg_pillow2sofa", 10, 10, "i-PG newton\\_gmres, k=20 (서브스텝 200/20=10)", "AF_IPG_FUSED=1", "ddfd74b"),
    ("ipg", "bread"): ("ipg_bread", 10, 5, "i-PG newton\\_gmres, k=20 (N 5) 발산 -> k=10 (N 10)",
                       "k=20 (N 5) 은 프레임 166 에서 Newton-GMRES 최소제곱 NaN (LinAlgError) 발산; 소성이라 통합 커널 끔 (AF_IPG_FUSED=0), "
                       "같은 계산의 GPU 판 AF_IPG_GPU=1 (bread 3 프레임 CPU 경로와 오차 같음 0.020/0.052/0.084 %); 재질 키는 정답 MPM 행 실행기 (pareto_mpm) 와 같이 "
                       "항복 응력을 넘기지 않음 (PG 전 구간 확인: 항복 0 vs 200 의 정답 오차 0.038 vs 0.020 %)", "3f6e8ee"),
    ("omni", "L000"): ("omni_L000", 56, 56, "OmniPhysGS 공식 dt 3e-4 s (서브스텝 ceil((1/60)/3e-4)=56)", "", "ddfd74b"),
    ("omni", "pillow2sofa"): ("omni_pillow2sofa", 67, 67, "OmniPhysGS 공식 dt 3e-4 s (서브스텝 67)", "", "ddfd74b"),
    ("omni", "bread"): ("omni_bread", 34, 34, "OmniPhysGS 공식 dt 3e-4 s (서브스텝 34), Corotated + VonMises 항복 200",
                        "항복 응력 = config 200 (PG 정답 경로 전 구간 확인에서 200 쪽이 정답과 더 가까움)", "ddfd74b"),
    ("omni", "wolf"): ("omni_wolf_corot", 1072, 134, "OmniPhysGS 공식 dt 3e-4 (N 134) 폭주 -> N 1072 (dt 3.7e-5), Corotated + DruckerPrager 마찰각 30",
                       "공식 dt (N 134·268) 은 프레임 1 에 바닥을 뚫고 영역 바닥까지 빠짐 (E 5e7 모래, 명시 MPM CFL dt < dx/sqrt(E/rho) ~ 6e-5), N 536 은 프레임 22 폭주; "
                       "처음 쓴 StVK + DruckerPrager (PG 모래의 StVK 대응) 는 N 1072·1999 에서도 Green 변형률 StVK 가 압축에 무너져 바닥을 뚫음 (N1999 3 회 중 2 회 발산, 완주 판 오차 20 %) "
                       "-> 공식 물리 구성모델 목록의 Corotated + DruckerPrager (OmniPhysGS bear_sand 후보) 로 바꿈 (lbours/final_fix/bl_mpm/omni_wolf_n1072·omni_wolf_n1999 = StVK 기록); "
                       "마찰각 = config 30 (PG 전 구간 확인: 25 vs 30 의 정답 오차 0.014 vs 0.006 %); 시간 = bread 카드 (재시작 뒤 CPU 가 느려진 호스트, 같은 판 StVK N1072 가 다른 A6000 에서 29.6 vs 37.3 s)", "4fd9633b"),
}
NOTE_OMNI = "공식 MPMModel·물리 구성모델, 길이 0.5 배 축소, 입자 질량 균일 (공식), clip_bound 상자"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


rows, phys, vis = [], [], []
for (m, s), (d, N, Noff, setting, note, commit) in CELLS.items():
    O = f"{RB}/{d}"
    R = [r for r in (load(f"{O}/N{N}_{rep}/err.json") for rep in "abc") if r is not None]
    ok = [r for r in R if r["div"] is None and r["frames_done"] == r["frames_total"]]
    ea = [100 * r["err_mean"] for r in ok]; ma = [r["ms_mean"] for r in ok]
    nt = "; ".join(q for q in (note, NOTE_OMNI if m == "omni" else "") if q)
    rows.append(dict(method=m, scene=s, N=N, setting=setting, reps=len(R), ok=len(ok), err=statistics.median(ea) if ea else None, err_all=ea,
                     ms=statistics.median(ma) if ma else None, ms_all=ma, div=[r["div"] for r in R if r["div"] is not None], note=nt,
                     commit=f"{commit} (브랜치 final_fix_mpmbl)", gpu=GPU, N_official=Noff, frames_total=R[0]["frames_total"] if R else None,
                     n_particles=R[0].get("n_particles") if R else None, src=f"lbours/final_fix/bl_mpm/{d}/"))
    for fn, L in (("phys.json", phys), ("vis.json", vis)):
        q = load(f"{O}/{fn}")
        if q is not None:
            L.append(dict(dict(method=m, scene=s, N=N), **{k: v for k, v in q.items() if k not in ("method", "scene", "N")}))
# i-PG wolf: 오차 칸 없음 -- 근거와 3 프레임 단독 시간
t3 = load(f"{RB}/ipg_wolf/N1000_t3/err.json")
rows.append(dict(method="ipg", scene="wolf", N=1000, setting="i-PG newton\\_gmres k=20·10·5 (N 100·200·400) 발산 -> k=2 (N 1000)", reps=1 if t3 else 0, ok=0,
                 err=None, err_all=[], ms=(t3 or {}).get("ms_mean"), ms_all=[x for x in ((t3 or {}).get("ms") or [])],
                 div=[{"N": 100, "frame": 1, "reason": "Newton 발산 (서브스텝 0 잔차 0.24 -> 302, 수렴 못 함) 뒤 GMRES 최소제곱 NaN (LinAlgError)"},
                      {"N": 200, "frame": 1, "reason": "같음"}, {"N": 400, "frame": 1, "reason": "같음"}],
                 note="k>=5 는 첫 서브스텝부터 Newton 이 발산 (CPU 경로 AF_IPG_GPU 끔도 같은 잔차 경로 -> 이식 버그 아님; 통합 커널은 소성이라 처음부터 안 씀). "
                      "k=2 (N 1000) 는 진행하나 서브스텝 ~5.1 s -> 프레임 ~5070 s, 50 프레임 ~70 h 라 오차 칸 없음; ms = 같은 판 3 프레임 단독 실행의 3 프레임째 (첫 2 프레임 뺀 평균 규약), "
                      "오차는 3 프레임까지만: " + (", ".join(f"{100 * e:.3f}%" for e in t3["err"]) if t3 else "-"),
                 commit="afcd058 (브랜치 final_fix_mpmbl)", gpu=GPU.replace(", 3 회", ", 1 회 3 프레임"), N_official=100, frames_total=50, frames_done=(t3 or {}).get("frames_done"),
                 src="lbours/final_fix/bl_mpm/ipg_wolf/"))
for fn, L in (("pareto_mpmbl.json", rows), ("phys_mpmbl.json", phys), ("vis_mpmbl.json", vis)):
    json.dump(L, open(f"/tmp/{fn}", "w"), ensure_ascii=False, indent=1)
    print(fn, len(L))
for r in rows:
    print(r["method"], r["scene"], r["N"], f"{r['ok']}/{r['reps']}", r["err"] and round(r["err"], 3), r["ms"] and round(r["ms"], 1))
