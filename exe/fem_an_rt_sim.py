"""절 AN 실시간 판 롤아웃 (lib/anchorflow/fem_an_rt.py): 프레임마다 CUDA 그래프 재생 한 번 (P2G -> 고정 예산 풀이 -> G2P), 블록 야코비 전처리.
정의는 exe/fem_an_sim.py --fast --pc jacobi 와 같다 (정밀도: 곱·누적 f32, 기울기·내적·에너지 f64). 프레임 시간 = 재생 GPU 시간 (CUDA 이벤트), 오차 평가는 재생 밖."""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                    # noqa: E402
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_an_rt as ART                                             # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--cell", type=float, default=1.0)
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--integ", choices=["mid", "be"], default="mid")
ap.add_argument("--newton", type=int, default=2)
ap.add_argument("--cg", type=int, default=5)
ap.add_argument("--ls", type=int, default=1)
ap.add_argument("--no_vfl_out", action="store_true")
ap.add_argument("--fl_bar", action="store_true", help="장벽 바닥 (절 AN.19, 정의 변경): 법선 = 팔분 표본점 IPC 거리 장벽 (d̂ = ½ dx_정답, κ = (λ+2μ) V_q / d̂²), 고착 = 정답 sticky 가중 (띠 제한 없음) 을 접선에만; --ls_n 선탐색 필요")
ap.add_argument("--fl_pt", action="store_true", help="점별 고착 (절 AN.17, 정의 변경): 고착 거듭제곱을 가우스 점에서 한 뒤 노드로 질량 평균 (be·HHT·GA 처럼 단계 frac 1 판만, --pk·--K_contact 와 함께 쓰지 않음)")
ap.add_argument("--mg_levels", type=int, default=1, help="1 = 블록 야코비, 2 이상 = 그 층 수의 기하 멀티그리드 (거친 층 조밀 상자)")
ap.add_argument("--mg_nc", type=int, default=8)
ap.add_argument("--mg_pow", type=int, default=6)
ap.add_argument("--dump", action="store_true")
ap.add_argument("--dump_F", action="store_true", help="--dump 와 함께: 입자 변형 기울기 F 도 pred.pt 에 (영상 렌더용; 풀이 결과 불변)")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--no_fuse2", action="store_true", help="결과 불변 융합 2 판 끔 (비교용)")
ap.add_argument("--mg_csolve", action="store_true", help="가장 거친 층 한 블록 커널 (결과 불변, lego 셀 1 에서 더 느림)")
ap.add_argument("--mg_lreuse", action="store_true", help="정의 변경: 거친 층 ω 를 첫 프레임 값으로 재사용")
ap.add_argument("--mg_wreuse", action="store_true", help="정의 변경: 프레임 안 단계 τ 가 같으면 층 0 ω 재사용")
ap.add_argument("--gfin", action="store_true", help="걸음 끝 |g|/|g0| 를 fem.jsonl 에 (기울기 한 번 더 -- 시간 측정 판에서는 끌 것)")
ap.add_argument("--det", action="store_true", help="결정적 합산 (같은 명령 두 번이 비트 단위로 같게; lib/anchorflow/fem_an_det.py, MG 와 함께 못 씀) -- 판정용, 속도는 비결정 판으로")
ap.add_argument("--ga", choices=["ga", "hht"], default=None, help="generalized-α (Chung-Hulbert) 또는 HHT-α (implicit-err 절 AN.err3 증분 포텐셜 꼴, 시작점 = 걸음 시작 위치, 끝 = Newmark 위치 식)")
ap.add_argument("--rho", type=float, default=0.7)
ap.add_argument("--no_areset", action="store_true", help="고착 노드 걸음 끝 가속도 0 처치 끔")
ap.add_argument("--scheme", choices=["theta", "trbdf2"], default=None, help="단계 방식 (안 주면 --integ: mid = θ ½, be = θ 1)")
ap.add_argument("--theta", type=float, default=0.5)
ap.add_argument("--mfloor", type=float, default=0.0, help="관성 질량 하한 계수 c: m' = max(m, c τ² tr(B)/3) (0 = 없음)")
ap.add_argument("--K_contact", type=int, default=0, help="접촉 적응 걸음: 앞 프레임에 고착 노드가 있으면 이 걸음 수 (0 = 없음)")
ap.add_argument("--ls_n", type=int, default=0, help="선탐색 후보 2^0 .. 2^-(ls_n−1) (0 = 옛 1·½·¼)")
ap.add_argument("--descent", action="store_true", help="하강 방향 보장: g·δ ≥ 0 이면 δ ← α0 M r0 (PCG 첫 방향)")
ap.add_argument("--diag", action="store_true", help="프레임마다 뉴턴 반복별 진단 (에너지·|g|·g·δ·보폭·PCG 잔차·고착 불일치) 을 fem.jsonl 에")
ap.add_argument("--tolN", type=float, default=-1.0, help="절 AN.14 허용치 조기 종료: 걸음마다 뉴턴을 |g|/|g0| <= tolN 이면 멈춤 (--newton 은 상한; 음수 = 고정 반복)")
ap.add_argument("--tolP", type=float, default=-1.0, help="절 AN.14: PCG 를 sqrt(r·z / r0·z0) <= tolP 이면 멈춤 (--cg 는 상한; 음수 = 고정 반복)")
ap.add_argument("--gq_oct", type=float, default=-1.0, help="정의 변경 (절 AN.err4): >= 0 이면 칸 가우스 점 가중 (부피·판정 질량) = (1 − κ) x 팔분 칸 실제 입자 부피 (질량) + κ x 칸 합 / 8. 음수 = 원래 (칸 합 / 8)")
ap.add_argument("--pk", action="store_true", help="절 AN.18 지속형 커널 판: 프레임 하나 = 협동 실행 커널 한 번 (결과 정의 같음)")
ap.add_argument("--pk_loc", type=int, default=2, help="지속형 판: 이 MG 층 이상은 블록 0 하나가 (격자 장벽 줄이기; 99 = 끔)")
ap.add_argument("--pkb", type=int, default=256, help="지속형 판 물리 블록 크기")
ap.add_argument("--pkminb", type=int, default=1, help="지속형 판 __launch_bounds__ 최소 블록 수 (레지스터 상한)")
ap.add_argument("--mv2", action="store_true", help="절 AN.20: 칸 협동 헤시안 곱 (고운·거친 층; 결과 정의 같음, 반올림 순서만 다름)")
ap.add_argument("--mg_occ", action="store_true", help="절 AN.20: 거친 층 곱을 점유 칸 목록만 (빈 칸은 부피 0 이라 연산자 같음; --mv2 포함)")
ap.add_argument("--mg_fuse", type=int, default=0, help="절 AN.20: 이 MG 층 이상 V 순환을 한 블록 커널 하나로 (0 = 끔; 결정적 판과 같은 비트)")
ap.add_argument("--kbench", type=int, default=0, help="절 AN.20: 이 프레임 수를 돈 뒤 커널별 그래프 재생 시간 (µs/회) 을 kbench.json 에 쓰고 끝냄")
ap.add_argument("--prof", type=int, default=0, help="이 프레임마다 커널별 시간 (그래프 밖 한 번, 상태 되돌림) 을 fem.jsonl 에")
a = ap.parse_args()
dev = "cuda"
RTC.RAWPTR[0] = True
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
NODE_H = a.cell * DXG * math.sqrt(2.0)
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=torch.float64)
dt = h / a.K
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (dt / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
FL = None
if a.floor == "vel":
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    NR = np.array(pl["normal"], dtype=np.float64); NR = NR / np.linalg.norm(NR)
    PT = [float(q) for q in pl["point"]]
    FL = dict(PT=PT, NR=NR.tolist(), PN=float(np.dot(PT, NR)), DXG=DXG, KEX=dt / float(cfg["substep_dt"]), vout=not a.no_vfl_out, pt=a.fl_pt, bar=a.fl_bar, DH=0.5 * DXG, KS=(lam + 2.0 * mu) / (0.5 * DXG) ** 2)
BC = None
if a.cfg_bc:
    BC = []
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid":
            t0, t1 = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
            BC.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]], [float(q) for q in bc["velocity"]], t0, t1))
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid" and int(bc.get("reset", 0)) == 1:
            t1 = float(bc.get("end_time", 1e3))
            BC.append(([0.0, 0.0, 0.0], [1e9, 1e9, 1e9], [0.0, 0.0, 0.0], t1, t1 + 15.0 * h))
    BC = BC or None
F0 = DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 9)
if a.det:
    from anchorflow import fem_an_det as ADET
SOLVER = ADET.DetRT if a.det else ART.RTSolver
S = SOLVER(d["x"][0].to(dev), d["v"][0].to(dev), F0, m64, vol64, lat.A, lat.Ai, mu, lam, [float(q) for q in cfg["g"]], h, a.K, a.integ, a.newton, a.cg, a.ls,
                 floor=FL, damp=DAMP, bc=BC, mg_levels=a.mg_levels, mg_nc=a.mg_nc, mg_pow=a.mg_pow, descent=a.descent, ls_n=a.ls_n, scheme=a.scheme, theta=a.theta, mfloor=a.mfloor, K_contact=a.K_contact or None, gfin=a.gfin, fuse2=not a.no_fuse2, mg_wreuse=a.mg_wreuse, mg_lreuse=a.mg_lreuse, mg_csolve=a.mg_csolve, ga=a.ga, rho=a.rho, areset=not a.no_areset, tolN=a.tolN, tolP=a.tolP, goct=a.gq_oct, pk=a.pk, pk_loc=a.pk_loc, pkb=a.pkb, pkminb=a.pkminb, mv2=a.mv2, mg_occ=a.mg_occ, mg_fuse=a.mg_fuse)
print(f"[절 AN 실시간 판] 입자 {Np}, 프레임 {L}, 셀 {float(lat.A[:, 0].norm()):.4f} m (정답 dx {DXG} 의 {a.cell} 배), K {a.K} {a.integ}, 뉴턴 {a.newton} x PCG {a.cg}, 선탐색 {a.ls}, MG 층 {a.mg_levels}, 적분기 {a.ga + ' ρ ' + str(a.rho) if a.ga else a.integ}, "
      f"바닥 {a.floor}, 구속 {len(BC) if BC else 0}", flush=True)
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, ms = [], []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
PREDF = torch.empty(L, Np, 3, 3) if (a.dump and a.dump_F) else None
for fi in range(L):
    kp = ((S.pk_profile(fi * h) if a.pk else S.profile(fi * h)) if (a.prof and fi >= a.warm and (fi - a.warm) % a.prof == 0) else None)
    redo = S.step(fi * h)
    gt = d["x"][fi + 1].to(dev).float()
    xo = S.x_orig()
    e = float((xo - gt).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(xo).all()) and math.isfinite(e)
    errs.append(e if fin else float("nan")); ms.append(S.last_ms)
    c = S.counts()
    if a.diag:
        c["diag"] = S.diag()
    c["K"] = S.last_K
    if a.gfin:
        c["grel"] = S.grel()
    flog.write(json.dumps(dict(t=fi, err=e, ms=S.last_ms, redo=redo, **c, **({"kernel_ms": kp} if kp else {}))) + "\n"); flog.flush()
    print(f"  프레임 {fi + 1}/{L} {S.last_ms:.2f} ms 오차 {100 * e:.3f}%{' (용량 늘려 다시)' if redo else ''} 점 {c['nq']} 노드 상자 {c['nn']} 유클리드 평균 점 {c['nbad']}{(' 반복 ' + str(c['it'])) if 'it' in c else ''}", flush=True)
    if PRED is not None:
        PRED[fi] = xo.cpu()
        if PREDF is not None:
            _fo = torch.empty_like(S.F); _fo[S.perm] = S.F
            PREDF[fi] = _fo.reshape(-1, 3, 3).cpu()
    if a.kbench and fi + 1 == a.kbench:
        kb = S.kbench()
        kb["_frame_ms"] = S.last_ms; kb["_counts"] = c; kb["_mg_cnt"] = S.mg_cnt.cpu().tolist() if hasattr(S, "mg_cnt") else None
        kb["_LM"] = S.LM.cpu().tolist() if hasattr(S, "LM") else None
        json.dump(kb, open(os.path.join(a.out, "kbench.json"), "w"), indent=1, ensure_ascii=False)
        print("[kbench]", json.dumps(kb, ensure_ascii=False), flush=True)
        L = fi + 1
        break
    if not fin or e > 2.0:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = [q for i, q in enumerate(ms) if i >= a.warm]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=float(np.mean(sc)) if sc else None, ms_min=float(np.min(sc)) if sc else None)
if S.adapt:                                                                          # 반복 분포 (프레임별 뉴턴·PCG 합; 첫 warm 프레임 빼지 않음)
    its = [json.loads(l_).get("it") for l_ in open(os.path.join(a.out, "fem.jsonl"))]
    its = [q for q in its if q]
    if its:
        for kk in ("nN", "nP", "capN", "capP"):
            arr = np.array([q[kk] for q in its], dtype=float)
            summ["it_" + kk] = dict(mean=float(arr.mean()), p50=float(np.median(arr)), p90=float(np.percentile(arr, 90)), max=float(arr.max()), min=float(arr.min()))
summ["pk"] = getattr(S, "pk_info", None)
summ["ms_max"] = float(np.max(sc)) if sc else None
summ["ms_p90"] = float(np.percentile(sc, 90)) if sc else None
import hashlib                                                                        # noqa: E402
summ["x_sha1"] = hashlib.sha1(S.x_orig().cpu().numpy().tobytes()).hexdigest()[:10]     # 끝 상태 해시 (결정성 확인)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None and div_at is None:
    torch.save({"pred": PRED, **({"F": PREDF} if PREDF is not None else {})}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} 실시간 판 셀 {a.cell} K {a.K} 뉴턴 {a.newton} PCG {a.cg}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['ms_mean']} ms (재생 GPU 시간)", flush=True)
