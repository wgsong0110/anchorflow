"""파레토 비교 (doc/pareto.md): VR-GS 직접 구현 (lib/anchorflow/vrgs_xpbd.py) 실행기.

  --mode fit   설정 (서브스텝 N, 반복 수) 마다 E 를 논문 절차로 정함: 논문은 E 를 자유 낙하 충돌 변형에 맞춰 손으로 조정하고 "10 회 안에" 정한다 ->
               여기서는 정답 궤적 롤아웃 오차 (우리와 같은 정의, 평균 |x−x_gt|/EXT) 를 목적으로 log10 E 황금분할 탐색 10 회 (정답 E 의 1e-2..1e2 배).
               ν·ρ 는 정답 config 그대로 (논문은 E 와 물체 사이 상대 밀도만 조정; 물체 하나라 상대 밀도 없음).
  --mode run   fit 결과 E 로 설정마다 한 번 롤아웃 (단독 카드에서 시간 측정용): 프레임 GPU 시간 (서브스텝 N 회 + 가우시안 중심 갱신).
  --mode scale 논문 표 1 규모 대조: 케이지 정점 10,611 / 19,472 / 28,205 · 가우시안 1,295,223 / 2,525,891 / 304,875 에서 서브스텝 1 회 시간 (반복 1·5·11) 과 중심 갱신 시간.
"""
from __future__ import annotations

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
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import vrgs_xpbd as VG                                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--vmax", type=int, default=30000, help="케이지 정점 상한 (논문 10K~30K)")
ap.add_argument("--mode", choices=["fit", "run", "scale"], default="fit")
ap.add_argument("--set", nargs="+", default=["16x5"], help="설정 목록 NxIT (서브스텝 x 반복)")
ap.add_argument("--fit_json", default=None, help="run: 설정별 E (fit 출력 fit.json 들, 쉼표)")
ap.add_argument("--E", type=float, default=0.0, help="run: 고정 E (0 = fit_json)")
ap.add_argument("--nfit", type=int, default=10)
ap.add_argument("--span", type=float, default=2.0, help="탐색 폭 (정답 E 의 10^±span)")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--dump_F", action="store_true", help="run: 첫 설정 덤프에 입자 F (결속 사면체 F) 도 -> pred.pt {pred, F} (안정 서브스텝 표 렌더용, doc/pareto.md 절 S)")
a = ap.parse_args()
RTC.RAWPTR[0] = True
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho_p = d["_rho"].to(dev).double() if (a.region_density and "_rho" in d) else torch.full_like(m, float(cfg["density"]))
L = min(a.frames, int(d["x"].shape[0]) - 1)
FZ = None
if a.floor == "vel":
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    assert abs(float(pl["normal"][2]) - 1.0) < 1e-6, "바닥 법선이 z 가 아님"
    FZ = float(pl["point"][2])
BOX = []
if a.cfg_bc:
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") == "cuboid":
            assert max(abs(float(q)) for q in bc["velocity"]) == 0.0
            BOX.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]]))
DSC = float(cfg.get("grid_v_damping_scale", 1.0)) if a.pg_vdamp else 1.0
E_GT, NU = float(cfg["E"]), float(cfg["nu"])
GT = d["x"][1:L + 1].float()                                                          # CPU
x0 = d["x"][0].to(dev).float()


def parse(sv):
    n, it = sv.split("x")
    return int(n), int(it)


SETS = [parse(q) for q in a.set]
S = VG.VRGS(x0, d["v"][0], m, rho_p, E_GT, NU, cfg["g"], float(cfg["frame_dt"]), SETS[0][0], SETS[0][1], vmax=a.vmax, floor_z=FZ, fix_boxes=BOX,
            damp_scale=DSC, damp_dt=float(cfg["substep_dt"]))
INFO = S.info()
print(f"[VR-GS] {a.tag} 입자 {S.Np}, 케이지 정점 {S.nv} 사면체 {S.nt} 칸 {S.h:.5f} 색 {len(S.colors)} 고정 정점 {INFO['nfixed']}, 질량 케이지 {S.mass_cage:.4g} / 입자 {S.mass_part:.4g}, "
      f"결속 무게중심 최소 {S.bind_min:.2e}, 바닥 {FZ}, 감쇠 {DSC}, 정답 E {E_GT:g} ν {NU}, 프레임 {L}, EXT {EXT:.5f}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)


def rollout(E, keep=False):
    S.set_E(E); S.reset()
    errs, ms = [], []
    P = torch.empty(L, S.Np, 3) if keep else None
    PF = torch.empty(L, S.Np, 3, 3) if (keep and a.dump_F) else None
    div = None
    for f in range(L):
        t = S.frame()
        xo = S.xo
        gt = GT[f].to(dev)
        e = float((xo - gt).norm(dim=-1).mean()) / EXT
        if not math.isfinite(e) or e > 2.0:
            div = f + 1
            errs.append(float("nan"))
            break
        errs.append(e); ms.append(t)
        if keep:
            P[f] = xo.cpu()
        if PF is not None:
            PF[f] = S.F_part().cpu()
    sc = ms[a.warm:] if len(ms) > a.warm else ms
    return dict(E=E, err=(float(np.mean(errs)) if div is None else None), err_done=(float(np.nanmean(errs)) if np.isfinite(errs).any() else None),
                div=div, frames=len(errs), ms_mean=float(np.mean(sc)) if sc else None, ms_max=float(np.max(sc)) if sc else None, errs=errs), \
        (P if PF is None else (P, PF))


def obj(r):
    return r["err"] if r["err"] is not None else 1e9


if a.mode == "fit":
    res = []
    for (N, it) in SETS:
        S.set_setting(N, it)
        t0 = time.time()
        lo, hi = math.log10(E_GT) - a.span, math.log10(E_GT) + a.span
        gr = (math.sqrt(5) - 1) / 2
        c_, d_ = hi - gr * (hi - lo), lo + gr * (hi - lo)
        ev = []
        rc, _ = rollout(10 ** c_); ev.append(rc)
        rd, _ = rollout(10 ** d_); ev.append(rd)
        while len(ev) < a.nfit:
            if obj(rc) <= obj(rd):
                hi, d_, rd = d_, c_, rc
                c_ = hi - gr * (hi - lo)
                rc, _ = rollout(10 ** c_); ev.append(rc)
            else:
                lo, c_, rc = c_, d_, rd
                d_ = lo + gr * (hi - lo)
                rd, _ = rollout(10 ** d_); ev.append(rd)
        best = min(ev, key=obj)
        r = dict(N=N, iters=it, dt=S.dt, E_best=best["E"], err=best["err"], div=best["div"], ms_mean_nonexcl=best["ms_mean"],
                 evals=[dict(E=q["E"], err=q["err"], div=q["div"]) for q in ev], sec=time.time() - t0)
        res.append(r)
        print(f"  [맞춤] {a.tag} N {N} 반복 {it}: E {best['E']:.4g} (정답 {E_GT:g} 의 {best['E'] / E_GT:.3g} 배) 오차 "
              f"{'발산' if best['err'] is None else '%.3f%%' % (100 * best['err'])}, {r['sec']:.0f}s, 평가 "
              + " ".join(f"{q['E']:.3g}:{'x' if q['err'] is None else '%.2f' % (100 * q['err'])}" for q in ev), flush=True)
        json.dump(dict(info=INFO, tag=a.tag, E_gt=E_GT, res=res), open(os.path.join(a.out, "fit.json"), "w"), indent=1, ensure_ascii=False)
elif a.mode == "run":
    EB = {}
    if a.fit_json:
        for fj in a.fit_json.split(","):
            for r in json.load(open(fj))["res"]:
                EB[(r["N"], r["iters"])] = r["E_best"]
    res = []
    for (N, it) in SETS:
        E = a.E if a.E > 0 else EB.get((N, it))
        if E is None:
            print(f"  [없음] {a.tag} {N}x{it}: 맞춘 E 없음 -- 건너뜀", flush=True)
            continue
        S.set_setting(N, it)
        r, P = rollout(E, keep=(N, it) == SETS[0])
        r.update(N=N, iters=it, dt=S.dt)
        res.append(r)
        if isinstance(P, tuple):
            if r["div"] is None:
                torch.save({"pred": P[0], "F": P[1]}, os.path.join(a.out, "pred.pt"))
        elif P is not None:
            torch.save(P, os.path.join(a.out, f"pred_{N}x{it}.pt"))
        print(f"  [실행] {a.tag} N {N} 반복 {it} E {E:.4g}: 오차 {'발산 %d' % r['div'] if r['err'] is None else '%.3f%%' % (100 * r['err'])}, "
              f"프레임 {r['ms_mean']:.3f} ms (최대 {r['ms_max']:.3f})", flush=True)
        json.dump(dict(info=INFO, tag=a.tag, E_gt=E_GT, res=[{k: v for k, v in q.items() if k != "errs"} | dict(errs=q["errs"]) for q in res]),
                  open(os.path.join(a.out, "run.json"), "w"), indent=1, ensure_ascii=False)
else:                                                                                # scale
    PAPER = [("Horse", 10611, 1295223, 5, 115.4), ("Bear", 19472, 2525891, 5, 76.9), ("Fox", 28205, 304875, 1, 161.2)]
    out = []
    for (nm, nv, ng, it_p, fps) in PAPER:
        T = VG.VRGS(x0, d["v"][0], m, rho_p, E_GT, NU, cfg["g"], float(cfg["frame_dt"]), 1, 1, vmax=nv, floor_z=FZ, fix_boxes=BOX,
                    damp_scale=DSC, damp_dt=float(cfg["substep_dt"]))
        rep = (ng + T.Np - 1) // T.Np                                                 # 가우시안 수를 맞추려고 결속을 겹쳐 씀 (갱신 비용만 잼)
        T.tid = T.tid.repeat(rep)[:ng].contiguous(); T.bw = T.bw.repeat(rep, 1)[:ng].contiguous(); T.Np = ng
        T.xo = torch.zeros(ng, 3, device=dev)
        row = dict(scene=nm, nv_paper=nv, nv=T.nv, nt=T.nt, ncolor=len(T.colors), gauss=ng, iters_paper=it_p, fps_paper=fps, sub_ms={})
        for it in sorted({1, 5, 11, it_p}):
            T.set_setting(1, it); T.set_E(E_GT); T.capture()
            for _ in range(20):
                T.g_sub.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(200):
                T.g_sub.replay()
            e1.record(); torch.cuda.synchronize()
            row["sub_ms"][it] = e0.elapsed_time(e1) / 200
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(5):
            T.g_skin.replay()
        e0.record()
        for _ in range(50):
            T.g_skin.replay()
        e1.record(); torch.cuda.synchronize()
        row["skin_ms"] = e0.elapsed_time(e1) / 50
        row["max_sub_in_budget"] = (1000.0 / fps - row["skin_ms"]) / row["sub_ms"][it_p]
        out.append(row)
        print(f"  [규모] {nm}: 정점 {T.nv} (논문 {nv}) 사면체 {T.nt} 색 {len(T.colors)}, 서브스텝 1 회 " + " ".join(f"반복 {k} {v:.4f} ms" for k, v in row["sub_ms"].items())
              + f", 가우시안 {ng} 중심 갱신 {row['skin_ms']:.4f} ms -> 논문 {fps} fps (프레임 {1000 / fps:.2f} ms, 렌더 포함) 안에 서브스텝 최대 {row['max_sub_in_budget']:.1f} 회 (반복 {it_p})", flush=True)
        del T
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(a.out, "scale.json"), "w"), indent=1, ensure_ascii=False)
print("[끝]", flush=True)
