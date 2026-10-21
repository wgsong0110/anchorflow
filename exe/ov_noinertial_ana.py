"""관성 예측 없이 시작한 출력만 최적화 vs 기존(관성 예측 시작) 비교 집계 (사용자 10-21, exe/ov_noinertial.sh 가 부른다).

입력 (--dir):
  stop_{z,n}_<T>.jsonl          롤아웃 프레임별 L-BFGS 기록 (AF_OV_STOP_LOG): 반복·평가·종료 사유·시작점 크기·E0/E1
  ms_{p,z,n}_<T>.json           **같은 상태**(기존 관성 예측 롤아웃 덤프의 프레임 시작 상태 80 개)에서 세 시작점으로 푼 E (AF_OV_MIN_STATES)
  stop_ms_{p,z,n}_<T>.jsonl     그 실행들의 L-BFGS 기록
  v_{z,n}_<T>.log               롤아웃 로그 (위치 오차)
  roll_ov_{z,n}_<T>.pt          롤아웃 덤프
--spstick: 기존 출력만 최적화 (err_ov_<T>.txt, roll_ov_<T>.pt, v_ov_<T>.log)
출력: summary.json, summary.txt
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--spstick", required=True)
ap.add_argument("--trajs", nargs="+", default=["L000", "0004"])
ap.add_argument("--node_h", type=float, default=0.028284)
a = ap.parse_args()
D, S = a.dir, a.spstick
NAME = {"p": "관성 예측 시작(기존)", "z": "0 시작", "n": "작은 잡음 시작"}


def err_of(log, T):
    if not os.path.exists(log):
        return None
    m = None
    for ln in open(log, errors="replace"):
        if f"롤아웃] {T}" in ln:
            q = re.search(r"평균 ([0-9.]+)%", ln)
            if q:
                m = float(q.group(1))
    return m


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def stop_sum(rec):
    if not rec:
        return None
    it = np.array([r["iters"] for r in rec])
    ev = np.array([r["evals"] for r in rec])
    return dict(frames=len(rec), iters_mean=float(it.mean()), iters_min=int(it.min()), iters_max=int(it.max()),
                evals_mean=float(ev.mean()), evals_max=int(ev.max()),
                why=dict(Counter(r["why"] for r in rec)),
                init_absmax_first=rec[0]["init_absmax"], init_rms_mean=float(np.mean([r["init_rms"] for r in rec])),
                E0_mean=float(np.mean([r["E0"] for r in rec])), E1_mean=float(np.mean([r["E1"] for r in rec])),
                per_frame=[dict(t=r["t"], iters=r["iters"], evals=r["evals"], why=r["why"], E0=r["E0"], E1=r["E1"],
                                init_rms=r["init_rms"]) for r in rec])


out, txt = {}, []
for T in a.trajs:
    o = {}
    o["err"] = {"p": err_of(f"{S}/v_ov_{T}.log", T), "z": err_of(f"{D}/v_z_{T}.log", T), "n": err_of(f"{D}/v_n_{T}.log", T)}
    o["roll_stop"] = {k: stop_sum(jl(f"{D}/stop_{k}_{T}.jsonl")) for k in ("z", "n")}
    o["ms_stop"] = {k: stop_sum(jl(f"{D}/stop_ms_{k}_{T}.jsonl")) for k in ("p", "z", "n")}
    ms = {k: (json.load(open(f"{D}/ms_{k}_{T}.json")) if os.path.exists(f"{D}/ms_{k}_{T}.json") else None) for k in ("p", "z", "n")}
    o["same_state"] = {}
    if ms["p"]:
        Ep = np.array([r["E_min"] for r in ms["p"]]); E0p = np.array([r["E_init"] for r in ms["p"]])
        o["same_state"]["p_E_min_mean"] = float(Ep.mean()); o["same_state"]["p_E_init_mean"] = float(E0p.mean())
        for k in ("z", "n"):
            if not ms[k]:
                continue
            Ek = np.array([r["E_min"] for r in ms[k]]); E0k = np.array([r["E_init"] for r in ms[k]])
            n = min(len(Ek), len(Ep))
            rel = (Ek[:n] - Ep[:n]) / np.abs(Ep[:n])                         # 기존 해 E 대비 상대 차
            frac = (Ek[:n] - Ep[:n]) / np.maximum(E0p[:n] - Ep[:n], 1e-30)   # 기존이 시작점에서 내려간 양 대비
            o["same_state"][k] = dict(
                states=n, E_init_mean=float(E0k[:n].mean()), E_min_mean=float(Ek[:n].mean()),
                rel_median=float(np.median(rel)), rel_absmax=float(np.abs(rel).max()), rel_mean=float(rel.mean()),
                n_lower=int((rel < 0).sum()), n_within_1e4=int((np.abs(rel) < 1e-4).sum()), n_within_1e3=int((np.abs(rel) < 1e-3).sum()),
                n_within_1e2=int((np.abs(rel) < 1e-2).sum()),
                frac_median=float(np.median(frac)), frac_max=float(frac.max()),
                per_state=[dict(state=i, E_init=float(E0k[i]), E_min=float(Ek[i]), E_min_p=float(Ep[i]), rel=float(rel[i]))
                           for i in range(n)])
    # 롤아웃 궤적이 기존과 얼마나 갈라지는가: 프레임별 입자 평균 거리 (노드 간격 h 단위)
    o["traj_gap"] = {}
    fp = f"{S}/roll_ov_{T}.pt"
    if os.path.exists(fp):
        P = torch.load(fp, map_location="cpu", weights_only=False, mmap=True)["pred"]
        for k in ("z", "n"):
            fk = f"{D}/roll_ov_{k}_{T}.pt"
            if not os.path.exists(fk):
                continue
            K = torch.load(fk, map_location="cpu", weights_only=False, mmap=True)["pred"]
            g = [float((K[i].float() - P[i].float()).norm(dim=-1).mean()) / a.node_h for i in range(min(len(K), len(P)))]
            o["traj_gap"][k] = dict(per_frame_h=g, f1=g[0], f10=g[min(9, len(g) - 1)], f40=g[min(39, len(g) - 1)],
                                    last=g[-1], max=max(g))
    out[T] = o
    e = o["err"]
    txt.append(f"=== {T} ===")
    txt.append("위치 오차: " + ", ".join(f"{NAME[k]} {e[k]}%" for k in ("p", "z", "n")))
    for k in ("z", "n"):
        s = o["roll_stop"][k]
        if s:
            txt.append(f"[롤아웃 {NAME[k]}] 프레임 {s['frames']}, 반복 평균 {s['iters_mean']:.1f} (최소 {s['iters_min']} 최대 {s['iters_max']}), "
                       f"평가 평균 {s['evals_mean']:.1f}, 종료 사유 {s['why']}, 시작점 |u|max(첫 프레임) {s['init_absmax_first']:.3e}, "
                       f"E 평균 {s['E0_mean']:.4e} -> {s['E1_mean']:.4e}")
    for k in ("p", "z", "n"):
        s = o["ms_stop"][k]
        if s:
            txt.append(f"[같은 상태 {NAME[k]}] 상태 {s['frames']}, 반복 평균 {s['iters_mean']:.1f} (최소 {s['iters_min']} 최대 {s['iters_max']}), "
                       f"종료 사유 {s['why']}, 시작점 rms {s['init_rms_mean']:.3e}, E 평균 {s['E0_mean']:.4e} -> {s['E1_mean']:.4e}")
    ss = o["same_state"]
    for k in ("z", "n"):
        if k in ss:
            q = ss[k]
            txt.append(f"[같은 상태 해 E 비교 {NAME[k]} vs 기존] 상태 {q['states']}: 상대 차 중앙 {q['rel_median']:+.3e}, 평균 {q['rel_mean']:+.3e}, "
                       f"|최대| {q['rel_absmax']:.3e}; 더 낮음 {q['n_lower']}, |상대 차|<1e-4 {q['n_within_1e4']}, <1e-3 {q['n_within_1e3']}, <1e-2 {q['n_within_1e2']}; "
                       f"기존 하강량 대비 중앙 {q['frac_median']:+.3e} 최대 {q['frac_max']:+.3e}")
    for k, g in o["traj_gap"].items():
        txt.append(f"[궤적 차 {NAME[k]} vs 기존] 입자 평균 거리/h: 프레임1 {g['f1']:.3e}, 10 {g['f10']:.3e}, 40 {g['f40']:.3e}, 마지막 {g['last']:.3e}, 최대 {g['max']:.3e}")
json.dump(out, open(f"{D}/summary.json", "w"), ensure_ascii=False, indent=1)
open(f"{D}/summary.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
print("OVNOIN_ANA_DONE")
