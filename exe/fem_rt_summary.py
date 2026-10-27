"""FEM 실시간 판 실행 요약 (doc/fem_realtime.md 표용): fem.jsonl + v_fem.log -> 오차, 벽시계/프레임 (프레임 2->80 누적 차 / 78), fps,
뉴턴 루프/프레임 중앙, 걸음당 구간 중앙 (첫 프레임 제외): P2G, E+기울기, 헤시안 (+ 사면체 부분·바닥 후보), 투영, CSR, PCG (반복), 선탐색,
루프 밖 (앞 프레임 루프 끝 -> 이번 루프 시작) 과 그중 G2P (끝 파이프라인 순전파), 사면체 적분점 근사 진단.
사용: python3 exe/fem_rt_summary.py <실행 폴더>...   (torch 불필요)"""
import sys, json, re, statistics as st


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


for d in sys.argv[1:]:
    R = [json.loads(l) for l in open(f"{d}/fem.jsonl")]
    log = open(f"{d}/v_fem.log", errors="replace").read()
    cum = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"프레임 (\d+)/\d+\s+서브스텝 \d+\s+누적 ([0-9.]+) s", log)}
    err = re.findall(r"\[롤아웃\] L000.*?평균 ([0-9.]+)%", log)
    n = max(cum) if cum else 0
    wall = (cum[n] - cum[2]) / (n - 2) if n > 2 and 2 in cum else float("nan")
    R1 = R[1:]
    S = [s for r in R1 for s in r["steps"]]
    ms = lambda k: 1e3 * med([s.get(k) for s in S])
    out = dict(run=d.rstrip("/").split("/")[-1], frames=len(R), err=float(err[-1]) if err else None, wall_s=wall, fps=1.0 / wall if wall == wall else None,
               loop_s_med=med([r["sec"] for r in R1]), steps_per_frame=mean([r["iters"] for r in R1]),
               conv_frames=sum(r["conv"] for r in R),
               per_step_ms=dict(p2g=ms("t_p2g"), fwd0=ms("t_fwd0"), eg=ms("t_eg"), hess=ms("t_hess"), proj=ms("t_proj"),
                                csr=1e3 * med([(s["t_asm"] - s["t_proj"]) if "t_asm" in s else None for s in S]),
                                pcg=ms("t_pcg"), pcg_it=med([s.get("cg") for s in S]), ls=ms("t_ls"), n_ls=med([s.get("n_ls") for s in S])),
               out_of_loop_ms=1e3 * med([r.get("t_out_prev") for r in R1[1:]]), g2p_ms=1e3 * med([r.get("t_g2p_prev") for r in R1[1:]]))
    hp = [s.get("t_hess_parts") for s in S if s.get("t_hess_parts")]
    if hp:
        out["hess_parts_ms"] = {k: med([h.get(k) for h in hp]) for k in hp[0]}
    tq = [s.get("tetq") for s in S if s.get("tetq")]
    if tq:
        out["tetq"] = dict(n_act=med([q["n_act"] for q in tq]), n_near=med([q["n_near"] for q in tq]), n_far=med([q["n_far"] for q in tq]))
        dg = [q["diag"] for q in tq if q.get("diag")]
        if dg:
            out["tetq"]["diag"] = {k: dict(mean=mean([x[k] for x in dg]), med=med([x[k] for x in dg]), max=max(x[k] for x in dg), min=min(x[k] for x in dg))
                                   for k in ("E_el_rel", "E0_el_rel", "dF_mean", "dF_max", "F_mean")}
            out["tetq"]["diag_t"] = {str(r["t"]): {k: round(r["steps"][0]["tetq"]["diag"][k], 4) for k in ("E_el_rel", "E0_el_rel", "dF_mean")}
                                     for r in R if r["steps"] and (r["steps"][0].get("tetq") or {}).get("diag") and r["t"] % 10 == 0}
        vi = [s.get("tetq_viol") for s in S if s.get("tetq_viol") is not None]
        if vi:
            out["tetq"]["viol"] = dict(mean=mean(vi), max=max(vi))
    print(json.dumps(out, ensure_ascii=False, indent=1))
