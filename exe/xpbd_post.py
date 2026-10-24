"""XPBD 비교 정리 (사용자 10-24): 정답 | 출력만 최적화 | XPBD (N, K) -- 그림·표·json·tex 조각.

  python exe/xpbd_post.py --scene lego_L000 --gt_pt L000.pt --ov_dump roll_ov_L000.pt --ov_psi psi_ov.npy --gt_psi psi_gt.npy \
      --run "n1k1=DIR/roll_xpbd.pt:DIR/summary_xpbd.json:DIR/stats_xpbd.jsonl:psi.npy" ... --out_json lego.json
  python exe/xpbd_post.py --merge lego.json ficus.json --fig cmp.png --table cmp.md --tex cmp.tex

지표 (모든 칸 같은 정의):
  위치 오차 = 프레임별 입자 평균 |x − x_gt| / EXT (DM.frame_err, EXT = 정답 0 프레임 대각선), 평균은 끝낸 프레임 전부
  탄성 에너지 E_el(t) = Σ V_p ψ_FCR(F_p) (psi_color 가 남긴 ψ, 씬 물성 jelly 고정 공동회전 -- 정답과 같은 식), V_p = 질량/밀도
  운동 K = ½ Σ m |v|², 중력 위치 U = −Σ m g·(x − x0), 총 = K + E_el + U
  ficus 진동 (ficosc.py 와 같은 신호): 평균 변위 벡터의 첫 주성분 축 성분 sig(t) (대각선 대비), 주기 = 영점 통과 간격 평균 x 2,
        멈춤 = 프레임 속도 (평균 |Δx|/대각선) 가 최대의 5 % 아래로 내려가 끝까지 머무는 첫 프레임
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="")
ap.add_argument("--gt_pt", default="")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--ov_dump", default="")
ap.add_argument("--ov_psi", default="")
ap.add_argument("--gt_psi", default="")
ap.add_argument("--run", action="append", default=[], help="키=덤프:요약json:통계jsonl:psi.npy")
ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--out_json", default="")
ap.add_argument("--merge", nargs="*", default=[])
ap.add_argument("--fig", default="")
ap.add_argument("--table", default="")
ap.add_argument("--tex", default="")
ap.add_argument("--commit", default="")
a = ap.parse_args()

LAB = {"gt": "정답 (PG)", "ov": "출력만 최적화", "n1k1": "XPBD N=1 K=1", "n1k10": "XPBD N=1 K=10", "n4k1": "XPBD N=4 K=1",
       "n16k1": "XPBD N=16 K=1", "n64k1": "XPBD N=64 K=1"}
COL = {"gt": "#222222", "ov": "#d9480f", "n1k1": "#1c7ed6", "n1k10": "#7048e8", "n4k1": "#2b8a3e", "n16k1": "#e8590c", "n64k1": "#c2255c"}


def osc(X, diag):
    """ficosc.py analyze 와 같은 신호 -> sig, speed, 주기[프레임], 멈춤 프레임."""
    D = X - X[0]
    mm = D.mean(1) / diag
    _, _, vt = np.linalg.svd(mm - mm.mean(0), full_matrices=False)
    sig = mm @ vt[0]
    sig = sig * np.sign(sig[np.argmax(np.abs(sig))])
    sp = np.r_[0, np.linalg.norm(np.diff(X, axis=0), axis=2).mean(1) / diag]
    z = [i + sig[i] / (sig[i] - sig[i + 1]) for i in range(len(sig) - 1) if sig[i] != 0 and np.sign(sig[i]) != np.sign(sig[i + 1])]
    per = float(2 * np.mean(np.diff(z))) if len(z) >= 2 else None
    thr = 0.05 * sp.max()
    above = np.nonzero(sp > thr)[0]
    stop = int(above[-1] + 1) if len(above) and above[-1] + 1 < len(sp) else None
    return dict(sig=sig.tolist(), speed=sp.tolist(), zero_cross=[float(q) for q in z], period_frames=per, stop_frame=stop)


def one_scene():
    from anchorflow import deform_model as DM
    DM.REGION_DENSITY[0] = bool(a.region_density)
    d = torch.load(a.gt_pt, map_location="cpu", weights_only=False, mmap=True)
    cfg = d["cfg"]
    m = DM.traj_mass(d, "cpu").double()
    rho = d.get("_rho")
    Vp = (m / (rho.double() if rho is not None else float(cfg["density"]))).numpy()
    mn = m.numpy()
    EXT = float(d["_ext"]); fd = float(cfg["frame_dt"])
    g = np.array(cfg["g"], dtype=np.float64)
    X0 = d["x"][0].double().numpy()
    T = a.frames
    Xg = np.stack([d["x"][t].double().numpy() for t in range(T + 1)])
    diag = np.linalg.norm(X0.max(0) - X0.min(0))
    R = dict(scene=a.scene, EXT=EXT, frame_dt=fd, frames=T, runs={})

    def ener(X, Vv, psi_path, nf):
        ps = np.load(psi_path, mmap_mode="r") if psi_path and os.path.exists(psi_path) else None
        K, E, U = [], [], []
        for t in range(1, nf + 1):
            K.append(0.5 * float((mn * (Vv[t - 1] ** 2).sum(-1)).sum()))
            E.append(float((Vp * np.asarray(ps[t], dtype=np.float64)).sum()) if ps is not None and t < ps.shape[0] else float("nan"))
            U.append(-float((mn * ((X[t] - X0) @ g)).sum()))
        return K, E, U

    # 정답
    Vg = np.stack([d["v"][t].double().numpy() for t in range(1, T + 1)])
    K, E, U = ener(Xg, Vg, a.gt_psi, T)
    ps0 = np.load(a.gt_psi, mmap_mode="r")
    E0 = float((Vp * np.asarray(ps0[0], dtype=np.float64)).sum())
    K0 = 0.5 * float((mn * (d["v"][0].double().numpy() ** 2).sum(-1)).sum())
    R["E0"] = dict(K=K0, E=E0)
    R["runs"]["gt"] = dict(err=[0.0] * T, K=K, E=E, U=U, frames_done=T, osc=osc(Xg, diag))
    del Vg

    def add(key, dump, psi_path, summ=None, stats=None):
        r = torch.load(dump, map_location="cpu", weights_only=False, mmap=True)
        P = r["pred"]; nf = min(P.shape[0], T)
        X = np.concatenate([X0[None], np.stack([P[t].double().numpy() for t in range(nf)])])
        err = [float(np.linalg.norm(X[t + 1] - Xg[t + 1], axis=1).mean() / EXT) for t in range(nf)]
        Vv = np.stack([r["v"][t].double().numpy() for t in range(nf)]) if r.get("v") is not None else np.zeros((nf,) + X0.shape)
        K, E, U = ener(X, Vv, psi_path, nf)
        dets = [torch.linalg.det(r["F"][t].double()) for t in range(nf)] if r.get("F") is not None else []
        out = dict(err=err, err_mean=float(np.mean(err)), frames_done=nf, K=K, E=E, U=U,
                   detF_min=float(min(float(q.min()) for q in dets)) if dets else None,
                   inv_particles_max=int(max(int((q <= 0).sum()) for q in dets)) if dets else None,
                   inv_frames=int(sum(int((q <= 0).any()) for q in dets)) if dets else None, osc=osc(X, diag), dump=dump)
        if summ:
            s = json.load(open(summ)); out.update(diverged_at=s.get("diverged_at"), N=s["N"], K_it=s["K"], frames_req=s["frames_req"],
                                                   err_mean_run=s.get("err_mean"), detJ_min=s.get("detJ_min"), tets_inv_max=s.get("tets_inv_max"),
                                                   mu_s=s.get("mu_s"), vdamp_sub=s.get("vdamp_sub"))
        if stats:
            st = [json.loads(q) for q in open(stats)]
            ok = [q for q in st if not q.get("diverged")]
            out["E_fcr64"] = [q["E_fcr64"] for q in ok]
            out["floor_min_s"] = min((q["floor_min_s"] for q in ok if q.get("floor_min_s") is not None), default=None)
        R["runs"][key] = out
        print(f"[{a.scene}] {key}: 프레임 {nf}, 위치 오차 평균 {100 * out['err_mean']:.3f}%"
              + (f" (실행 기록 {100 * out['err_mean_run']:.3f}%)" if out.get("err_mean_run") is not None else "")
              + f", 탄성 평균 {np.nanmean(E):.4e} J, det F 최소 {out['detF_min']}, 뒤집힌 입자 최대 {out['inv_particles_max']}"
              + (f", 발산 {out.get('diverged_at')}" if out.get("diverged_at") else ""), flush=True)

    add("ov", a.ov_dump, a.ov_psi)
    for s in a.run:
        k, rest = s.split("=", 1)
        dump, summ, stats, psi = rest.split(":")
        add(k, dump, psi, summ, stats)
    json.dump(R, open(a.out_json, "w"), ensure_ascii=False)
    print(f"[저장] {a.out_json}", flush=True)


def merge():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in fm.findSystemFonts():
        if "Noto" in f and ("CJK" in f or "KR" in f):
            fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    SC = [json.load(open(p)) for p in a.merge]
    fig, ax = plt.subplots(len(SC), 3, figsize=(19, 5 * len(SC)), squeeze=False)
    lines = ["# XPBD 한 스텝 vs 출력만 최적화 (사용자 10-24)", "",
             f"커밋 {a.commit}. XPBD = 우리 표현(세계 고정 Kuhn 사면체 격자, 입자 barycentric, x·v·F) 그대로, 한 프레임을 서브스텝 N x 반복 K 의 XPBD 로 "
             "(사면체마다 fixed corotated 회전·부피 구속 결합 풀이, 48 색 가우스-자이델, 바닥 = 사면체별 최심 입자 비침투 + 정적 마찰 μ_s = ∞). "
             "위치 오차 = 입자 평균 |x − x_gt| / 대각선, 끝낸 프레임 평균. 탄성 = Σ V ψ_FCR (정답과 같은 식).", ""]
    for si, R in enumerate(SC):
        sc = R["scene"]; fd = R["frame_dt"]; T = R["frames"]
        keys = [k for k in ["ov", "n1k1", "n1k10", "n4k1", "n16k1", "n64k1"] if k in R["runs"]]
        lines += [f"## {sc} ({T} 프레임)", "",
                  "| 실행 | 위치 오차 평균 | 끝낸 프레임 | 발산 | 탄성 에너지 평균 [J] | det F 최소 | 뒤집힌 입자 최대 (프레임 수) | 총 에너지 처음 → 끝 [J] |"
                  + (" 주기 [s] | 멈춤 [s] |" if sc.startswith("ficus") else ""),
                  "|---|---|---|---|---|---|---|---|" + ("---|---|" if sc.startswith("ficus") else "")]
        E0 = R["E0"]["K"] + R["E0"]["E"]
        for k in ["gt"] + keys:
            r = R["runs"][k]
            tot = np.array(r["K"]) + np.array(r["E"]) + np.array(r["U"])
            em = np.nanmean(r["E"])
            row = (f"| {LAB[k]} | {'—' if k == 'gt' else '%.2f%%' % (100 * r['err_mean'])} | {r['frames_done']}/{T} | "
                   f"{'—' if k in ('gt', 'ov') else (('프레임 %d' % r['diverged_at']) if r.get('diverged_at') else '없음')} | {em:.4e} | "
                   f"{'—' if r.get('detF_min') is None else '%.4f' % r['detF_min']} | "
                   f"{'—' if r.get('inv_particles_max') is None else '%d (%d)' % (r['inv_particles_max'], r['inv_frames'])} | "
                   f"{E0:.4e} → {tot[-1]:.4e} |")
            if sc.startswith("ficus"):
                o = r["osc"]
                row += (f" {'—' if o['period_frames'] is None else '%.3f' % (o['period_frames'] * fd)} | "
                        f"{'끝까지 움직임' if o['stop_frame'] is None else '%.2f' % (o['stop_frame'] * fd)} |")
            lines.append(row)
        lines.append("")
        A = ax[si]
        for k in keys:
            r = R["runs"][k]
            A[0].plot(np.arange(1, len(r["err"]) + 1), 100 * np.array(r["err"]), c=COL[k], lw=2.2 if k == "ov" else 1.4,
                      label=f"{LAB[k]} {100 * r['err_mean']:.2f}%" + (f" (발산 {r['diverged_at']})" if r.get("diverged_at") else ""))
        A[0].set_xlabel("프레임"); A[0].set_ylabel("위치 오차 (%)"); A[0].set_title(f"{sc} 프레임별 위치 오차"); A[0].legend(fontsize=8)
        for k in ["gt"] + keys:
            r = R["runs"][k]
            tot = np.array(r["K"]) + np.array(r["E"]) + np.array(r["U"])
            A[1].plot(np.arange(1, len(tot) + 1), tot, c=COL[k], lw=2.2 if k == "gt" else 1.3, label=LAB[k])
        A[1].axhline(E0, c="gray", ls=":", label=f"처음 {E0:.3g} J")
        A[1].set_xlabel("프레임"); A[1].set_ylabel("J"); A[1].set_title(f"{sc} 총 에너지 (운동 + 탄성" + (" + 중력 위치)" if "lego" in sc else ")"))
        A[1].legend(fontsize=8)
        if sc.startswith("ficus"):
            for k in ["gt"] + keys:
                o = R["runs"][k]["osc"]
                A[2].plot(np.arange(len(o["sig"])), o["sig"], c=COL[k], lw=2.2 if k == "gt" else 1.3, label=LAB[k])
            A[2].set_title(f"{sc} 흔들림 (평균 변위의 주축 성분 / 대각선)"); A[2].set_xlabel("프레임")
        else:
            for k in ["gt"] + keys:
                r = R["runs"][k]
                A[2].plot(np.arange(1, len(r["E"]) + 1), r["E"], c=COL[k], lw=2.2 if k == "gt" else 1.3, label=LAB[k])
            A[2].set_title(f"{sc} 탄성 에너지 Σ V ψ_FCR"); A[2].set_xlabel("프레임"); A[2].set_ylabel("J"); A[2].set_yscale("log")
        A[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.fig, dpi=100)
    open(a.table, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # tex 조각
    PCT = "\\%"
    q = lambda s: s.replace("%", PCT).replace("_", "\\_")    # noqa: E731
    tx = [f"\\subsection*{{한 스텝을 XPBD 로 (2026-10-24, {a.commit})}}",
          "우리 표현(세계 고정 Kuhn 사면체 격자, 입자는 셀 안 barycentric, 상태 $x, v, F$)은 그대로 두고 한 프레임을 XPBD 로 진행했다. "
          "노드 질량 $M_i = \\sum_p \\lambda_{ip} m_p$, 사면체 $c$ 마다 증분 포텐셜 탄성과 같은 fixed corotated 구속 "
          "$C_R^2 = \\mathrm{tr}(J^\\top J \\bar B) - 2\\,\\mathrm{tr}(R_c^\\top J \\bar M) + 3$, $R_c = \\mathrm{polar}(J\\bar M)$, "
          "$C_V = \\sqrt{m_2}\\det J - m_1/\\sqrt{m_2}$ "
          "($\\bar B = \\sum V_p F_p F_p^\\top / V$, $\\bar M = \\sum V_p F_p / V$, $m_k = \\sum V_p (\\det F_p)^k / V$, $V = \\sum_{p\\in c} V_p$; "
          "모든 $F_p$ 가 같으면 $\\|F - R\\|_F$, $\\det F - 1$), 컴플라이언스 $1/(2\\mu V)$, $1/(\\lambda V)$ "
          "(씬 $E,\\nu$; $\\tfrac12 C_R^2/\\alpha_R + \\tfrac12 C_V^2/\\alpha_V = V\\psi_{\\mathrm{FCR}}$), $\\partial C_R/\\partial J = (J\\bar B - R_c \\bar M^\\top)/C_R$ "
          "(극분해 미분을 빼는 꼴이지만 이 경우 정확), 두 구속 결합 풀이, 48 색 가우스-자이델. 바닥은 사면체마다 가장 깊은 입자의 비침투 구속과 "
          "정적 마찰 $\\mu_s=\\infty$ (PG sticky 와 같은 뜻). 서브스텝 $N$, 반복 $K$, 서브스텝마다 $\\lambda = 0$, 끝에 $F_p \\leftarrow J_c F_p$.",
          ""]
    for R in SC:
        sc = R["scene"]
        keys = [k for k in ["ov", "n1k1", "n1k10", "n4k1", "n16k1", "n64k1"] if k in R["runs"]]
        tx += ["\\begin{center}", f"{q(sc)} ({R['frames']} 프레임)\\\\", "\\begin{tabular}{lcccc}", "\\hline",
               " & 위치 오차 평균 & 발산 & 탄성 에너지 평균 [J] & det $F$ 최소 \\\\", "\\hline"]
        for k in ["gt"] + keys:
            r = R["runs"][k]
            ep = "---" if k == "gt" else ("%.2f" % (100 * r["err_mean"])) + PCT
            tx.append(f"{LAB[k]} & {ep} & "
                      f"{'---' if k in ('gt', 'ov') else (('프레임 %d' % r['diverged_at']) if r.get('diverged_at') else '없음')} & "
                      f"{np.nanmean(r['E']):.3e} & {'---' if r.get('detF_min') is None else '%.3f' % r['detF_min']} \\\\")
        tx += ["\\hline", "\\end{tabular}", "\\end{center}"]
        if sc.startswith("ficus"):
            fd = R["frame_dt"]
            tx.append("ficus 흔들림: " + ", ".join(
                f"{LAB[k]} 주기 {'---' if R['runs'][k]['osc']['period_frames'] is None else '%.3f s' % (R['runs'][k]['osc']['period_frames'] * fd)}, "
                f"멈춤 {'끝까지 움직임' if R['runs'][k]['osc']['stop_frame'] is None else '%.2f s' % (R['runs'][k]['osc']['stop_frame'] * fd)}"
                for k in ["gt"] + keys) + ".")
    open(a.tex, "w").write("\n".join(tx) + "\n")
    print("XPBD_POST_DONE", flush=True)


if a.merge:
    merge()
else:
    one_scene()
