"""A5 (PG 공식 씬 plane·pillow2sofa) 서브스텝 안정성 결과 정리 -> json + LaTeX 조각 + 그림 (글자 한국어, 계산 시간 없음).

입력 (exe/stab_a5_driver.py 결과): stab/<씬>/res/pg_N1024 (기준), scan/<방법>_N<N> (거듭제곱 + 사이 정수),
robust/r<k>/ (반복 확정), vis/vis_a5.json (시각 지표), stab/a5/nconf.json (드라이버가 정한 확정 N).
판정은 exe/stab_is_stable.py 의 바닥 없는 씬 규칙과 같다 (전 프레임 완주, (E-E_ref)/E_ref ≤ 1, E_ref = 기준 최대 총에너지).
물리 정합성 (확정 N 의 실행 전부 평균, 통과 = 모든 실행이 문턱 안; stab_phys_confirmed.py 와 같은 문턱):
  뒤집힘 비율 0, 떨림 ≤ 2 × 기준, PG N1024 대비 Chamfer ≤ 5% L. 바닥 관통·자유낙하는 바닥·자유낙하가 없어 해당 없음.

  python exe/stab_a5_report.py --root /home/dkta/work/stab --out /home/dkta/work/stab/a5
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stab_metrics as sm                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/dkta/work/stab")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

SC = (("plane", "plane (metal, 400 프레임)"), ("pillow2sofa", "pillow2sofa (jelly, 300 프레임)"))
METH = [("pg", "PG"), ("gf", "GF"), ("ipg", "i-PG"), ("ovs", "우리")]
THR = dict(detneg=0.0, jit_mult=2.0, cd=0.05)
NCONF = json.load(open(f"{a.out}/nconf.json")) if os.path.exists(f"{a.out}/nconf.json") else {}


def energy(r):
    return r["rows"]["ke"] + r["rows"]["pe"]


def done(d, m, n):
    return os.path.exists(f"{d}/{m}_N{n}.done")


R = {}
for S, _ in SC:
    o = f"{a.root}/{S}"
    cfg = json.load(open(f"{o}/config.json"))
    dt = float(cfg["frame_dt"]); g = float(cfg["g"][2]); dxg = float(cfg["grid_lim"]) / int(cfg["n_grid"])
    ref = sm.load(f"{o}/res", "pg", 1024)
    X0 = ref["X"]; L = float(np.linalg.norm(X0[0].max(0) - X0[0].min(0)))
    e_ref = float(np.nanmax(np.abs(energy(ref))))

    def judge(r):
        if r is None or r["status"] == "진행":
            return None
        if r["status"] != "끝":
            return 0
        E = energy(r)
        return int(bool(np.isfinite(E).all()) and float(np.max((E - e_ref) / e_ref)) <= 1.0 and len(E) == len(energy(ref)))

    def mets(r):
        with np.errstate(all="ignore"):
            mt = sm.metrics(r, X0, L, -1e9, g, dt, 1.0, dxg)
        E = energy(r)
        return dict(e_ratio=float(np.nanmax((E - e_ref) / e_ref)), detneg=mt.get("detneg_max"),
                    jit=mt.get("jitter_d2"), cd=mt.get("cd_mpm"))

    with np.errstate(all="ignore"):
        refm = mets(ref)
    R[S] = dict(L=L, e_ref=e_ref, ref=refm, methods={})
    for m, name in METH:
        sc = f"{o}/scan"
        ns = sorted({int(re.match(rf"{m}_N(\d+)\.done$", os.path.basename(p)).group(1))
                     for p in glob.glob(f"{sc}/{m}_N*.done")})
        scan = {}
        for n in ns:
            r = sm.load(sc, m, n)
            v = judge(r)
            rec = dict(stable=v)
            if v == 1:
                rec.update(mets(r))
            scan[n] = rec
        smin = next((n for n in ns if scan[n]["stable"] == 1), None)
        nc = NCONF.get(S, {}).get(m)
        rec = dict(scan=scan, scan_min=smin, confirmed=nc)
        if isinstance(nc, int):
            runs, bad = [], 0
            for d in [sc] + [f"{o}/robust/r{k}" for k in (1, 2, 3)]:
                if not done(d, m, nc):
                    continue
                r = sm.load(d, m, nc); v = judge(r)
                if v != 1:
                    bad += 1; continue
                runs.append(mets(r))
            ch = {}
            for k in ("detneg", "jit", "cd", "e_ratio"):
                vs = [q[k] for q in runs if q[k] is not None]
                ok = {"detneg": lambda x: x <= THR["detneg"], "jit": lambda x: x <= THR["jit_mult"] * refm["jit"],
                      "cd": lambda x: x <= THR["cd"], "e_ratio": lambda x: x <= 1.0}[k]
                ch[k] = dict(value=float(np.mean(vs)) if vs else None, ok=(all(ok(x) for x in vs) if vs else None), per_run=vs)
            rec.update(runs=len(runs), diverged=bad, check=ch)
        R[S]["methods"][m] = rec
    vj = f"{o}/vis/vis_a5.json"
    R[S]["vis"] = json.load(open(vj)) if os.path.exists(vj) else {}
json.dump(R, open(f"{a.out}/a5.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------------------------------------- 그림
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib import font_manager as fm                   # noqa: E402

for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Noto Sans KR"):
    if any(fam in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams["axes.unicode_minus"] = False
COL = {"pg": "#2a78d6", "gf": "#eb6834", "ipg": "#1baf7a", "ovs": "#eda100"}
MK = {"pg": "o", "gf": "s", "ipg": "^", "ovs": "D"}
PAN = (("e_ratio", "총에너지 최대 변화 (E−E기준)/E기준", 1.0), ("detneg", "뒤집힌 입자 비율 (%)", 100.0),
       ("jit", "떨림: 위치 2차 차분 / L (%)", 100.0), ("cd", "PG N=1024 대비 Chamfer / L (%)", 100.0))
fig, axs = plt.subplots(len(SC), len(PAN), figsize=(4.2 * len(PAN), 3.4 * len(SC)), squeeze=False)
for i, (S, title) in enumerate(SC):
    for j, (k, lab, mul) in enumerate(PAN):
        ax = axs[i, j]
        for m, name in METH:
            sc = R[S]["methods"][m]["scan"]
            ns = [n for n in sorted(sc) if sc[n]["stable"] == 1 and sc[n].get(k) is not None]   # 발산 점은 뺀다
            if ns:
                ax.plot(ns, [sc[n][k] * mul for n in ns], color=COL[m], marker=MK[m], ms=7, lw=2, label=name)
        if k != "e_ratio":
            rv = R[S]["ref"].get(k)
            if rv is not None:
                ax.axhline(rv * mul, color="#888888", lw=1, ls="--", label="기준 PG N=1024")
        ax.set_xscale("log", base=2); ax.set_xlabel("프레임당 서브스텝 N"); ax.set_ylabel(lab, fontsize=9)
        ax.set_title(f"{title}", fontsize=10); ax.grid(alpha=0.25, lw=0.6)
        if i == 0 and j == 0:
            ax.legend(fontsize=8, frameon=False)
fig.suptitle("A5: N 에 따른 물리 지표 (정수 스캔 실행, 발산한 실행 제외)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{a.out}/stab_a5_metrics.png", dpi=150)

# ---------------------------------------------------------------------------------------------- LaTeX 조각


def f(v, mul=100.0, p=2):
    return "--" if v is None else f"{mul * v:.{p}f}"


def mark(o):
    return "" if o is None else (r"\,\checkmark" if o else r"\,$\times$")


def tried(sc):
    return ", ".join(f"{n}{'' if sc[n]['stable'] == 1 else '✗'}" for n in sorted(sc))


T = [r"\section*{PG 공식 씬: plane · pillow2sofa (A5)}",
     r"PhysGaussian 공식 config 그대로: plane (metal, 고정 상자 2 + 회전 원기둥 2, 중력 0, 400 프레임 $\times$ 0.01 s), "
     r"pillow2sofa (jelly, 소파 고정 + 임시 지지대 2, $g=-5$, 300 프레임 $\times$ 0.02 s). 0 프레임 = 공식 gs\_simulation 채우기. "
     r"우리 = 기본 실행 조합 + 공식 경계조건(노드 하드 구속). "
     r"\textbf{판정}: 전 프레임을 NaN·오류 없이 끝내고 총에너지(운동+위치) $(E-E_\text{ref})/E_\text{ref} \le 1$, "
     r"$E_\text{ref}$ = PG $N{=}1024$ 의 최대 총에너지 (바닥이 없어 관통·자유낙하 판정은 없다). "
     r"\textbf{최소 $N$}: 2 의 거듭제곱 1--128 을 작은 것부터, 처음 안정한 거듭제곱과 직전 발산한 거듭제곱 사이 정수를 아래부터. "
     r"\textbf{확정}: 같은 $N$ 3 번 모두 안정 (하나라도 발산하면 $N{+}1$). 128 까지 안정이 없으면 ``$>128$'' (더 큰 $N$ 은 돌리지 않았다).",
     "",
     r"\begin{table}[h]\centering\small",
     r"\caption{A5 최소 서브스텝 수 $N$ -- 확정값 (괄호: 정수 스캔 1 회 때). 아래 줄: 돌린 $N$ ($\times$ 표시 = 발산).}",
     r"\begin{tabular}{lcccc}", r"\toprule", r"씬 & PG & GF & i-PG & 우리 \\", r"\midrule"]
for S, _ in SC:
    M = R[S]["methods"]
    T.append(f"{S} & " + " & ".join(f"{M[m]['confirmed'] if M[m]['confirmed'] is not None else '--'} ({M[m]['scan_min'] or '--'})"
                                      for m, _ in METH) + r" \\")
T += [r"\bottomrule", r"\end{tabular}", r"\\[4pt]\footnotesize"]
for S, _ in SC:
    M = R[S]["methods"]
    T.append(f"{S}: " + "; ".join(f"{name} " + ", ".join(
        f"{n}" + ("" if M[m]["scan"][n]["stable"] == 1 else r"$^\times$") for n in sorted(M[m]["scan"])) for m, name in METH) + r"\\")
T += [r"\end{table}", "",
      r"\begin{table}[h]\centering\small",
      r"\caption{A5 확정 최소 $N$ 에서 물리 정합성. 값 = 그 $N$ 의 실행 전부 평균, 통과($\checkmark$) = 모든 실행이 문턱 안. "
      r"기준 = PG $N{=}1024$. 문턱: 뒤집힘 비율 0, 떨림 $\le 2\times$기준, Chamfer $\le 5\%L$, 에너지 비 $\le 1$. "
      r"값은 \% (떨림·Chamfer 는 $L$ 대비), 에너지 비는 그대로.}",
      r"\begin{tabular}{llcccccc}", r"\toprule", r"씬 & 방법 & $N$ & 실행 & 뒤집힘 & 떨림 & Chamfer & 에너지 비 \\", r"\midrule"]
for S, _ in SC:
    rf = R[S]["ref"]
    T.append(f"{S} & 기준 PG & 1024 & 1 & {f(rf['detneg'])} & {f(rf['jit'])} & 0 & {f(rf['e_ratio'], 1.0)} \\\\")
    for m, name in METH:
        rr = R[S]["methods"][m]
        if not rr.get("check"):
            T.append(f" & {name} & {rr['confirmed'] if rr['confirmed'] is not None else '--'} & & & & & \\\\"); continue
        c = rr["check"]
        runs = f"{rr['runs']}" + (f" (발산 {rr['diverged']})" if rr["diverged"] else "")
        T.append(f" & {name} & {rr['confirmed']} & {runs} & {f(c['detneg']['value'])}{mark(c['detneg']['ok'])} & "
                 f"{f(c['jit']['value'])}{mark(c['jit']['ok'])} & {f(c['cd']['value'])}{mark(c['cd']['ok'])} & "
                 f"{f(c['e_ratio']['value'], 1.0)}{mark(c['e_ratio']['ok'])} \\\\")
    T.append(r"\midrule")
T[-1] = r"\bottomrule"
T += [r"\end{tabular}", r"\end{table}", "",
      r"\begin{table}[h]\centering\small",
      r"\caption{A5 확정 최소 $N$ 에서 시각 지표 (같은 렌더러 rep\_ip, F 반영 3DGS; 기준 = PG $N{=}1024$ 렌더, 전 프레임 평균). 괄호: 마지막 프레임 PSNR.}",
      r"\begin{tabular}{llcccc}", r"\toprule", r"씬 & 방법 & $N$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\", r"\midrule"]
VN = {"pg": "PG", "gf": "GF", "ipg": "iPG", "ovs": "ours"}
for S, _ in SC:
    first = True
    for m, name in METH:
        n = R[S]["methods"][m]["confirmed"]
        v = R[S]["vis"].get(f"{VN[m]}_N{n}") if isinstance(n, int) else None
        lead = S if first else ""
        first = False
        if v is None:
            T.append(f"{lead} & {name} & {n if n is not None else '--'} & -- & -- & -- \\\\"); continue
        T.append(f"{lead} & {name} & {n} & {v['psnr']:.2f} ({v['psnr_last']:.1f}) & {v['ssim']:.3f} & {v['lpips']:.3f} \\\\")
    T.append(r"\midrule")
T[-1] = r"\bottomrule"
T += [r"\end{tabular}", r"\end{table}", "",
      r"\begin{figure}[h]\centering", r"\includegraphics[width=\linewidth]{stab_a5_metrics.png}",
      r"\caption{A5: $N$ 에 따른 물리 지표 (정수 스캔 실행, 발산한 실행 제외; 점선 = 기준 PG $N{=}1024$).}", r"\end{figure}", ""]
open(f"{a.out}/stab_a5_section.tex", "w").write("\n".join(T))
print("[보고]", f"{a.out}/a5.json", f"{a.out}/stab_a5_section.tex", f"{a.out}/stab_a5_metrics.png")
