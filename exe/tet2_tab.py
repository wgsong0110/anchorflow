"""절 CM103 표·그림 (로컬, 읽기만): 판 디렉토리 (summary.json · t2.jsonl · period.txt) 를 무리로 묶어 위반 기록 표와 프레임별 그림 (한국어).

사용: python exe/tet2_tab.py <그림 디렉토리> <씬=무리이름:판[,판...];무리이름:판...> ...  > 표.md
- 프레임 = s.step 한 번 (같은 t 는 마지막 시도만), 걸음 = 그 프레임의 걸음 끝 행.
- 칸: det ≤ 0 사면체 수 (걸음 최대)·최소 det · 띠 밖 수·최대 넘침 (σ_lo/σ_min − 1, σ_max/σ_hi − 1) · 보폭 자른 반복 수 / 확인 수 · 최소 보폭 · 장벽 에너지 몫 (최대·평균).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FP = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
if os.path.exists(FP):
    fm.fontManager.addfont(FP)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False


def frames(p):
    by = {}
    for l in open(p):
        q = json.loads(l)
        by[round(q["t"], 9)] = q
    return [by[k]["steps"] for k in sorted(by)]


def band(sd):
    for d in (sd,):
        cm = os.path.join(d, "cmd.txt")
        if os.path.exists(cm):
            s = open(cm).read()
            if "--t2b " in s:
                return [float(x) for x in s.split("--t2b ")[1].split()[0].split(",")]
    return None


def run_stats(d):
    S = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    p = os.path.join(d, "t2.jsonl")
    F = frames(p) if os.path.exists(p) else []
    b = band(d)
    st = [r for f in F for r in f]
    e = S.get("err_mean") if S.get("err_mean") is not None else S.get("err_mean_done")
    o = dict(err=e, div=S.get("diverged"), nf=len(F), F=F, band=b)
    if st and b:
        slo, shi = b[0], b[1]
        o.update(ndet=max(r["ndet"] for r in st), steps_det=sum(r["ndet"] > 0 for r in st), dmin=min(r["dmin"] for r in st),
                 nband=max(r["nband"] for r in st), steps_band=sum(r["nband"] > 0 for r in st),
                 exlo=max(slo / r["smin"] - 1 if r["smin"] > 0 else float("inf") for r in st), exhi=max(r["smax"] / shi - 1 for r in st),
                 ncap=sum(r["ncap"] for r in st), nchk=sum(r["nchk"] for r in st), amin=min(r["amin"] for r in st), nstart=sum(r["nstart"] for r in st),
                 nnofeas=sum(r["nnofeas"] for r in st), ninf=max(r["ninf"] for r in st), nact=max(r["nact"] for r in st), nq=max(r["nq"] for r in st),
                 shmax=max(r["Eb"] / (r["Eb"] + r["Eh"]) if r["Eb"] + r["Eh"] > 0 else 0.0 for r in st),
                 shmean=float(np.mean([r["Eb"] / (r["Eb"] + r["Eh"]) if r["Eb"] + r["Eh"] > 0 else 0.0 for r in st])), nstep=len(st))
    pp = os.path.join(d, "period.txt")
    if os.path.exists(pp):
        o["period"] = open(pp).read().strip().splitlines()[-1][:120]
    return o


def fmt(x, f="%.3g"):
    if x is None:
        return "-"
    if isinstance(x, float) and not np.isfinite(x):
        return "∞"
    return f % x


def per_frame(F, key, red):
    return [red([r[key] for r in f]) if f else np.nan for f in F]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for arg in sys.argv[2:]:
        scene, rest = arg.split("=", 1)
        groups = []
        for g in rest.split(";"):
            nm, runs = g.split(":", 1)
            groups.append((nm, [r for r in runs.split(",") if r]))
        print(f"\n#### {scene}\n")
        print("| 판 | 발산 | 오차 % | det≤0 사면체 (걸음 최대) · 걸린 걸음 | 최소 det | 띠 밖 (최대) · 걸린 걸음 | 최대 넘침 아래 · 위 | 보폭 자른 반복 / 확인 | 최소 보폭 | 시작점 섞기 | 실현 가능 후보 없음 | 장벽 에너지 몫 최대 · 평균 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
        ax = ax.ravel()
        for gi, (nm, runs) in enumerate(groups):
            R = [run_stats(r) for r in runs if os.path.exists(r)]
            if not R:
                continue
            nd = sum(1 for q in R if q.get("div"))
            errs = [q["err"] for q in R if q.get("err") is not None]
            es = (f"{100 * np.mean(errs):.2f} [{100 * min(errs):.2f}~{100 * max(errs):.2f}]" if len(errs) > 1 else (f"{100 * errs[0]:.2f}" if errs else "-"))
            dv = f"{nd}/{len(R)}" + ((" (f" + "·".join(str(q["div"]) for q in R if q.get("div")) + ")") if nd else "")
            Q = [q for q in R if "ndet" in q]
            if Q:
                mx = lambda k: max(q[k] for q in Q)
                mn = lambda k: min(q[k] for q in Q)
                sm = lambda k: sum(q[k] for q in Q)
                print(f"| {nm} | {dv} | {es} | {mx('ndet')} · {sm('steps_det')}/{sm('nstep')} | {fmt(mn('dmin'))} | {mx('nband')} · {sm('steps_band')}/{sm('nstep')} | "
                      f"{fmt(mx('exlo'))} · {fmt(mx('exhi'))} | {sm('ncap')} / {sm('nchk')} | {fmt(mn('amin'))} | {sm('nstart')} | {sm('nnofeas')} | {fmt(mx('shmax'))} · {fmt(np.mean([q['shmean'] for q in Q]))} |")
            else:
                print(f"| {nm} | {dv} | {es} | - | - | - | - | - | - | - | - | - |")
            q = Q[0] if Q else None
            if q is None:
                continue
            F, b = q["F"], q["band"]
            x = np.arange(1, len(F) + 1)
            c = f"C{gi}"
            ax[0].plot(x, per_frame(F, "ndet", max), c, label=nm)
            ax[1].plot(x, per_frame(F, "dmin", min), c, label=nm)
            ax[2].plot(x, per_frame(F, "nband", max), c, label=nm)
            ax[3].plot(x, [max(b[0] / min(r["smin"] for r in f) if min(r["smin"] for r in f) > 0 else 10.0, max(r["smax"] for r in f) / b[1]) - 1 if f else np.nan for f in F], c, label=nm)
            ax[4].plot(x, per_frame(F, "ncap", sum), c, label=nm + " 자른 반복")
            ax[4].plot(x, per_frame(F, "amin", min), c + "--", label=nm + " 최소 보폭")
            ax[5].plot(x, [max(r["Eb"] / (r["Eb"] + r["Eh"]) if r["Eb"] + r["Eh"] > 0 else 0.0 for r in f) if f else np.nan for f in F], c, label=nm)
        T = ["det ≤ 0 사면체 수 (프레임 안 걸음 최대)", "최소 det F", "띠 [σ_lo, σ_hi] 밖 사면체 수", "최대 넘침 (σ_lo/σ_min − 1 또는 σ_max/σ_hi − 1)", "보폭 자른 반복 수 · 최소 보폭", "장벽 에너지 몫 E_b/(E_b+E_Hencky)"]
        for i, a in enumerate(ax):
            a.set_title(T[i], fontsize=10)
            a.set_xlabel("프레임")
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[3].axhline(0, color="k", lw=0.6)
        fig.suptitle(f"절 CM103 {scene}: 위반량·장벽 걸림 (첫 판, 걸음 끝 기록)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"tet2_viol_{scene}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
