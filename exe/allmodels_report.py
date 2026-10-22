"""allmodels 결과 표.

  err  : 모델 × 궤적 위치 오차 표 (manifest.tsv) -> md
  fail : 실패 양상 판정 (allmodels_failmode.py json 들) -> md + png

판정 (프레임 t ≥ max(c,1), c = 충돌 프레임):
  (A) 멈춤      : 이동량 비 < 0.2 이고 무게중심 이동 비 < 0.2 인 프레임이 10 개 연속, 그리고 뒤집힘(det F ≤ 0) 0
  (B) 늘어남    : max|σ−1| 99% > 0.5 또는 σ1/σ3 99% > 3 인 프레임이 있고, 뒤집힘 0, (A) 아님
  (C) 망가짐    : det F ≤ 0 입자 > 0.1% 또는 ψ 평균 > 정답 ψ 평균 × 1e3 또는 비유한 값
  (D) 정상 범위 : 해당 없음
경계 표시 "≈" : 판정을 정한 값과 문턱의 비가 [0.8, 1.25] 안 (문턱 근처라 문턱을 조금 바꾸면 판정이 바뀔 수 있음).
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["err", "fail"])
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--commit", default="")
ap.add_argument("--gtdir0000", default="")
ap.add_argument("--cmdcheck", default="")
ap.add_argument("--errcmp", default="")
ap.add_argument("--fm", nargs="*", default=[])
ap.add_argument("--png", default="")
ap.add_argument("--font", default="")
a = ap.parse_args()

M = []
for ln in open(a.manifest, encoding="utf-8"):
    f = ln.rstrip("\n").split("\t")
    M.append(dict(name=f[0], lab=f[1], T=f[2], dump=f[3], log=f[4], err=f[5], it=f[6], it_ck=f[7], commit=f[8], ck=f[9], src=f[10]))
NAMES = list(dict.fromkeys(m["name"] for m in M))
LAB = {m["name"]: m["lab"] for m in M}
TRJ = list(dict.fromkeys(m["T"] for m in M))
G = {(m["name"], m["T"]): m for m in M}

if a.mode == "err":
    o = ["# lego 학습 모델 롤아웃 위치 오차 (궤적 × 모델)", "",
         f"- 생성: exe/allmodels.sh (커밋 {a.commit}); 학습 없음 (--iters 0), 80 프레임, 정답 처음 상태에서 시작, 오차 = 입자 평균 위치 차 / EXT (train_deform 롤아웃 `평균`).",
         "- 각 모델은 자기 학습 커밋 워크트리에서 자기 원래 롤아웃 명령 그대로 (BASE 를 원래 스크립트 원문에서 다시 만들어 기록된 명령줄과 글자 대조 -- 아래 `명령줄 대조`). "
         "바꾼 것은 AF_ROLL_DUMP_F=32(F float32 저장, 계산 같음)와 출력 경로뿐.",
         f"- 0000 정답 디렉토리: {a.gtdir0000} (덤프 처음 상태·첫 프레임과 일치하는 쪽, gt_check_0000.log). L000 = lbdata/test_low/L000, 0004 = lbdata/test/0004.",
         "- L000·0000 은 다섯 궤적 모델의 학습 궤적(학습 상태 = 출력만 최적화 덤프)이고, 0004 는 학습에 쓰지 않은 궤적이다.", "",
         "| # | 모델 | 커밋 | 반복 | " + " | ".join(f"{T} 오차" for T in TRJ) + " |", "|---|---|---|---|" + "---|" * len(TRJ)]
    for i, n in enumerate(NAMES):
        m0 = G[(n, TRJ[0])]
        it = m0["it"] if m0["it"] == m0["it_ck"] else f"{m0['it']} (사본 {m0['it_ck']})"
        cells = []
        for T in TRJ:
            m = G[(n, T)]
            cells.append(f"{m['err']}%" + (" (재사용)" if m["src"] == "재사용" else ""))
        o.append(f"| {i + 2} | {LAB[n]} (`{n}`) | {m0['commit']} | {it} | " + " | ".join(cells) + " |")
    o += ["", "## 체크포인트", ""]
    for n in NAMES:
        m0 = G[(n, TRJ[0])]
        o.append(f"- `{n}`: {m0['ck']} (반복 {m0['it']})")
    o += ["", "## 롤아웃 덤프 (F float32)", ""]
    for m in M:
        o.append(f"- `{m['name']}` {m['T']}: {m['dump']} ({m['src']}; 로그 {m['log']})")
    if a.errcmp:
        o += ["", "## 기존 오차와 대조", "", "```", open(a.errcmp, encoding="utf-8").read().strip(), "```"]
    if a.cmdcheck:
        o += ["", "## 명령줄 대조", "", "```", open(a.cmdcheck, encoding="utf-8").read().strip(), "```"]
    open(a.out, "w", encoding="utf-8").write("\n".join(o) + "\n")
    print("[저장]", a.out)
    sys.exit(0)

# ---------------- fail ----------------
TH = dict(stop=0.2, run=10, smax=0.5, cond=3.0, flip=1e-3, psi=1e3, gmin=1e-4)


def near(v, th):
    return th / 1.25 <= v <= th * 1.25


def judge(r, gt, c, N, EXT):
    L = len(r["mv"]); s = max(c, 1)
    fr = list(range(s, L + 1))                      # 프레임 번호
    ix = [t - 1 for t in fr]
    mvr = np.array([r["mv"][i] / max(gt["mv"][i], 1e-12) for i in ix]); cmr = np.array([r["com"][i] / max(gt["com"][i], 1e-12) for i in ix])
    # 정답이 거의 안 움직이는 프레임(프레임당 < GMIN·EXT)은 비가 0/0 이라 멈춤 판정에서 뺀다 (창이 끊긴다);
    # 무게중심만 거의 안 움직이면 그 프레임은 입자 평균 이동량 비로만 본다
    vmv = np.array([gt["mv"][i] >= TH["gmin"] * EXT for i in ix]); vcm = np.array([gt["com"][i] >= TH["gmin"] * EXT for i in ix])
    both = np.where(vmv, np.maximum(mvr, np.where(vcm, cmr, 0.0)), np.inf)
    flips = max(r["ndet"][i] for i in ix)
    # A: 길이 10 창 중 max(비) 가 가장 작은 것
    wins = [both[k:k + TH["run"]].max() for k in range(len(both) - TH["run"] + 1)]
    mA = float(min(wins)) if wins else float("inf")
    n_gstill = int((~vmv).sum())
    kA = next((k for k, w in enumerate(wins) if w < TH["stop"]), None)
    condA = kA is not None
    # B
    bv = np.array([max(r["smax99"][i] / TH["smax"], r["cond99"][i] / TH["cond"]) for i in ix])
    mB = float(bv.max()); kB = next((k for k, v in enumerate(bv) if v > 1), None); condB = kB is not None
    # C
    cv = np.array([max(r["ndet"][i] / N / TH["flip"], r["psi"][i] / max(gt["psi"][i], 1e-12) / TH["psi"]) if np.isfinite(r["psi"][i]) else np.inf for i in ix])
    nf = [r["nonfin"][i] for i in ix]
    kC = next((k for k, (v, n_) in enumerate(zip(cv, nf)) if v > 1 or n_ > 0), None); condC = kC is not None
    mC = float(np.nanmax(np.where(np.isfinite(cv), cv, np.nan))) if np.isfinite(cv).any() else float("inf")
    A = condA and flips == 0; B = condB and flips == 0 and not A; Cc = condC
    v = dict(A=A, B=B, C=Cc, D=not (A or B or Cc), flips_max=flips, nonfin=int(sum(nf)),
             A_start=fr[kA] if condA else None, B_start=fr[kB] if condB else None, C_start=fr[kC] if condC else None,
             mA=mA, mB=mB, mC=mC, nearA=near(mA, TH["stop"]), nearB=near(mB, 1.0), nearC=near(mC, 1.0),
             condA=condA, condB=condB, condC=condC,
             n_gstill=n_gstill, mv_ratio_mean=float(np.mean(mvr[vmv])) if vmv.any() else float("nan"), com_ratio_mean=float(np.mean(cmr[vcm])) if vcm.any() else float("nan"),
             smax99_max=float(max(r["smax99"][i] for i in ix)), cond99_max=float(max(r["cond99"][i] for i in ix)),
             psi_ratio_max=float(max(r["psi"][i] / max(gt["psi"][i], 1e-12) for i in ix)),
             err_mean=100 * float(np.mean(r["err"])), err_last=100 * float(r["err"][-1]))
    # 프레임별 판정 구간 (그림)
    span = dict(A=[], B=[], C=[])
    if condA:
        for k, w in enumerate(wins):
            if w < TH["stop"]:
                span["A"] += list(range(fr[k], fr[k] + TH["run"]))
    span["B"] = [fr[k] for k, x in enumerate(bv) if x > 1]
    span["C"] = [fr[k] for k, (x, n_) in enumerate(zip(cv, nf)) if x > 1 or n_ > 0]
    v["span"] = {k: sorted(set(x)) for k, x in span.items()}
    return v


FM = {}
for p in a.fm:
    d = json.load(open(p)); FM[d["traj"]] = d
V = {}
for T, d in FM.items():
    for n, r in d["runs"].items():
        V[(n, T)] = judge(r, d["gt"], d["c"], d["N"], d["EXT"])
# 정답 자체 참고값
GTR = {}
for T, d in FM.items():
    g = d["gt"]; s = max(d["c"], 1)
    GTR[T] = dict(smax99=max(g["smax99"][s - 1:]), cond99=max(g["cond99"][s - 1:]), ndet=max(g["ndet"][s - 1:]), mv=float(np.mean(g["mv"][s - 1:])))


def lab(v):
    out = []
    for k, nm in (("A", "멈춤"), ("B", "늘어남"), ("C", "망가짐")):
        if v[k]:
            out.append(f"({k}) {nm} @{v[k + '_start']}" + (" ≈" if v["near" + k] else ""))
    if v["D"]:
        out.append("(D) 정상 범위")
        for k, nm in (("A", "멈춤"), ("B", "늘어남"), ("C", "망가짐")):
            if v["near" + k]:
                out.append(f"≈{nm} 문턱 근처")
    if not v["A"] and v["condA"] and v["flips_max"] > 0:
        out.append(f"[멈춤 조건 충족 @{v['A_start']}, 뒤집힘 {v['flips_max']} 이라 A 아님]")
    if not v["B"] and v["condB"] and not v["A"] and v["flips_max"] > 0:
        out.append(f"[늘어남 조건 충족 @{v['B_start']}, 뒤집힘 {v['flips_max']} 이라 B 아님]")
    return "; ".join(out)


o = ["# 바닥 충돌 이후 실패 양상 판정 (lego 학습 모델 × 궤적)", "",
     f"- 생성: exe/allmodels_failmode.py + exe/allmodels_report.py (커밋 {a.commit}); 덤프는 allmodels_err.md 와 같은 롤아웃 (F float32).",
     "- 충돌 프레임 c = 정답에서 입자가 바닥 법선 거리 0.01 안에 처음 들어온 프레임 (L000 은 처음부터 닿아 있어 c=0). 판정은 프레임 t ≥ max(c,1).",
     f"- 문턱: (A) 멈춤 = 입자 평균 이동량 비 < {TH['stop']} **그리고** 무게중심 이동 비 < {TH['stop']} 가 {TH['run']} 프레임 연속, 뒤집힘(det F≤0) 0; "
     f"(B) 비물리적 늘어남 = max|σ−1| 99% > {TH['smax']} 또는 σ1/σ3 99% > {TH['cond']:g}, 뒤집힘 0, 멈춤 아님; "
     f"(C) 망가짐 = det F≤0 입자 > {100 * TH['flip']:g}% 또는 ψ 평균 > 정답 × {TH['psi']:g} 또는 비유한 값; (D) 해당 없음. 여러 개면 모두 표시, @ = 처음 나타난 프레임.",
     "- ≈ = 판정을 정한 값이 문턱의 0.8~1.25 배 (경계 근처). 이동량 비 = 모델/정답 (같은 프레임), 무게중심 = 입자 평균 위치.",
     f"- 정답이 거의 안 움직이는 프레임(프레임당 이동량 < {TH['gmin']:g}·EXT)은 비가 0/0 이라 멈춤 판정에서 뺀다 (그 프레임에서 10 연속 창이 끊김; 무게중심만 그러면 입자 평균 비로만 본다). 표의 '정답 정지 프레임' 수.",
     "- σ·det·ψ 는 덤프의 누적 F (렌더에 쓰는 것) 로, ψ 는 phys_resid.psi_of (jelly FCR). 정답 ψ 는 lbdata h5 의 float32 F 로 같은 식.", ""]
for T in TRJ:
    if T not in FM:
        continue
    d = FM[T]; g = GTR[T]
    o += [f"## 궤적 {T} (c = {d['c']}, 계산값 {d['c_calc']}; 입자 {d['N']}; 정답 참고: max|σ−1| 99% 최대 {g['smax99']:.3f}, σ1/σ3 99% 최대 {g['cond99']:.2f}, "
          f"det≤0 최대 {g['ndet']}, 프레임당 평균 이동량 {g['mv']:.2e})", "",
          "| 모델 | 판정 | 오차 평균 | 정답 정지 프레임 | 이동량 비 평균 | 무게중심 비 평균 | 10프레임 창 최소 max(비) | max\\|σ−1\\| 99% 최대 | σ1/σ3 99% 최대 | det≤0 최대 (비율) | ψ/정답 최대 | 비유한 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in NAMES:
        v = V.get((n, T))
        if v is None:
            continue
        o.append(f"| {LAB[n]} | {lab(v)} | {v['err_mean']:.2f}% | {v['n_gstill']} | {v['mv_ratio_mean']:.3f} | {v['com_ratio_mean']:.3f} | {v['mA']:.3f} | "
                 f"{v['smax99_max']:.3f} | {v['cond99_max']:.2f} | {v['flips_max']} ({100 * v['flips_max'] / d['N']:.3f}%) | {v['psi_ratio_max']:.2e} | {v['nonfin']} |")
    o.append("")
o += ["## 오차 대조 (지표 스크립트 재계산 vs 롤아웃 로그)", ""]
for T in TRJ:
    for n in NAMES:
        v = V.get((n, T))
        if v is not None:
            o.append(f"- {n} {T}: 재계산 {v['err_mean']:.3f}% / 로그 {G[(n, T)]['err']}%  (정답 동일성 최대차 {FM[T]['runs'][n]['gt_maxdiff']:.1e})")
open(a.out, "w", encoding="utf-8").write("\n".join(o) + "\n")
json.dump({f"{n}|{T}": v for (n, T), v in V.items()}, open(a.out.replace(".md", ".json"), "w"), ensure_ascii=False, indent=1)
print("[저장]", a.out)

if a.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if a.font:
        font_manager.fontManager.addfont(a.font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    TT = [T for T in TRJ if T in FM]
    MET = [("mv", "이동량 비 (모델/정답)"), ("smax99", "max|σ−1| 99%"), ("ndet", "det F≤0 입자 수")]
    COL = dict(A="#4C78A8", B="#F58518", C="#E45756")
    nr, nc = len(NAMES), len(TT) * len(MET)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 1.9 * nr + 1.2), squeeze=False, sharex=True)
    for i, n in enumerate(NAMES):
        for j, T in enumerate(TT):
            d = FM[T]; r = d["runs"].get(n); v = V.get((n, T))
            if r is None:
                continue
            x = np.arange(1, d["L"] + 1)
            for k, (key, ttl) in enumerate(MET):
                ax = axs[i, j * len(MET) + k]
                for kk in ("A", "B", "C"):
                    if v[kk] or (kk == "B" and v["condB"]) or (kk == "A" and v["condA"]):
                        for t in v["span"][kk]:
                            ax.axvspan(t - 0.5, t + 0.5, color=COL[kk], alpha=0.18 if v[kk] else 0.07, lw=0)
                if key == "mv":
                    gm, gc = np.array(d["gt"]["mv"]), np.array(d["gt"]["com"])
                    y = np.where(gm >= TH["gmin"] * d["EXT"], np.array(r["mv"]) / np.maximum(gm, 1e-12), np.nan)    # 정답 정지 프레임은 비우기
                    yc = np.where(gc >= TH["gmin"] * d["EXT"], np.array(r["com"]) / np.maximum(gc, 1e-12), np.nan)
                    ax.plot(x, np.clip(y, 1e-3, 1e3), color="k", lw=1.0, label="입자 평균")
                    ax.plot(x, np.clip(yc, 1e-3, 1e3), color="0.5", lw=0.8, ls="--", label="무게중심")
                    ax.axhline(TH["stop"], color=COL["A"], lw=0.8, ls=":"); ax.set_yscale("log")
                elif key == "smax99":
                    ax.plot(x, r["smax99"], color="k", lw=1.0); ax.plot(x, d["gt"]["smax99"], color="0.6", lw=0.8, ls="--")
                    ax.axhline(TH["smax"], color=COL["B"], lw=0.8, ls=":"); ax.set_yscale("log")
                else:
                    ax.plot(x, r["ndet"], color="k", lw=1.0); ax.axhline(TH["flip"] * d["N"], color=COL["C"], lw=0.8, ls=":")
                    ax.set_yscale("symlog", linthresh=1)
                ax.axvline(d["c"], color="0.3", lw=0.7)
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_title(f"{T} · {ttl}", fontsize=9)
                if k == 0:
                    ax.set_ylabel(LAB[n], fontsize=7.5)
                if k == 1:
                    ax.text(0.02, 0.92, lab(v)[:40], transform=ax.transAxes, fontsize=6.5, va="top")
    for ax in axs[-1]:
        ax.set_xlabel("프레임", fontsize=8)
    fig.suptitle("바닥 충돌 이후 실패 양상 -- 색 구간: 파랑 (A) 멈춤, 주황 (B) 늘어남, 빨강 (C) 망가짐 (옅은 색 = 조건만 충족, 판정 아님); "
                 "세로선 = 충돌 프레임 c, 점선 = 문턱, 회색 파선 = 정답", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(a.png, dpi=70)
    print("[저장]", a.png)
