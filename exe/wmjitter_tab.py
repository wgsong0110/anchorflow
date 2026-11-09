"""절 CM90 묶음 wmjitter 로컬 표·그림 (numpy·matplotlib, torch 없음; 읽기만).
사용: python3 exe/wmjitter_tab.py <내려받은 R2 lbours/wmjitter 디렉토리>  -> <디렉토리>/wmjitter_table.md, wmjitter_curves.png, wmjitter_psd.png, wmjitter_rows.json
판: jA_* (H9 그대로), jC_* (수렴 판), jT·jX·jN (한 요인), gt (정답 GF). 창 = f30~f100 (배열 fi 29~99).
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for p in glob.glob(os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

D = sys.argv[1]
F0, F1 = 29, 100                                                                      # fi 29..99 = f30..f100
NAME = {"jA": "H9 그대로", "jC": "수렴 판", "jT": "예산 끔", "jX": "예산 끔 + xit 문 끔", "jN": "예산 끔 + 뉴턴 20·PCG 40", "gt": "정답 (GF)"}
COL = {"jA": "#2a78d6", "jC": "#eb6834", "jT": "#1baf7a", "jX": "#eda100", "jN": "#6250d6", "gt": "#52514e"}
EPSN = 1e-2


def jn(n, x):
    s, c = np.sin(x), np.cos(x)
    if n == 0:
        return s / x
    if n == 1:
        return s / x ** 2 - c / x
    if n == 2:
        return (3 / x ** 2 - 1) * s / x - 3 * c / x ** 2
    if n == 3:
        return (15 / x ** 3 - 6 / x) * s / x - (15 / x ** 2 - 1) * c / x


def root(f, a, b, n=4000):
    x = np.linspace(a, b, n); y = f(x)
    i = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0][0]
    lo, hi = x[i], x[i + 1]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(f(lo)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def psd(Y, h):
    """Y [T, n] -> 직선 추세 뺀 뒤 Hann 창 rfft 파워 평균 (열 평균), 주파수 Hz."""
    T = Y.shape[0]
    ok = np.isfinite(Y).all(0)
    Y = Y[:, ok]
    t = np.arange(T) - (T - 1) / 2
    Y = Y - Y.mean(0) - np.outer(t, (t[:, None] * Y).sum(0) / (t * t).sum())
    w = np.hanning(T)
    P = (np.abs(np.fft.rfft(Y * w[:, None], axis=0)) ** 2).mean(1)
    return np.fft.rfftfreq(T, h), P, int(ok.sum())


def load_run(d):
    z = np.load(os.path.join(d, "wj.npz") if os.path.exists(os.path.join(d, "wj.npz")) else os.path.join(d, "gt.npz"))
    r = {k: z[k] for k in z.files}
    r["c"] = {k: i for i, k in enumerate(list(z["cols"]))}
    r["ac"] = {k: i for i, k in enumerate(list(z["acols"]))}
    s = os.path.join(d, "summary.json")
    r["sum"] = json.load(open(s)) if os.path.exists(s) else None
    return r


gi = json.load(open(os.path.join(D, "gt", "gtinfo.json")))
h = float(gi["h"])
runs = {}
for d in sorted(glob.glob(os.path.join(D, "j*"))) + [os.path.join(D, "gt")]:
    n = os.path.basename(d)
    if os.path.exists(os.path.join(d, "wj.npz")) or os.path.exists(os.path.join(d, "gt.npz")):
        runs[n] = load_run(d)

# ---- 물리 기준: 고유 주기
mu, lam, rho = gi["mu"], gi["lam"], gi["density"]
cs, cp = np.sqrt(mu / rho), np.sqrt((lam + 2 * mu) / rho)
Req = (3 * gi["vol_total"] / (4 * np.pi)) ** (1 / 3)
xt = root(lambda x: jn(2, x) - x * jn(3, x), 1.0, 4.0)                                  # 비틀림 n = 2 (자유 구)
xb = root(lambda x: (lam + 2 * mu) * x * jn(0, x) - 4 * mu * jn(1, x), 0.5, 6.0)      # 숨쉬기 n = 0
modes = []
for nm, x, c in (("비틀림 n=2", xt, cs), ("숨쉬기 n=0", xb, cp)):
    for lab, R in (("구 등가 반지름", Req), ("f0 가장 짧은 반폭", 0.5 * min(gi["ext_f0"]))):
        f = x * c / (2 * np.pi * R)
        fa = abs(f - round(f * h) / h)
        modes.append(dict(mode=nm, R=lab, Rm=R, f=f, T_ms=1e3 / f, alias=fa, wdt=2 * np.pi * f * h))
fny = 0.5 / h


def win(a):
    return a[F0:min(F1, a.shape[0])]


rows = []
for n, r in runs.items():
    T, c, A, ac = r["T"], r["c"], r["A"], r["ac"]
    if T.shape[0] <= F0 + 5:
        continue
    g = lambda k: win(T[:, c[k]])
    row = dict(run=n, kind=n.split("_")[0], frames=int(T.shape[0]))
    if r["sum"]:
        row.update(div=r["sum"].get("diverged"), err=100 * (r["sum"].get("err_mean") or r["sum"].get("err_mean_done") or np.nan))
    row["KEdef"] = float(np.nanmean(g("KEdef"))); row["KE"] = float(np.nanmean(g("KE"))); row["KEr"] = float(np.nanmean(g("KEr")))
    row["vdef"] = 1e3 * float(np.nanmean(g("vdef_rms")))
    row["vcom"] = 1e3 * float(np.nanmean(np.sqrt(g("Px") ** 2 + g("Py") ** 2 + g("Pz") ** 2) / g("M")))
    row["wabs"] = float(np.nanmean(np.sqrt(g("wx") ** 2 + g("wy") ** 2 + g("wz") ** 2)))
    row["Eel"] = float(np.nanmean(g("Eel"))); row["Eel_sd"] = float(np.nanstd(np.diff(g("Eel"))))
    row["arel"] = 1e3 * float(np.nanmean(win(A[:, ac["arms_rel"]])))
    row["a99"] = 1e3 * float(np.nanmean(win(A[:, ac["a99"]])))
    row["dJp"] = float(np.nanmean(g("dJrms"))); row["dJp05"] = float(np.nanmean(g("dJn05") / g("nfin")))
    if not np.all(np.isnan(T[:, c["GdJrms"]])):
        row["dJq"] = float(np.nanmean(g("GdJrms"))); row["dJq05"] = float(np.nanmean(g("GdJn05") / g("Gmatch"))); row["dJq20"] = float(np.nanmean(g("GdJn20") / g("Gmatch")))
        row["Gle0"] = float(np.nanmean(g("Gle0")))
    # 변형 변위 스펙트럼 (질량중심 뺀 부분 표본 위치)
    XS = win(r["XS"]); COM = win(r["COM"])
    U = (XS - COM[:, None, :]).reshape(XS.shape[0], -1)
    fq, P, _ = psd(U, h)
    hi = fq > 0.5 * fny
    row["psd_hi"] = float(P[hi].sum() / P[1:].sum()); row["fpk"] = float(fq[1:][np.argmax(P[1:])])
    r["psd"] = (fq, P)
    if "GS" in r and r["GS"].shape[1]:
        fq2, P2, ng = psd(win(r["GS"]), h); r["psdG"] = (fq2, P2); row["psdG_hi"] = float(P2[fq2 > 0.5 * fny].sum() / P2[1:].sum()); row["fpkG"] = float(fq2[1:][np.argmax(P2[1:])]); row["nGs"] = ng
    fq3, P3, _ = psd(win(r["JS"]), h); r["psdJ"] = (fq3, P3); row["psdJ_hi"] = float(P3[fq3 > 0.5 * fny].sum() / P3[1:].sum())
    if "UF" in r:
        K = r["K"]; UF = r["UF"]; TB = r["TB"]
        fl, gF, nst, cut = [], [], 0, 0
        frame_uf = np.zeros(T.shape[0], bool)
        for fi in range(F0, min(F1, T.shape[0])):
            k = max(int(K[fi]), 1)
            fl += list(UF[fi, 0, :k]); gF += list(np.sqrt(np.maximum(UF[fi, 1, :k], 0)))
            frame_uf[fi] = np.nansum(UF[fi, 0, :k]) > 0
        fl, gF = np.array(fl, float), np.array(gF, float)
        row.update(nstep=int(fl.size), uf=float(np.nanmean(fl > 0)), gF_med=float(np.nanmedian(gF)), gF_max=float(np.nanmax(gF)), conv=float(np.nanmean(gF <= EPSN)),
                   cutf=int((win(TB[:, 0]) > 0).sum()), Kgt1=int((win(K) > 1).sum()))
        r["frame_uf"] = frame_uf
        # 걸음 덜 풀림 프레임과 가우스 점 흔들림
        dq = T[:, c["GdJrms"]]
        sel = np.arange(T.shape[0]); sel = sel[(sel >= F0) & (sel < F1)]
        cutm = TB[:, 0] > 0
        for tag, m_ in (("uf", frame_uf), ("cut", cutm)):
            a_, b_ = dq[sel][m_[sel]], dq[sel][~m_[sel]]
            row[f"dJq_{tag}1"] = float(np.nanmean(a_)) if a_.size else None; row[f"dJq_{tag}0"] = float(np.nanmean(b_)) if b_.size else None
            row[f"n_{tag}1"] = int(a_.size)
        gfm = np.array([np.nanmax(np.sqrt(np.maximum(UF[fi, 1, :max(int(K[fi]), 1)], 0))) for fi in sel])
        ok = np.isfinite(gfm) & np.isfinite(dq[sel])
        row["corr_gF_dJq"] = float(np.corrcoef(np.log(gfm[ok]), np.log(dq[sel][ok]))[0, 1]) if ok.sum() > 5 else None
    rows.append(row)
json.dump(dict(rows=rows, modes=modes, h=h, fny=fny, cs=cs, cp=cp, Req=Req, gi=gi), open(os.path.join(D, "wmjitter_rows.json"), "w"), ensure_ascii=False, indent=1, default=float)


def fmt(v, f="{:.3g}"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


L = []
L.append("# 절 CM90 wmjitter -- 수박 H9 바닥 뒤 꿈틀거림 (창 f30~f100 평균; 단위 m·s, 정답 GF 와 같은 단위)\n")
L.append("## (1) 흔들림 지표\n")
L.append("| 판 | 발산 · 오차 % | 강체 뺀 운동 에너지 | 강체 뺀 속도 RMS mm/s | 질량중심 속력 mm/s | 각속도 rad/s | 탄성 에너지 · 프레임 간 변화 표준편차 | 가속도 RMS (질량중심 뺀) · 99 % mm/프레임² | 입자 ΔJ RMS · >0.05 몫 | 가우스 점 ΔJ RMS · >0.05 몫 · >0.2 몫 | 변형 변위 스펙트럼 위 절반 몫 · 봉우리 Hz | 가우스 점 J 스펙트럼 위 절반 몫 · 봉우리 Hz |")
L.append("|" + "---|" * 12)
for r in rows:
    L.append(f"| {r['run']} ({NAME.get(r['kind'], r['kind'])}) | {fmt(r.get('div'), '{}')} · {fmt(r.get('err'), '{:.2f}')} | {fmt(r['KEdef'])} | {fmt(r['vdef'])} | {fmt(r['vcom'])} | {fmt(r['wabs'])} | {fmt(r['Eel'])} · {fmt(r['Eel_sd'])} | "
             f"{fmt(r['arel'])} · {fmt(r['a99'])} | {fmt(r['dJp'])} · {fmt(r['dJp05'])} | {fmt(r.get('dJq'))} · {fmt(r.get('dJq05'))} · {fmt(r.get('dJq20'))} | {fmt(r['psd_hi'], '{:.2f}')} · {fmt(r['fpk'], '{:.1f}')} | {fmt(r.get('psdG_hi'), '{:.2f}')} · {fmt(r.get('fpkG'), '{:.1f}')} |")
L.append("\n## (2) 걸음 수렴 (창 f30~f100 의 모든 걸음; ‖g‖/F = 걸음 끝 잔차 비, εN = 0.01)\n")
L.append("| 판 | 걸음 수 | 덜 풀림 판정 몫 | ‖g‖/F 중앙 · 최대 | εN 닿은 몫 | 예산 자름 프레임 | K > 1 프레임 | 가우스 점 ΔJ RMS: 덜 풀린 걸음 있는 프레임 (수) · 없는 프레임 | 예산 자른 프레임 (수) · 아닌 프레임 | log ‖g‖/F 최대 대 log ΔJ_q 상관 |")
L.append("|" + "---|" * 10)
for r in rows:
    if "nstep" not in r:
        continue
    L.append(f"| {r['run']} | {r['nstep']} | {fmt(r['uf'], '{:.2f}')} | {fmt(r['gF_med'])} · {fmt(r['gF_max'])} | {fmt(r['conv'], '{:.2f}')} | {r['cutf']} | {r['Kgt1']} | {fmt(r['dJq_uf1'])} ({r['n_uf1']}) · {fmt(r['dJq_uf0'])} | "
             f"{fmt(r['dJq_cut1'])} ({r['n_cut1']}) · {fmt(r['dJq_cut0'])} | {fmt(r['corr_gF_dJq'], '{:.2f}')} |")
L.append(f"\n## (3) 물리 기준 -- 수박 재질 탄성 고유 주기 (E {gi['E']:g}, ν {gi['nu']:g}, ρ {gi['density']:g}; c_s {cs:.2f}, c_p {cp:.2f} m/s; 구 등가 반지름 {Req:.3f} m, f0 크기 {' x '.join('%.3f' % q for q in gi['ext_f0'])} m; 프레임 h {h:g} s, 나이퀴스트 {fny:.1f} Hz)\n")
L.append("| 모드 | 길이 | 진동수 Hz | 주기 ms | ω h | 프레임 표본에서 보이는 (겹침) 진동수 Hz |")
L.append("|---|---|---|---|---|---|")
for m in modes:
    L.append(f"| {m['mode']} | {m['R']} {m['Rm']:.3f} m | {m['f']:.1f} | {m['T_ms']:.1f} | {m['wdt']:.1f} | {m['alias']:.1f} |")
open(os.path.join(D, "wmjitter_table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- 그림 1: 시간 곡선
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
fr = lambda T: np.arange(1, T.shape[0] + 1)
spec = [("강체 뺀 운동 에너지", lambda r: r["T"][:, r["c"]["KEdef"]], True), ("가우스 점 J 프레임 간 변화 RMS (정답은 입자 J)", lambda r: r["T"][:, r["c"]["GdJrms"]] if not np.all(np.isnan(r["T"][:, r["c"]["GdJrms"]])) else r["T"][:, r["c"]["dJrms"]], True),
        ("입자 가속도 RMS, 질량중심 뺀 것 (mm/프레임²)", lambda r: 1e3 * r["A"][:, r["ac"]["arms_rel"]], True), ("걸음 끝 잔차 비 ‖g‖/F (프레임 최대)", None, True)]
for a_, (tt, fn, lg) in zip(ax.flat, spec):
    for n, r in runs.items():
        k = n.split("_")[0]; ls = "-" if not n.endswith("_2") else "--"
        if fn is None:
            if "UF" not in r:
                continue
            y = np.array([np.nanmax(np.sqrt(np.maximum(r["UF"][fi, 1, :max(int(r["K"][fi]), 1)], 0))) for fi in range(r["T"].shape[0])])
        else:
            y = fn(r)
        a_.plot(fr(r["T"]), y, ls, color=COL.get(k, "#888"), lw=1.6, label=f"{n} {NAME.get(k, '')}")
    if fn is None:
        a_.axhline(EPSN, color="#52514e", lw=1, ls=":"); a_.text(2, EPSN * 1.15, "εN 0.01", color="#52514e", fontsize=9)
    a_.axvspan(30, 100, color="#f0efec", zorder=0)
    a_.set_title(tt, fontsize=11); a_.set_xlabel("프레임"); a_.grid(alpha=0.25, lw=0.6)
    if lg:
        a_.set_yscale("log")
ax[0, 0].set_ylim(1e-7, 3); ax[0, 0].legend(fontsize=8, ncol=2, frameon=False, loc="lower right")
fig.suptitle("수박 H9 바닥 뒤 흔들림: H9 그대로 · 수렴 판 · 한 요인 판 · 정답 (회색 띠 = 창 f30~f100)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(D, "wmjitter_curves.png"), dpi=130); plt.close(fig)

# ---- 그림 2: 스펙트럼
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for n, r in runs.items():
    k = n.split("_")[0]; ls = "-" if not n.endswith("_2") else "--"
    if "psd" in r:
        fq, P = r["psd"]; ax[0].semilogy(fq[1:], P[1:], ls, color=COL.get(k, "#888"), lw=1.6, label=f"{n} {NAME.get(k, '')}")
    key = "psdG" if "psdG" in r else "psdJ"
    if key in r:
        fq, P = r[key]; ax[1].semilogy(fq[1:], P[1:], ls, color=COL.get(k, "#888"), lw=1.6, label=f"{n} {'가우스 점' if key == 'psdG' else '입자'} J")
for a_ in ax:
    for m in modes:
        if m["R"].startswith("구"):
            a_.axvline(m["alias"], color="#52514e", lw=1, ls=":")
            a_.text(m["alias"], a_.get_ylim()[1] if False else 1, "", fontsize=8)
    a_.set_xlabel("진동수 Hz (프레임 표본, 나이퀴스트 %.1f)" % fny); a_.grid(alpha=0.25, lw=0.6)
ax[0].set_title("질량중심 뺀 입자 변위 스펙트럼 (f30~f100, 직선 추세 뺌)", fontsize=11)
ax[1].set_title("J 시계열 스펙트럼 (f30~f100)", fontsize=11)
ax[0].legend(fontsize=8, frameon=False); ax[1].legend(fontsize=7, frameon=False, ncol=2, loc="center right")
fig.text(0.5, 0.005, "점선 = 고유 모드 (비틀림 n=2, 숨쉬기 n=0, 구 등가 반지름) 가 프레임 표본에서 겹쳐 보이는 진동수", ha="center", fontsize=9, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(D, "wmjitter_psd.png"), dpi=130); plt.close(fig)
