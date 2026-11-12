"""절 CM113 표·그림 (로컬, 작은 json 만): exe/startpt_an.py 의 startpt_<판>.json -> startpt_tables.md · startpt.png.
사용: python exe/startpt_tab.py <json 디렉토리> <출력 디렉토리> [H9판 H5판]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jd, od = sys.argv[1:3]
R9, R5 = (sys.argv[3:5] if len(sys.argv) >= 5 else ("p9_1", "p5_1"))
for f in fm.findSystemFonts():
    if any(k in os.path.basename(f).lower() for k in ("nanum", "notosanscjk", "malgun")):
        fm.fontManager.addfont(f)
for nm in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(x.name == nm for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
D = {k: json.load(open(os.path.join(jd, f"startpt_{k}.json"))) for k in (R9, R5) if os.path.exists(os.path.join(jd, f"startpt_{k}.json"))}
LAB = {R9: "H9 wolf", R5: "H5 wolf (대조)"}
LL = ["0", "1", "2", "3", "4", "5+"]
KN = {"U_이전끝": "이전 걸음 끝 U", "Ut_관성예측": "관성 예측 Ũ", "y0_시작점": "뉴턴 시작점 y0", "yh_기준": "기준 ŷ (고착·구속)", "y_갱신": "뉴턴 갱신"}


def lab(e):
    if e["kind"] == "y":
        return f"y ← {e['kernel']}"
    if e["kind"] == "y_갱신":
        return f"뉴턴 갱신 {e['it'] + 1}"
    return KN.get(e["kind"], e["kind"])


def g(v):
    return "-" if v is None else (f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}")


md = []
for k, d in D.items():
    md += [f"## {LAB[k]} ({k}) -- 걸음 시작 단계별 칸 r (전체 칸; r<0.98 몫 · 중앙 · 10 백분위 · 최소; 줄어든 칸 = Δr < −0.005, Δr_C = 구속 노드 몫, Δr_F = 자유 노드 몫)", ""]
    for f, FO in sorted(d["frames"].items(), key=lambda q: int(q[0])):
        for S in FO["steps"]:
            md += [f"### f{f} 걸음 {S['k']} (구속 노드 {S['ncons']}/{S['nact']})", "",
                   "| 단계 | r<0.98 몫 | r<0.9 몫 | 중앙 | 10 백분위 | 최소 | 줄어든 칸 | 새로 r<0.98 | Δr 중앙 | Δr_C 중앙 | Δr_F 중앙 | 줄어든 칸 층 0/1/2/3/4/5+ |", "|" + "---|" * 12]
            for e in S["seq"]:
                a = e["r"].get("전체", {}); dc = e.get("dec") or {}
                md.append(f"| {lab(e)} | {a.get('f098', 0):.4f} | {a.get('f09', 0):.4f} | {g(a.get('med'))} | {g(a.get('p10'))} | {g(a.get('min'))} | {dc.get('n', '-')} | {dc.get('n_new098', '-')} | {g(dc.get('dr_med'))} | {g(dc.get('drC_med'))} | {g(dc.get('drF_med'))} | "
                          + ("/".join(str(dc['lay'][q]) for q in LL) if dc else "-") + " |")
            a = FO["rEnd"].get("전체", {})
            md.append(f"| 프레임 끝 | {a.get('f098', 0):.4f} | {a.get('f09', 0):.4f} | {g(a.get('med'))} | {g(a.get('p10'))} | {g(a.get('min'))} | | | | | | |")
            if S.get("ev"):
                md += ["", "평가별 항 중앙값 (N, 양수 = 누름) [누름 %]:", ""]
                for e, o in S["ev"].items():
                    for nm, v in o.items():
                        md.append(f"- 평가 {e} · {nm} ({v['n']} 칸): " + ", ".join(f"{q} {v[q]['med']:.3g} [{100 * v[q]['pos']:.0f}]" for q in v if q != "n"))
            md.append("")
open(os.path.join(od, "startpt_tables.md"), "w").write("\n".join(md) + "\n")

# 그림: 판마다, 프레임 (걸음 0) 별로 단계 차례의 r<0.98 몫 (위) · 최소 r (아래)
fig, ax = plt.subplots(2, len(D), figsize=(9 * len(D), 9), squeeze=False)
for c, (k, d) in enumerate(D.items()):
    frs = sorted(d["frames"], key=int)
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(frs)))
    for fi, f in enumerate(frs):
        S = d["frames"][f]["steps"][0]
        seq = [e for e in S["seq"] if not e["kind"].startswith("yh")]
        x = np.arange(len(seq))
        y1 = [e["r"].get("전체", {}).get("f098", np.nan) for e in seq]; y2 = [e["r"].get("전체", {}).get("min", np.nan) for e in seq]
        ax[0][c].plot(x, y1, "-o", ms=3, lw=1.5, color=cm[fi], label=f"f{f}")
        ax[1][c].plot(x, y2, "-o", ms=3, lw=1.5, color=cm[fi], label=f"f{f}")
        if fi == len(frs) - 1:
            for a in (ax[0][c], ax[1][c]):
                a.set_xticks(x); a.set_xticklabels([lab(e) for e in seq], rotation=70, ha="right", fontsize=7)
    ax[0][c].set_title(f"{LAB[k]} · 걸음 0 단계별 눌린 칸 (r<0.98) 몫"); ax[0][c].set_ylabel("r<0.98 칸 몫")
    ax[1][c].set_title(f"{LAB[k]} · 걸음 0 단계별 최소 r"); ax[1][c].set_ylabel("최소 r")
    for a in (ax[0][c], ax[1][c]):
        a.grid(color="#e5e4df", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0][c].legend(fontsize=8, ncol=5, frameon=False)
fig.suptitle("wolf 걸음 시작점 단계별 칸 높이비 r (절 CM113; x = 시작점을 만드는 커널 차례)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(od, "startpt.png"), dpi=110)
print("끝", od)
