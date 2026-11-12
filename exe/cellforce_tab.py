"""절 CM111 표·그림 (로컬, 작은 json 만): exe/cellforce_an.py 의 cellforce_<판>.json -> cellforce.md · cellforce.png.
사용: python exe/cellforce_tab.py <json 디렉토리> <출력 디렉토리> [H9판 H5판]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

jd, od = sys.argv[1:3]
R9, R5 = (sys.argv[3:5] if len(sys.argv) >= 5 else ("g9_1", "g5_1"))
for f in fm.findSystemFonts():
    if any(k in os.path.basename(f).lower() for k in ("nanum", "notosanscjk", "malgun")):
        fm.fontManager.addfont(f)
for nm in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
    if any(x.name == nm for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = nm; break
plt.rcParams["axes.unicode_minus"] = False
D = {}
for k in (R9, R5):
    p = os.path.join(jd, f"cellforce_{k}.json")
    if os.path.exists(p):
        D[k] = json.load(open(p))
LAB = {R9: "H9 wolf", R5: "H5 wolf (대조)"}
RL = ["<0.5", "0.5~0.7", "0.7~0.9", "0.9~0.98", "0.98~1.02", ">1.02"]
LL = ["0", "1", "2", "3", "4", "5+"]
SH = ["관성", "중력", "탄성", "장벽", "정규화", "합"]
SHN = {"관성": "관성 (중력 뺌)", "중력": "중력", "탄성": "탄성", "장벽": "바닥 장벽", "정규화": "정규화 (P1X 고스트)", "합": "합 = 남은 잔차"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b"]


def g(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.2g}" if a < 1e-3 else f"{v:.3f}"


def tab(st, keys, kind):
    hd = f"| {kind} | 칸 수 | " + " | ".join(f"{SHN[q]} 중앙 (평균) [누름 %]" for q in SH) + " |"
    L = [hd, "|" + "---|" * (len(SH) + 2)]
    for lb in keys:
        s = st.get(lb, {})
        if not s.get("n"):
            continue
        L.append(f"| {lb} | {s['n']} | " + " | ".join(f"{g(s[q]['med'])} ({g(s[q]['mean'])}) [{100 * s[q]['pos']:.0f}]" for q in SH) + " |")
    return "\n".join(L)


md = ["# wolf 칸 세로 압축 -- 증분 포텐셜 항별 일반화 힘 (절 CM111, 묶음 cellforce)", ""]
for k, d in D.items():
    ck = d.get("check", {})
    rel = [v["stat"]["끝"]["chk_max"] / max(v["stat"]["끝"]["chk_ref"], 1e-30) for v in ck.values() if v.get("stat") and "끝" in v["stat"]]
    codes = [v["itb"][1] for v in ck.values() if v.get("itb")]
    md.append(f"- {LAB[k]} ({k}): 프레임 {len(d['frames'])} 개, 항 이름 {d['terms']}; 재합 대 풀이기 잔차 최대 상대 차 {max(rel) if rel else float('nan'):.1e}; "
              f"끝 평가 (걸음·반복) 분포 {dict(sorted({c: codes.count(c) for c in set(codes)}.items()))}")
    md.append(f"  - 이름별 |F| 합 (끝 칸, 모든 프레임): " + ", ".join(f"{n} {v['sumabs']:.3g}" for n, v in d["regnames"].items()))
md.append("")
for k, d in D.items():
    for ph in ("f5", "f8~9", "f10~30", "f31~50"):
        P = d["phases"].get(ph)
        if not P:
            continue
        for sl in ("끝", "시작"):
            md += [f"## {LAB[k]} · {ph} · {sl} 평가 -- r 구간 (모든 칸)", "", tab(P[sl]["rbin"], RL, "r"), ""]
            if ph == "f10~30" and sl == "끝":
                md += [f"## {LAB[k]} · {ph} · 끝 -- r 구간 (내부 칸만: 꼭짓점에 확장 묶인 노드·바닥 T≠I 노드 없음)", "", tab(P[sl]["rbin_inner"], RL, "r"), ""]
                md += [f"## {LAB[k]} · {ph} · 끝 -- 층", "", tab(P[sl]["layer"], LL, "층"), ""]
open(os.path.join(od, "cellforce.md"), "w").write("\n".join(md) + "\n")

# 그림: 행 = 판, 열 = 프레임 묶음 (f5 · f8~9 · f10~30), 막대 = 항 중앙값 (끝 평가), x = r 구간; y 는 대칭 로그
PHS = ["f5", "f8~9", "f10~30"]
ks = list(D)
fig, ax = plt.subplots(len(ks), len(PHS), figsize=(18, 5.2 * len(ks)), squeeze=False)
for i, k in enumerate(ks):
    for j, ph in enumerate(PHS):
        a = ax[i][j]
        P = D[k]["phases"].get(ph)
        if not P:
            a.set_visible(False); continue
        st = P["끝"]["rbin"]
        bins = [b for b in RL if st.get(b, {}).get("n")]
        x = np.arange(len(bins)); w = 0.8 / len(SH)
        for t, (q, c) in enumerate(zip(SH, COL)):
            v = [st[b][q]["med"] for b in bins]
            a.bar(x + (t - (len(SH) - 1) / 2) * w, v, w * 0.92, color=c, label=SHN[q])
        a.set_yscale("symlog", linthresh=1e-3)
        a.axhline(0, color="#52514e", lw=0.8)
        a.set_xticks(x); a.set_xticklabels([f"{b}\n({st[b]['n']} 칸)" for b in bins], fontsize=9)
        a.set_title(f"{LAB[k]} · {ph} · 끝 평가", fontsize=12)
        a.set_xlabel("칸 두께 비 r = h끝/h시작")
        if j == 0:
            a.set_ylabel("일반화 힘 중앙값 (N)  양수 = 누름, 음수 = 버팀")
        a.grid(axis="y", color="#e5e4df", lw=0.6); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
h_, l_ = ax[0][0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=len(SH), fontsize=11, frameon=False)
fig.suptitle("wolf 칸 세로 압축 모드에 걸리는 항별 일반화 힘 중앙값 (절 CM111, y 대칭 로그)", fontsize=14)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(od, "cellforce.png"), dpi=110)
print("끝", od)
