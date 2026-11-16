#!/usr/bin/env python3
"""[final_fix_phys] 작업 폴더 phys.json·vis.json 모음 -> phys_metrics.json·vis_metrics.json + LaTeX 조각 둘 (\\begin{center}...\\end{center}).
  python3 exe/ffx_phys_tab.py <내려받은 phys 폴더> <출력 폴더> <논문 폴더>
"""
import glob
import json
import os
import sys

src, out, paper = sys.argv[1:4]
SC = [("L000", "레고"), ("pillow2sofa", "pillow2sofa"), ("bread", "bread"), ("wolf", "wolf")]
MS = [("mpm", "정답 MPM (PG)"), ("ours", "우리"), ("ipg", "i-PG"), ("omni", "OmniPhysGS"), ("vrgs", "VR-GS"), ("phystwin", "PhysTwin"), ("springgaus", "Spring-Gaus")]   # 절 CM192: i-PG·OmniPhysGS 추가
ELONLY = ("vrgs", "phystwin", "springgaus")
SKIP = ("dV_series", "E_series")
P, V = [], []
for d in sorted(glob.glob(f"{src}/*/")):
    j = os.path.basename(d.rstrip("/"))
    if os.path.exists(f"{d}/phys.json"):
        r = json.load(open(f"{d}/phys.json"))
        r = {k: v for k, v in r.items() if k not in SKIP}
        r["job"] = j
        P.append(r)
    if os.path.exists(f"{d}/vis.json"):
        ph = json.load(open(f"{d}/phys.json")) if os.path.exists(f"{d}/phys.json") else {}
        r = dict(method=ph.get("method", j.split("_")[0]), scene=ph.get("scene"), N=ph.get("N"), job=j)
        r.update(json.load(open(f"{d}/vis.json")))
        r["note"] = ph.get("note", "")
        V.append(r)


def order(r):
    return ([s for s, _ in SC].index(r["scene"]) if r["scene"] in [s for s, _ in SC] else 9, [m for m, _ in MS].index(r["method"]) if r["method"] in [m for m, _ in MS] else 9, str(r["N"]))


# 빠진 칸 표시
def missing(rows):
    have = {(r["method"], r["scene"]) for r in rows}
    add = []
    for s, _ in SC:
        for m, _ in MS:
            if (m, s) in have:
                continue
            if m in ELONLY and s in ("bread", "wolf"):
                add.append(dict(method=m, scene=s, N=None, note="final_blx (베이스라인 재질 확장 에이전트가 같은 정의로 잼)"))
            elif m == "springgaus" and s == "pillow2sofa":
                add.append(dict(method=m, scene=s, N=None, note="Spring-Gaus pillow2sofa 준비 판 (final_fix_sg) 학습 끝나지 않아 미측정"))
            else:
                add.append(dict(method=m, scene=s, N=None, note="미측정"))
    return sorted(rows + add, key=order)


# 자유낙하 해당 없음 (장면 조건, 정의는 그대로): 접촉 전 프레임이 없음 (레고: 0 프레임부터 최저점이 바닥 + 1.5 dx 안) 또는
# 외부 구속이 일을 함 (wolf: PG config release_particles_sequentially 로 층마다 풀림 -> 자유낙하 아님)
FF_NA = dict(wolf="wolf: PG config release_particles_sequentially (층별 순차 풀림) 구속 -> 자유낙하 구간 아님")
for r in P:
    if r.get("frames") is None:
        continue
    if r.get("scene") in FF_NA:
        r["freefall_mom_err_raw"] = r.get("freefall_mom_err"); r["freefall_mom_err"] = None; r["note"] = (r.get("note", "") + "; " + FF_NA[r["scene"]]).strip("; ")
    elif r.get("freefall_frames") == 0:
        r["freefall_mom_err"] = None; r["note"] = (r.get("note", "") + "; 접촉 전 프레임 없음 (0 프레임부터 최저점이 바닥 + 1.5 dx 안) -> 자유낙하 해당 없음").strip("; ")


def e0_na(rows):
    for r in rows:
        if r.get("frames") is not None and r.get("E0") is not None and abs(r["E0"]) < 1e-12:
            r["e_drift_max_raw"] = r.get("e_drift_max"); r["e_drift_max"] = None; r["e_ratio_last"] = None
            r["note"] = (r.get("note", "") + "; E_0 = 0 (중력 0 이고 처음 정지) -> 에너지 비 해당 없음").strip("; ")


e0_na(P)


def var_of(r):
    j = r.get("job", "")
    return j[len(f"ours_{r['scene']}_"):] if j.startswith("ours_") else None


PAR = ("def", "k2", "k4")
for r in P + V:
    if r["method"] == "ours":
        r["variant"] = var_of(r)
AP = [r for r in P if r["method"] == "ours" and not str(r.get("variant")).rsplit("_", 1)[-1] in ("r2", "r3")]   # 절 CM192: 노이즈용 반복 폴더 (ours_<장면>_def_r2·r3) 는 표에서 뺌
AV = [r for r in V if r["method"] == "ours" and not str(r.get("variant")).rsplit("_", 1)[-1] in ("r2", "r3")]
P = [r for r in P if r["method"] != "ours" or r.get("variant") in PAR]
V = [r for r in V if r["method"] != "ours" or r.get("variant") in PAR]
P, V = missing(P), missing(V)
json.dump(P, open(f"{out}/phys_metrics.json", "w"), ensure_ascii=False, indent=1)
json.dump(V, open(f"{out}/vis_metrics.json", "w"), ensure_ascii=False, indent=1)
AVARS = ["def", "be", "mid", "q1", "nofpi", "q27", "q64", "c075", "c15", "c20", "k2", "k4"] + [f"ss{n}.dump" for n in (1, 2, 4, 8, 16, 32, 64, 128)]


def amiss(rows):
    have = {(r["scene"], r["variant"]) for r in rows}
    rows = rows + [dict(method="ours", scene=s, variant=v, N=v, note="미측정") for s, _ in SC for v in AVARS if (s, v) not in have]
    return sorted(rows, key=lambda r: ([s for s, _ in SC].index(r["scene"]), AVARS.index(r["variant"]) if r["variant"] in AVARS else 99))


AP, AV = amiss(AP), amiss(AV)
json.dump(AP, open(f"{out}/abl_phys_metrics.json", "w"), ensure_ascii=False, indent=1)
json.dump(AV, open(f"{out}/abl_vis_metrics.json", "w"), ensure_ascii=False, indent=1)


def f(v, sc=1.0, d=2, sci=False):
    if v is None:
        return "--"
    v = v * sc
    if sci:
        return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{v:.{d}f}"


def cell_or_note(r, cells):
    if r.get("frames") is None and r.get("psnr") is None and r.get("warp_err") is None:
        n = r.get("note", "")
        tag = "final\\_blx" if "final_blx" in n else ("학습 중" if "final_fix_sg" in n else "미측정")
        return [r"\multicolumn{%d}{c}{%s}" % (len(cells), tag)]
    return cells


def MN(m):
    return dict(MS)[m]


def Nlab(r):
    n = r.get("N")
    if r.get("method") == "ours" and r.get("variant") in ("def", "k2", "k4"):              # 절 CM192: 새 판 지표 폴더의 N 은 원본 판 이름 -> 걸음 수 K
        return {"def": "1", "k2": "2", "k4": "4"}[r["variant"]]
    return "--" if n is None else str(n).replace("_", "\\_")


# 물리 표
L = [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{llr" + "c" * 10 + "}", r"\toprule",
     r"장면 & 방법 & $N$ & det$\le$0 (\%) & det 최소 & $|J{-}1|$ (\%) & $\Delta V$ (\%) & 관통 깊이 (\%$L$) & 관통 비율 (\%) & $\Delta E_{\max}/|E_0|$ & $E_T/E_0$ & 자유낙하 (\%) & 떨림 ($10^{-3}L$) \\",
     r"\midrule"]
for s, sn in SC:
    rows = [r for r in P if r["scene"] == s]
    for i, r in enumerate(rows):
        cells = [f(r.get("detneg_max"), 100, 3), f(r.get("detmin"), 1, 3), f(r.get("jdev_max"), 100, 2),
                 f(r.get("dV_last"), 100, 2), f(r.get("pen_depth_max"), 100, 2), f(r.get("pen_ratio_max"), 100, 2),
                 f(r.get("e_drift_max"), 1, 3), f(r.get("e_ratio_last"), 1, 3), f(r.get("freefall_mom_err"), 100, 2), f(r.get("jitter_d2"), 1e3, 2)]
        L.append(f"{sn if i == 0 else ''} & {MN(r['method'])} & {Nlab(r)} & " + " & ".join(cell_or_note(r, cells)) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", "",
      r"\begin{minipage}{0.97\linewidth}\scriptsize",
      r"정답 없이 재는 물리 지표 (안정성 실험 stabfinal 과 같은 코드: \texttt{lib/anchorflow/stabstat.py} 프레임 통계 + \texttt{exe/stab\_metrics.py}; 실행 \texttt{exe/ffx\_phys.py}, 정의 R2 \texttt{lbours/final\_fix/metrics\_def.md}). "
      r"모든 방법 같은 입자 (자료 .pt 번호)$\cdot$같은 질량 (밀도$\times dx^3$/칸 입자 수)$\cdot$같은 프레임 수, 속도 = 위치 중앙 차분. "
      r"det$\le$0 = det$F\le0$ 입자 비율의 프레임 최대; det 최소 = 전 프레임$\cdot$전 입자; $|J{-}1|$ = 프레임 평균 $|\det F-1|$ 의 최대; $\Delta V$ = 마지막 프레임 $\sum_i V_i(J_i-1)/\sum_i V_i$; "
      r"관통 = 부분표본 (32768 점) 바닥 아래 최대 깊이$/L$ ($L$ = 0 프레임 대각선) 와 바닥 아래 비율 최대; $\Delta E_{\max}/|E_0|$ = 운동+중력 위치 에너지 $\max_t(E_t-E_0)/|E_0|$ (위치 기준 = 바닥, 바닥 없으면 0 프레임 최저점), $E_T/E_0$ = 마지막/처음; "
      r"자유낙하 = 접촉 전 프레임에서 운동량 크기 $\sqrt{2MK_t}$ 와 $M|g|t$ 의 상대 오차 최대; 떨림 = 부분표본 $\|x_{t+1}-2x_t+x_{t-1}\|/L$ 전 구간 평균. "
      r"-- = 해당 없음: 바닥 없는 장면 (pillow2sofa$\cdot$bread, 실행이 바닥 없음) 의 관통$\cdot$자유낙하, bread 의 에너지 비 (중력 0$\cdot$처음 정지라 $E_0=0$), 레고의 자유낙하 (0 프레임부터 최저점이 바닥 $+1.5dx$ 안이라 접촉 전 프레임 없음), wolf 의 자유낙하 (PG config 의 층별 순차 풀림 구속이 일을 해 자유낙하가 아님), 변형 기울기가 없는 스프링-질량 표현 (PhysTwin$\cdot$Spring-Gaus) 의 det$\cdot$부피. "
      r"final\_blx = 탄성 전용 베이스라인의 bread$\cdot$wolf 는 재질 확장 실험에서 같은 정의로 잼. bread 는 손잡이 구속이 일을 하고 pillow2sofa 는 구간 고정이 풀리므로 에너지 증가가 곧 오류는 아님. "
      r"우리 $N$ = 프레임당 걸음 수 $K$ (K1 = 기본 판, 적응 쪼개기), 정답 MPM $N$ = PG 공식 config 서브스텝, 베이스라인 $N$ = 대표 설정 서브스텝.",
      r"\end{minipage}", r"\end{center}"]
open(f"{paper}/pareto_final_1115_phys.tex", "w").write("\n".join(L) + "\n")

# 시각 표
pv = V[0].get("pyiqa_version", "?") if V else "?"
for r in V:
    if r.get("pyiqa_version"):
        pv = r["pyiqa_version"]; break
L = [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{llr" + "c" * 8 + "}", r"\toprule",
     r"장면 & 방법 & $N$ & NIQE$\downarrow$ & BRISQUE$\downarrow$ & MUSIQ$\uparrow$ & CLIP-IQA$\uparrow$ & warp ($10^{-3}$)$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\",
     r"\midrule"]
for s, sn in SC:
    rows = [r for r in V if r["scene"] == s]
    for i, r in enumerate(rows):
        cells = [f(r.get("niqe"), 1, 2), f(r.get("brisque"), 1, 1), f(r.get("musiq"), 1, 1), f(r.get("clipiqa"), 1, 3),
                 f(r.get("warp_err"), 1e3, 2), f(r.get("psnr"), 1, 2), f(r.get("ssim"), 1, 3), f(r.get("lpips"), 1, 3)]
        L.append(f"{sn if i == 0 else ''} & {MN(r['method'])} & {Nlab(r)} & " + " & ".join(cell_or_note(r, cells)) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", "",
      r"\begin{minipage}{0.97\linewidth}\scriptsize",
      r"시각 품질 (모든 방법$\cdot$정답 MPM 을 같은 렌더러 \texttt{exe/lb\_render.py} (F 반영 공분산 3DGS) 로 장면마다 같은 카메라 (정답 config 카메라) 에서 렌더, 프레임 1..$T$ 평균; 실행 \texttt{exe/ffx\_vis.py}, 정의 R2 \texttt{lbours/final\_fix/metrics\_def.md}). "
      r"정답 없이: NIQE, BRISQUE, MUSIQ (KonIQ), CLIP-IQA -- pyiqa " + pv + r" \texttt{create\_metric} 기본 가중치; "
      r"warp = 시간 깜빡임 warping error (Lai et al. 2018): RAFT (torchvision raft\_large C\_T\_SKHT\_V2) 양방향 광류로 $I_t$ 를 $t{+}1$ 로 끌어와 $\sum M\|I_{t+1}-W(I_t)\|^2/\sum M$ (RGB 0--1 채널 합), "
      r"가림 마스크 $M$ = 앞뒤 광류 일관성 (Sundaram et al. 2010: $|F_b+F_f(x{+}F_b)|^2<0.01(|F_b|^2+|F_f|^2)+0.5$) 이고 영상 안인 픽셀, 프레임 쌍 평균. "
      r"정답 렌더 대비 (참고): PSNR, SSIM (skimage), LPIPS (alex). final\_blx = 탄성 전용 베이스라인의 bread$\cdot$wolf 는 재질 확장 실험에서 같은 정의로 잼.",
      r"\end{minipage}", r"\end{center}"]
open(f"{paper}/pareto_final_1115_vis.tex", "w").write("\n".join(L) + "\n")

# ablation 표 (우리 판마다: 물리 + 시각)
VI = {(r["scene"], r["variant"]): r for r in AV}
L = [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
     r"\begin{tabular}{ll" + "c" * 13 + "}", r"\toprule",
     r"장면 & 판 & det$\le$0 (\%) & det 최소 & $\Delta V$ (\%) & 관통 (\%$L$) & $\Delta E_{\max}/|E_0|$ & 자유낙하 (\%) & 떨림 ($10^{-3}L$) & NIQE$\downarrow$ & MUSIQ$\uparrow$ & CLIP-IQA$\uparrow$ & warp ($10^{-3}$) & PSNR & LPIPS \\",
     r"\midrule"]
for s_, sn in SC:
    rows = [r for r in AP if r["scene"] == s_]
    for i, r in enumerate(rows):
        v = VI.get((s_, r["variant"]), {})
        cells = [f(r.get("detneg_max"), 100, 3), f(r.get("detmin"), 1, 3), f(r.get("dV_last"), 100, 2), f(r.get("pen_depth_max"), 100, 2),
                 f(r.get("e_drift_max"), 1, 3), f(r.get("freefall_mom_err"), 100, 2), f(r.get("jitter_d2"), 1e3, 2),
                 f(v.get("niqe"), 1, 2), f(v.get("musiq"), 1, 1), f(v.get("clipiqa"), 1, 3), f(v.get("warp_err"), 1e3, 2), f(v.get("psnr"), 1, 2), f(v.get("lpips"), 1, 3)]
        if r.get("frames") is None:
            cells = [r"\multicolumn{13}{c}{미측정}"]
        L.append(f"{sn if i == 0 else ''} & " + r["variant"].replace(".dump", "").replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", "", r"\begin{minipage}{0.97\linewidth}\scriptsize",
      r"ablation 판마다 정답 없이 재는 물리 지표와 시각 품질 (파레토 표와 같은 정의$\cdot$같은 코드 \texttt{exe/ffx\_phys.py}$\cdot$\texttt{exe/ffx\_vis.py}, 정의 R2 \texttt{lbours/final\_fix/metrics\_def.md}). "
      r"덤프 = R2 \texttt{lbours/final\_fix/runs\_nd/<장면>\_<판>\_r1} (반복 1); ss$N$ (E, 프레임당 걸음 $N$, 50 프레임) 은 같은 명령에 덤프만 붙여 다시 돌림 (\texttt{ss$N$.dump}). "
      r"지표 정의는 파레토 물리$\cdot$시각 표 주석과 같음; 관통$\cdot$자유낙하는 바닥 없는 장면 (pillow2sofa$\cdot$bread) 에서 --; PSNR$\cdot$LPIPS 는 같은 카메라 정답 MPM 렌더 대비 (ss$N$ 은 50 프레임까지).",
      r"\end{minipage}", r"\end{center}"]
open(f"{paper}/ablation_final_1115_metrics.tex", "w").write("\n".join(L) + "\n")
print("ok", len(P), len(V), len(AP), len(AV))
