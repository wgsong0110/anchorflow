"""[blx] 논문 폴더 pareto_final_1115_blx.tex 생성 (재질 확장 설명 + wolf·bread 확장 베이스라인 물리·시각 지표 표). 입력 R2 lbours/final_pa/blx/<태그>/g/*_dump/{phys.json,vis_ffx.json}."""
import json, subprocess, sys
R = "r2:storage/result/anchorflow/lbours/final_pa/blx"
CELLS = [("wolf", "vrgs", "A", 400), ("wolf", "phystwin", "A", 400), ("wolf", "springgaus", None, None),
         ("bread", "vrgs", "A", 100), ("bread", "phystwin", "A", 200), ("bread", "springgaus", None, None)]
if len(sys.argv) > 1:                                    # 예: springgaus:wolf:B:200
    for a in sys.argv[1:]:
        m, t, tag, n = a.split(":")
        CELLS = [(t, m, tag, int(n)) if (c[0], c[1]) == (t, m) else c for c in CELLS]
LAB = dict(vrgs="VR-GS", phystwin="PhysTwin", springgaus="Spring-Gaus")


def cat(p):
    r = subprocess.run(["rclone", "cat", p], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def f(v, fmt):
    return "--" if v is None or v != v or v in (float("inf"),) else fmt % v


rows_p, rows_v = [], []
for (t, m, tag, n) in CELLS:
    nm = f"{t} & {LAB[m]}"
    if tag is None:
        rows_p.append(nm + r" & \multicolumn{8}{l}{측정 대기 (공식 3 단계 학습 중)}\\"); rows_v.append(nm + r" & \multicolumn{8}{l}{측정 대기}\\"); continue
    J = f"{R}/{tag}/g/{m}_{t}_n{n}_dump"
    P, V = cat(f"{J}/phys.json") or {}, cat(f"{J}/vis_ffx.json") or {}
    det = "F 없음" if P.get("detneg_max") is None else f"{100 * P['detneg_max']:.2f} / {P['detmin']:.3f}"
    vol = "F 없음" if P.get("jdev_max") is None else f"{P['jdev_max']:.3f} / {P['dV_last']:+.3f}"
    pen = "바닥 없음" if t == "bread" else f"{100 * P['pen_ratio_max']:.2f} / {100 * P['pen_depth_max']:.2f}"
    en = r"$E_0{=}0^\ddagger$" if not P.get("E0") else f"{P['e_drift_max']:.2e} / {P['e_ratio_last']:.3f}"
    ff = r"--$^\S$"
    rows_p.append(f"{nm} & {det} & {vol} & {pen} & {en} & {ff} & {f(P.get('jitter_d2'), '%.2e')} & {f(P.get('jitter_hf'), '%.3f')} & {n}\\\\")
    rows_v.append(f"{nm} & {f(V.get('niqe'), '%.2f')} & {f(V.get('brisque'), '%.1f')} & {f(V.get('musiq'), '%.1f')} & {f(V.get('clipiqa'), '%.3f')} & "
                  f"{f(V.get('warp_err'), '%.2e')} & {f(V.get('psnr'), '%.2f')} & {f(V.get('ssim'), '%.3f')} & {f(V.get('lpips'), '%.3f')}\\\\")
L = [r"% 자동 생성 (/tmp/blx/mk_blx_tex.py; 가지 final_pa_bl 절 CM193 -- 재질 확장 베이스라인)",
     r"\section*{wolf$\cdot$bread 의 VR-GS$\cdot$PhysTwin$\cdot$Spring-Gaus: 재질 확장 구현 (원 논문은 탄성 전용)$^\dagger$}",
     r"세 방법의 원 논문$\cdot$공식 코드는 탄성 전용이라 (예전 판은 해당 없음으로 비웠음, 근거 R2 \texttt{lbours/final\_pa/bl/na.json}) 표현과 적분기는 그대로 두고 재질 모델만 PhysGaussian 원본 소성 식에 맞춰 넣었다 (가지 final\_pa\_bl 702f93d$\cdot$44cc2cf, 문서 절 CM193).",
     r"\begin{itemize}",
     r"\item \textbf{VR-GS} (XPBD + 사면체 케이지, 직접 구현): 사면체마다 $F = F_e F_p$ ($D_m^{-1}F_p^{-1}$ 를 상태로), 매 서브스텝 구속 반복 뒤 PG \texttt{mpm\_utils.py} 반환 사상을 줄 단위로 (모래 = \texttt{sand\_return\_mapping} Drucker--Prager, 마찰각 30$^\circ$; bread = \texttt{von\_mises\_return\_mapping\_with\_damage}, 항복 200 Pa, softening 0.1, 항복 응력 $\le 0$ 이면 응력 0). 탄성 구속은 $F_e$ 기준 (plasticine = 기존 고정 공회전 $C_R, C_V$; 모래 = PG 와 같은 Hencky 에너지를 $C_1=\|\log\Sigma\|$, $C_2=\mathrm{tr}\log\Sigma$ 두 구속으로, 순응도는 같은 $1/(2\mu V)$, $1/(\lambda V)$). $E$ 는 기존과 같은 논문 절차 (황금분할 10 회, $\Delta t=10^{-4}$ s$\cdot$반복 11): wolf $7.75\times10^5$ (정답 $5\times10^7$), bread $4876$ (정답 $2000$).",
     r"\item \textbf{PhysTwin$\cdot$Spring-Gaus} (스프링--질량, 공식 코드): 스프링마다 소성 휴지 길이 (로그 변형률 $\varepsilon=\log(l/l_0)$ 이 항복을 넘으면 $l_0$ 가 따라감, 매 서브스텝 적분 뒤). plasticine: $\varepsilon_y=\sqrt{3/2}\,\sigma_y/E = 0.122$ (단축 응력에서 PG von Mises 항복이 되는 축 변형률). 모래 (\textbf{근사} -- 스프링으로는 정확히 옮길 수 없음): 끝점 체적 변형률 추정 (그 점 스프링 $\varepsilon$ 평균 $\times 3$) 의 평균 $\mathrm{tr}$, 축 편차 $e=\varepsilon-\mathrm{tr}/3$, $\delta\gamma=\sqrt{3/2}|e|+\frac{3\lambda+2\mu}{2\mu}\alpha\,\mathrm{tr}$ (PG 와 같은 $\alpha$, $\nu$); 인장 ($\mathrm{tr}>0$) 항복이면 응력 0 ($l_0=l$), 압축이면 $e$ 를 $\delta\gamma$ 만큼 줄임. 휴지 길이는 미분하지 않는 상태 (롤아웃 시작마다 처음 값). 물성 (강성 등) 은 각 방법의 공식 절차 그대로 (PhysTwin CMA-ES 20 세대 + 경사 학습 200, Spring-Gaus 다시점 영상 3 단계 학습), 소성 매개변수는 PG config 에서만.",
     r"\item \textbf{경계}: PG config 그대로 -- wolf \texttt{release\_particles\_sequentially} (50 상자, 구간 끝에 풀림) 는 구간 고정, bread 당김 띠 (\texttt{enforce\_particle\_translation} $\pm0.2$ m/s, 아래 띠 2.2 s 까지) 는 구간 등속 구동 (VR-GS 질량 무한 정점, PhysTwin 적분 커널의 고정 몫, Spring-Gaus 서브스텝 끝 앵커). 판정 시각: VR-GS$\cdot$PhysTwin 프레임 시작, Spring-Gaus 서브스텝.",
     r"\item 발산$\cdot$실패 없음 (세 방법 모두 3/3 완주; 서브스텝$\cdot$반복은 기존 대표 설정 그대로, 바꾼 것 없음).",
     r"\end{itemize}",
     r"\paragraph{정답 없이 재는 물리 지표} (final\_fix 와 같은 코드 \texttt{exe/ffx\_phys.py} b06c860, 대표 설정 덤프 판 하나; 스프링--질량은 표현에 $F$ 가 없어 뒤집힘$\cdot$부피 없음)",
     r"\begin{center}\scriptsize\begin{tabular}{lllllllllr}\toprule",
     r"장면 & 방법 & det$\le$0 \% / det 최소 & $|J{-}1|$ 최대 / $\Delta V$ 끝 & 관통 비율 / 깊이 (\%$L$) & 에너지 최대 증가 / 끝$/$처음 & 자유낙하 & 떨림 $d_2/L$ & 고주파 비 & N \\\midrule"] + rows_p + [
     r"\bottomrule\end{tabular}\end{center}",
     r"{\footnotesize $^\ddagger$ bread 는 중력 0$\cdot$정지 출발이라 $E_0=0$ (비가 정의 안 됨). $^\S$ 자유낙하 구간 없음 (wolf 는 0 프레임부터 순차 방출 고정, bread 는 중력 0; 공통 코드 값은 운동량 0 이라 1.0). 정지 구간 떨림은 공통 코드에 따로 없어 전체 $d_2$ 만.}",
     r"\paragraph{시각 품질} (final\_fix 와 같은 코드 \texttt{exe/ffx\_vis.py}: 같은 렌더러$\cdot$카메라 (\texttt{lb\_render}), pyiqa 0.1.16, RAFT warping error, 정답 렌더 대비 PSNR$\cdot$SSIM$\cdot$LPIPS; 프레임 평균)",
     r"\begin{center}\scriptsize\begin{tabular}{llrrrrrrrr}\toprule",
     r"장면 & 방법 & NIQE$\downarrow$ & BRISQUE$\downarrow$ & MUSIQ$\uparrow$ & CLIP-IQA$\uparrow$ & warp err$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\\midrule"] + rows_v + [
     r"\bottomrule\end{tabular}\end{center}"]
open("/home/wgsong/workspace/writing/anchorflow/pareto_final_1115_blx.tex", "w").write("\n".join(L) + "\n")
print("\n".join(rows_p + rows_v))
