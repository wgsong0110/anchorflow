#!/usr/bin/env python3
"""절 CM182: GF 공개 장면 결과 표 (R2 lbours/gfscenes/ours/<run>/{summary,topo,visual}.json -> tex + md).
  python gfs_table.py OUT.tex OUT.md"""
import json, subprocess, sys
R = "r2:storage/result/anchorflow/lbours/gfscenes/ours"; V = "result/anchorflow/v20261009"
def cat(p):
    r = subprocess.run(["rclone", "cat", f"{R}/{p}"], capture_output=True, text=True)
    try: return json.loads(r.stdout)
    except Exception: return None
# (장면, 실행 디렉토리, K, 영상, 대응 메모)
ROWS = [
 ("dragonfruit", "dragonfruit_v2", 1, "gf_dragonfruit_v2_cmp.mp4", "sticky 평면 z=0.5, 감쇠 보정 (substep\\_dt)"),
 ("kiwi", "kiwi", 1, "gf_kiwi_cmp.mp4", "cut(z=0.5)$\\to$sticky 평면"),
 ("pineple", "pineple_K2", 2, "gf_pineple_K2_cmp.mp4", "cut(z=0.5)$\\to$sticky, $\\beta{=}0,\\varphi{=}0$"),
 ("toast", "toast", 1, "gf_toast_cmp.mp4", "바닥 z=3dx"),
 ("pumkin", "pumkin", 1, "gf_pumkin_cmp.mp4", "바닥 z=3dx, cap\\_mul 2"),
 ("lollipop", "lollipop", 1, "gf_lollipop_cmp.mp4", "cut(z=0) 무효, 바닥 z=3dx"),
 ("cake", "cake", 1, "gf_cake_cmp.mp4", "jelly, 바닥 z=3dx"),
 ("milk", "milk", 1, "gf_milk_cmp.mp4", "NACC, FLIP 0.99. K=2,4,8 도 발산 (22--26프레임)"),
 ("sand\\_castle", "sand_castle", 1, "gf_sand_castle_cmp.mp4", "cut+impulse"),
 ("젤리+총알", "jellybullet", 1, "gf_jellybullet_gt.mp4 (정답만; 비교 영상 없음)", "공식 프레임 일정 (0--19, 200--239, 381--), 메모리 부족으로 135/199"),
]
rows = []
for name, d, K, vid, note in ROWS:
    s, t, v = cat(f"{d}/summary.json"), cat(f"{d}/topo.json"), cat(f"{d}/visual.json")
    if s is None:
        rows.append(dict(name=name, ok=False, note=note, vid=vid)); continue
    fd = s.get("frames_done"); nf = s["args"]["frames"]; dv = s.get("diverged")
    r = dict(name=name, ok=True, K=s["args"].get("K"), done=f"{fd}/{nf}", div=dv, err=100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None, ms=s.get("ms_mean"), note=note, vid=vid)
    if t:
        r.update(gesc=100 * t["gt"]["esc_max"], gp=t["gt"]["pieces_last"], oesc=100 * t["ours"]["esc_max"] if t["ours"]["esc_max"] is not None else None, op=t["ours"]["pieces_max"], olast=100 * t["ours"]["esc_last"] if t["ours"]["esc_last"] is not None else None)
    if v: r.update(psnr=v["psnr"], ssim=v["ssim"], lpips=v["lpips"])
    rows.append(r)
def f(x, p=1): return "--" if (x is None or x != x) else f"{x:.{p}f}"
tex = ["% lualatex gf_scenes_results.tex", "\\documentclass[10pt]{article}", "\\usepackage[margin=0.6in,landscape]{geometry}", "\\usepackage{fontspec}", "\\setmainfont{Noto Serif CJK KR}", "\\usepackage{booktabs,array}",
       "\\title{GaussianFluent 공개 장면 (watermelon 제외): 우리 판 결과 (절 CM182)}", "\\date{2026-11-15 기준}", "\\begin{document}\\maketitle", "\\small",
       "\\begin{table}[h]\\centering\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrrrrrrrrrrl}\\toprule",
       "장면 & K & 완주 & 오차 \\% & 정답 이탈 \\% & 우리 이탈 \\% (마지막) & 조각 정답/우리 & PSNR & SSIM & LPIPS & ms/프레임 & 비고 \\\\\\midrule"]
md = ["| 장면 | K | 완주 프레임 | 평균 오차 % | 정답 이웃 이탈 최대 % | 우리 이탈 최대 % (마지막) | 조각 정답/우리 최대 | PSNR | SSIM | LPIPS | ms/프레임 (비단독) | 영상 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
DIV = [r for r in rows if r["ok"] and r["div"]]; NONE = [r for r in rows if not r["ok"]]
rows = [r for r in rows if r["ok"] and not r["div"]]
for r in rows:
    if not r["ok"]:
        tex.append(f"{r['name']} & \\multicolumn{{10}}{{l}}{{결과 없음}} & {r['note']} \\\\"); md.append(f"| {r['name'].replace(chr(92),'')} | 결과 없음 | | | | | | | | | | |"); continue
    c = f"{r['done']}" + (" (발산)" if r["div"] else "")
    tex.append(f"{r['name']} & {r['K']} & {c} & {f(r['err'])} & {f(r.get('gesc'))} & {f(r.get('oesc'))} ({f(r.get('olast'))}) & {r.get('gp','--')}/{r.get('op','--')} & {f(r.get('psnr'),2)} & {f(r.get('ssim'),3)} & {f(r.get('lpips'),3)} & {f(r['ms'],0)} & {r['note']} \\\\")
    md.append(f"| {r['name'].replace(chr(92),'')} | {r['K']} | {c} | {f(r['err'])} | {f(r.get('gesc'))} | {f(r.get('oesc'))} ({f(r.get('olast'))}) | {r.get('gp','--')}/{r.get('op','--')} | {f(r.get('psnr'),2)} | {f(r.get('ssim'),3)} | {f(r.get('lpips'),3)} | {f(r['ms'],0)} | {V}/{r['vid']} |")
tex += ["\\bottomrule\\end{tabular}", "\\caption{오차는 정답 대비 입자 평균 위치 오차 (기존 정의, 완주한 프레임 평균). 이웃 이탈 = 초기 k-NN 이웃이 초기 거리의 3배 넘게 벌어진 비율. 발산한 장면은 발산 전 프레임만의 값. 프레임 시간은 진단·렌더 훅이 켜진 비단독 값.}\\end{table}", "\\end{document}"]
dv = ["", "발산하거나 결과가 없는 장면 (사용자 11-15: 부분 결과는 표에 쓰지 않음; 바닥 접촉 뒤집힘 수정 뒤 다시 돌릴 대상):"]
for r in DIV: dv.append(f"- {r['name'].replace(chr(92),'')}: {r['done']} 프레임에서 발산 (K={r['K']})")
for r in NONE: dv.append(f"- {r['name'].replace(chr(92),'')}: 결과 없음")
open(sys.argv[1], "w").write("\n".join(tex)); open(sys.argv[2], "w").write("\n".join(md + dv)); print("\n".join(md + dv))
