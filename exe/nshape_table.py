#!/usr/bin/env python3
"""절 CM186: 새 형상 장면 결과 표 (R2 lbours/nshape/ours/<장면>_K<K>{,_t}/{summary,topo,visual}.json -> tex 표 조각 + md).
  python exe/nshape_table.py OUT_TABLE.tex OUT.md
통과 = 완주 & 오차 < 10 % & 단독 시간 판 ms_mean <= 16.7."""
import json, subprocess, sys
R = "r2:storage/result/anchorflow/lbours/nshape/ours"
SC = [("chair_paste", "chair", "페이스트 (PG plasticine: $E$ 2e5, $\\sigma_y$ 2e2)", "PG"),
      ("hotdog_milk", "hotdog", "물 같은 유체 (GF milk: NACC, $E$ 1e3, $\\nu$ 0.49, $\\beta$ 0)", "GF"),
      ("drums_honey", "drums", "꿀 같은 점소성 (PG foam: $E$ 2e4, $\\sigma_y$ 50, $\\eta$ 200)", "PG"),
      ("chair_plast", "chair", "플라스티신 (PG plasticine: $E$ 2e5, $\\sigma_y$ 5e3)", "PG"),
      ("materials_sand", "materials", "모래 (PG sand: wolf 값)", "PG")]


def cat(p):
    r = subprocess.run(["rclone", "cat", f"{R}/{p}"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def f(x, p=1):
    return "--" if x is None or x != x else f"{x:.{p}f}"


rows = []
for sc, obj, mat, solver in SC:
    best = None
    for K in (1, 2, 4, 8):
        s = cat(f"{sc}_K{K}/summary.json")
        if s is None:
            continue
        best = (K, s)
        if not s.get("diverged"):
            break
    if best is None:
        rows.append(dict(sc=sc, obj=obj, mat=mat, solver=solver, none=True)); continue
    K, s = best
    t = cat(f"{sc}_K{K}_t/summary.json") or {}
    tp = cat(f"{sc}_K{K}/topo.json") or {}
    v = cat(f"{sc}_K{K}/visual.json") or {}
    err = 100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None
    ms = t.get("ms_mean")
    done = s.get("frames_done"); nf = s["args"]["frames"]; dv = bool(s.get("diverged"))
    g, o = tp.get("gt", {}), tp.get("ours", {})
    ok = (not dv) and err is not None and err < 10 and ms is not None and ms <= 16.7
    rows.append(dict(sc=sc, obj=obj, mat=mat, solver=solver, K=K, done=f"{done}/{nf}", div=dv, err=err, ms=ms, ms_main=s.get("ms_mean"),
                     gesc=100 * g["esc_max"] if g.get("esc_max") is not None else None, oesc=100 * o["esc_max"] if o.get("esc_max") is not None else None,
                     gp=g.get("pieces_last"), op=o.get("pieces_last"), psnr=v.get("psnr"), ssim=v.get("ssim"), lpips=v.get("lpips"), ok=ok))
tex = ["\\begin{tabular}{llrrrrrrrrl}\\toprule",
       "형상 & 재질 (정답 솔버) & $K$ & 완주 & 오차 \\% & ms/프레임 & 이웃 이탈 정답/우리 \\% & 조각 정답/우리 & PSNR & SSIM & 통과 \\\\\\midrule"]
md = ["| 형상 | 재질 | K | 완주 | 오차 % | ms (단독) | 이탈 정답/우리 % | 조각 정답/우리 | PSNR | SSIM | LPIPS | 통과 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("none"):
        tex.append(f"{r['obj']} & {r['mat']} ({r['solver']}) & \\multicolumn{{9}}{{l}}{{결과 없음}} \\\\"); md.append(f"| {r['obj']} | {r['sc']} | 결과 없음 |"); continue
    c = r["done"] + (" (발산)" if r["div"] else "")
    tex.append(f"{r['obj']} & {r['mat']} ({r['solver']}) & {r['K']} & {c} & {f(r['err'])} & {f(r['ms'])} & {f(r['gesc'])}/{f(r['oesc'])} & {r['gp'] if r['gp'] is not None else '--'}/{r['op'] if r['op'] is not None else '--'} & {f(r['psnr'], 2)} & {f(r['ssim'], 3)} & {'예' if r['ok'] else '아니오'} \\\\")
    md.append(f"| {r['obj']} | {r['sc']} | {r['K']} | {c} | {f(r['err'])} | {f(r['ms'])} (본 판 {f(r['ms_main'])}) | {f(r['gesc'])}/{f(r['oesc'])} | {r['gp']}/{r['op']} | {f(r['psnr'], 2)} | {f(r['ssim'], 3)} | {f(r['lpips'], 3)} | {'예' if r['ok'] else '아니오'} |")
tex.append("\\bottomrule\\end{tabular}")
open(sys.argv[1], "w").write("\n".join(tex) + "\n"); open(sys.argv[2], "w").write("\n".join(md) + "\n"); print("\n".join(md))
json.dump(rows, open(sys.argv[2] + ".json", "w"), indent=1)
