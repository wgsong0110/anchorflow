"""[stabfinal CM192] 장면 x 지표마다 우리 판이 1위인지 판정 (확정 최소 N 의 값, 표에 보이는 자리수로 반올림해 비교; 같은 값은 공동 1위).
입력 = stabfinal_tab.py 와 같은 수집 폴더. 출력 = 표준 출력 표 + <paper>/stab_final_1115_rank.tex (선택).
  python exe/stabfinal_rank.py <수집 폴더> [<논문 폴더>]
"""
import json, os, sys
import numpy as np
root = sys.argv[1]; paper = sys.argv[2] if len(sys.argv) > 2 else ""
SC = ["lego", "pillow2sofa", "ship_plastic_lego", "mic_metal_lego"]
SN = dict(lego="Lego", pillow2sofa="pillow2sofa", ship_plastic_lego="ship", mic_metal_lego="mic")
TAGS = dict(lego="_f5", pillow2sofa="_f5", ship_plastic_lego="_f7", mic_metal_lego="_f7")
MS = ["pg", "ipg", "omni", "ours"]
LEGO_OLD = dict(pg=(43, [0.00, 0.61, 0.44, 0.00, 0.73]), ipg=(8, [0.00, 0.63, 0.39, 0.00, 0.52]))   # stabfinal_tab.py 와 같음 (% 값)
# (이름, 출처, 키, 낮을수록 좋음)
MET = [("최소 N", "nmin", None, True),
       ("뒤집힘", "phys", "detneg", True), ("관통", "phys", "pen", True), ("떨림", "phys", "jit", True), ("자유낙하", "phys", "ff", True), ("Chamfer", "phys", "cd", True),
       ("에너지 증가", "ex", "e_drift_max", True), ("부피 변화", "ex", "jdev_max", True),
       ("NIQE", "vis", "niqe", True), ("BRISQUE", "vis", "brisque", True), ("CLIP-IQA", "vis", "clipiqa", False), ("MUSIQ", "vis", "musiq", False),
       ("깜빡임", "vis", "warp_err", True), ("PSNR", "ref", "psnr", False), ("SSIM", "ref", "ssim", False), ("LPIPS", "ref", "lpips", True)]


def J(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def nmin(s, m):
    if s == "lego" and m in LEGO_OLD:
        return LEGO_OLD[m][0]
    p = f"{root}/{s}/plan_{m}{TAGS[s] if m == 'ours' else ''}.json"
    return J(p).get("N_min")


def val(s, m, src, key):
    n = nmin(s, m)
    if src == "nmin":
        return n
    if not n:
        return None
    name = f"{m}{TAGS[s] if m == 'ours' else ''}_N{n}"
    if src == "phys":
        if s == "lego" and m in LEGO_OLD:
            return LEGO_OLD[m][1][["detneg", "pen", "jit", "ff", "cd"].index(key)] / 100
        x = J(f"{root}/{s}/an.json").get("methods", {}).get(m, {})
        vs = [r["phys"][key][0] for r in x.get("runs", {}).values() if r.get("N") == n and r.get("phys") and r["phys"].get(key) and r["phys"][key][0] is not None]
        return float(np.mean(vs)) if vs else None
    if src == "ex":
        return J(f"{root}/{s}/extra.json").get("phys", {}).get(name, {}).get(key)
    if src == "vis":
        return J(f"{root}/{s}/extra.json").get("visual", {}).get(name, {}).get(key)
    if src == "ref":
        return J(f"{root}/{s}/vis/vis.json").get(name, {}).get(key)


rows = []; lose = []
for s in SC:
    for nm, src, key, low in MET:
        v = {m: val(s, m, src, key) for m in MS}
        rd = dict(phys=lambda x: round(100 * x, 2), ex=lambda x: round(100 * x, 2), nmin=lambda x: x, ref=None, vis=None)[src] or \
            (lambda x, k=key: round(1e3 * x, 2) if k == "warp_err" else round(x, {"niqe": 2, "brisque": 1, "clipiqa": 3, "musiq": 2, "psnr": 2, "ssim": 3, "lpips": 3}[k]))
        have = {m: rd(x) for m, x in v.items() if x is not None}     # 표에 보이는 자리수로 판정 (같으면 공동 1위)
        if "ours" not in have:
            res = "미측정"
        else:
            best = min(have.values()) if low else max(have.values())
            tol = 1e-12 + 1e-9 * abs(best)
            win = abs(have["ours"] - best) <= tol
            res = "1위" if win else "짐"
            if not win:
                lose.append((SN[s], nm, {m: have.get(m) for m in MS}))
        rows.append((SN[s], nm, v, res))
for r in rows:
    print(f"{r[0]:12s} {r[1]:10s} " + " ".join(f"{m}={('%.4g' % r[2][m]) if r[2][m] is not None else '--':>9s}" for m in MS) + f"  {r[3]}")
print("\n지는 칸", len(lose))
for x in lose:
    print(x)
if paper:
    t = [r"\begin{table}[h]\centering\scriptsize\caption{장면$\cdot$지표별 우리 판 순위 (확정 최소 $N$ 의 값, 방법 넷 중; 1위 = 가장 좋음 (같으면 공동), 짐 = 다른 방법이 더 좋음, 미측정 = 우리 값 없음).}",
         r"\begin{tabular}{l" + "c" * len(MET) + r"}\toprule 장면 & " + " & ".join(n for n, *_ in MET) + r" \\\midrule"]
    for s in SC:
        t.append(SN[s] + " & " + " & ".join(r[3].replace("1위", r"\checkmark").replace("짐", r"$\times$") for r in rows if r[0] == SN[s]) + r" \\")
    t.append(r"\bottomrule\end{tabular}\end{table}")
    open(f"{paper}/stab_final_1115_rank.tex", "w").write("\n".join(t))
