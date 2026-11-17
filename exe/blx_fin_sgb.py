"""Spring-Gaus bread 측정값을 표 (table·phys·vis) 에 넣고 평균 행 다시 계산, pareto_blx.json 갱신. 인자: <grid.json> <phys.json> <vis_ffx.json> <WR 폴더>"""
import json, re, sys, statistics as S
G, PH, VI, WR = sys.argv[1:5]
g = [r for r in json.load(open(G)) if r['method'] == 'springgaus' and r['scene'] == 'bread'][0]
assert g['ok'] >= 1, g
ph, vi = json.load(open(PH)), json.load(open(VI))
N = g['N']; err, ms = g['err'], g['ms']
jit = 1e3 * ph['jitter_d2']
# --- table.tex ---
T = f"{WR}/pareto_final_1115_table.tex"; s = open(T).read()
row1 = f" & Spring-Gaus$^{{\\dagger\\ast}}$ & {err:.2f} & {ms:.1f} & -- & -- & -- & {jit:.2f} & --\\\\"
row2 = (f" & Spring-Gaus$^{{\\dagger\\ast}}$ & {vi['niqe']:.2f} & {vi['brisque']:.1f} & {vi['musiq']:.1f} & {vi['clipiqa']:.3f} & {1e3 * vi['warp_err']:.2f} & "
        f"{vi['psnr']:.2f} & {vi['ssim']:.3f} & {vi['lpips']:.3f}\\\\")
L = s.split('\n'); out = []; blk = None; k = 0
for l in L:
    if re.match(r'^[A-Za-z평]', l):
        blk = l.split(' &')[0]
    if blk == 'bread' and l.startswith(r'\midrule'):
        out.append([row1, row2][k] if k < 2 else row1); k += 1
    out.append(l)
assert k == 2, k
s = '\n'.join(out)
# 평균 행 (Spring-Gaus) 다시: 장면 행에서 값 모음
def sgrows(text, ncol):
    rows = {}; blk = None
    for l in text.split('\n'):
        m = re.match(r'^([A-Za-z0-9평]+) &', l)
        if m: blk = m.group(1)
        if 'Spring-Gaus' in l and blk not in (None, '평균'):
            cells = [c.strip().rstrip('\\').strip() for c in l.split('&')[2:]]
            rows.setdefault(blk, []).append(cells)
    return rows
def avg_line(rowsets, fmts):
    vals = list(zip(*rowsets))
    cells = []
    for col, f in zip(vals, fmts):
        nums = []
        for c in col:
            c = re.sub(r'\$\^\{?\(\d\)\}?\$', '', c)
            try: nums.append(float(c))
            except ValueError: pass
        if not nums: cells.append('--'); continue
        v = S.mean(nums); c = f(v)
        cells.append(c + (f"$^{{({len(nums)})}}$" if len(nums) != len(col) else ''))
    return cells
parts = s.split(r'\bottomrule')
tabs = []
for ti, (fm, n) in enumerate([( [lambda v: '%.2f' % v, lambda v: '%.1f' % v, lambda v: '%.3f' % v, lambda v: '%.2f' % v, lambda v: ('%.1e' % v).replace('e-0', 'e-').replace('e+0', 'e').replace('e+', 'e'), lambda v: '%.2f' % v, lambda v: '%.2f' % v], 7),
                               ([lambda v: '%.2f' % v, lambda v: '%.1f' % v, lambda v: '%.1f' % v, lambda v: '%.3f' % v, lambda v: '%.2f' % v, lambda v: '%.2f' % v, lambda v: '%.3f' % v, lambda v: '%.3f' % v], 8)]):
    p = parts[ti]
    rs = sgrows(p, n)
    flat = [r[0] for b, r in rs.items()]
    assert len(flat) == 4, (ti, rs.keys())
    cells = avg_line(flat, fm)
    newl = " & Spring-Gaus & " + " & ".join(cells) + "\\\\"
    lines = p.split('\n'); inavg = False
    for i, l in enumerate(lines):
        if l.startswith('평균'): inavg = True
        if inavg and l.startswith(' & Spring-Gaus'):
            lines[i] = newl; break
    else:
        raise SystemExit('평균 행 없음')
    parts[ti] = '\n'.join(lines)
s = r'\bottomrule'.join(parts)
fn = r"$^\ast$ Spring-Gaus bread 의 시간은 RTX 6000 Ada 단독. "
s = s.replace(r"{\footnotesize $^\dagger$ 재질 확장 구현", r"{\footnotesize " + fn + r"$^\dagger$ 재질 확장 구현", 1)
open(T, 'w').write(s)
# --- phys.tex · vis.tex ---
P = f"{WR}/pareto_final_1115_phys.tex"; s = open(P).read()
a = r" & Spring-Gaus & -- & \multicolumn{10}{c}{final\_blx} \\"
assert s.count(a) == 1
s = s.replace(a, f" & Spring-Gaus$^\\ast$ & {N} & -- & -- & -- & -- & -- & -- & -- & -- & -- & {jit:.2f} \\\\")
open(P, 'w').write(s)
V = f"{WR}/pareto_final_1115_vis.tex"; s = open(V).read()
a = r" & Spring-Gaus & -- & \multicolumn{8}{c}{final\_blx} \\"
assert s.count(a) == 1
s = s.replace(a, f" & Spring-Gaus$^\\ast$ & {N} & {vi['niqe']:.2f} & {vi['brisque']:.1f} & {vi['musiq']:.1f} & {vi['clipiqa']:.3f} & {1e3 * vi['warp_err']:.2f} & {vi['psnr']:.2f} & {vi['ssim']:.3f} & {vi['lpips']:.3f} \\\\")
open(V, 'w').write(s)
# --- pareto_blx.json ---
J = '/tmp/blx/res/pareto_blx.json'
o = [q for q in json.load(open(J)) if (q['method'], q['scene']) != ('springgaus', 'bread')]
o.append(dict(method='springgaus', scene='bread', N=N, setting='', reps=g['reps'], ok=g['ok'], err=err, err_all=g['err_all'], ms=ms, ms_all=g['ms_all'], div=g['div'], others=[0],
              E=None, note='재질 확장; 시간 RTX 6000 Ada 단독', src='final_pa/blx/SGB', ext='material'))
json.dump(o, open(J, 'w'), ensure_ascii=False, indent=1)
print('ok', N, err, ms, jit, vi['psnr'])
