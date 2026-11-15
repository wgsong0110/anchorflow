"""절 CM181: GF 수박 변형 실행 (exe/gf_mpm.py 새 플래그) 표. R2 FUSE 를 읽기만 함 (로컬에 안 받음).
사용: python exe/wmfrac_an.py NAME [NAME ...]   (R2 lbours/wmfrac/gfv/NAME/{energy.jsonl, sv/full_*.npz})
조각 수 = 부분표본 (간격 4) 입자를 처음 최근접 거리 중앙값 x 1.5 반경 그래프의 연결 성분, 1 % 넘는 것 + 가장 큰 조각 몫 (exe/measure_fragments.py 와 같은 방식).
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R = os.environ.get("WMR", os.path.expanduser("~/r2/result/anchorflow/lbours/wmfrac/gfv"))


def pieces(x0, x, al, stride=4, mult=1.5):
    i = np.arange(0, len(x), stride)
    ok = al[i] == 1
    ok &= np.isfinite(x[i]).all(1)
    t0 = cKDTree(x0[i][ok]); d, _ = t0.query(x0[i][ok], k=2); r = mult * np.median(d[:, 1])
    p = x[i][ok]
    t = cKDTree(p); pr = t.query_pairs(r, output_type="ndarray")
    n = len(p)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    sz = np.sort(np.bincount(lab))[::-1] / n
    return k, int((sz > 0.01).sum()), float(sz[0]), r


def rows(name):
    return [json.loads(l) for l in open(f"{R}/{name}/energy.jsonl")]


if __name__ == "__main__":
    print("| 판 | E_tot f3 | f4 | f5 | f6 | 최대 (프레임) | 탄성 f4 | 탄성 최대 | 손상 f5 · f8 · f100 % | 조각 f100 (1% 넘는 · 가장 큰 몫) | 질량중심 f100 | 이탈 입자 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in sys.argv[1:]:
        try:
            rw = rows(nm)
        except Exception as e:
            print(f"| {nm} | (없음 {e}) |"); continue
        fr = {int(r["f"]): r for r in rw if abs(r["f"] - round(r["f"])) < 1e-9}
        et = {f: r["KE"] + r["PE"] + r["Eel"] for f, r in fr.items()}
        fmax = max(et, key=et.get)
        elm = max(r["Eel"] for r in fr.values())
        g = lambda f, k: (f"{100 * fr[f][k]:.2f}" if f in fr else "-")
        pc = "-"
        fn = f"{R}/{nm}/sv/full_0100.npz"
        f0 = f"{R}/{nm}/sv/sub_0000.npz"
        last = max(fr)
        for ff in (100, last):
            fn = f"{R}/{nm}/sv/full_{ff:04d}.npz"
            if os.path.exists(fn):
                break
        if os.path.exists(fn) and os.path.exists(f0):
            a = np.load(fn); b = np.load(f0)
            pc = []
            for mm in (1.5, 3.0):
                k, k1, big, rr = pieces(b["x"], a["x"][::8], a["al"][::8], stride=1, mult=mm)
                pc.append(f"x{mm}: {k1} ({k} 전체, 최대 {100 * big:.1f} %)")
            pc = " / ".join(pc)
        n0 = fr[0]["n"]; nl = fr[last]["n"]
        print(f"| {nm} | {et.get(3, float('nan')):.2f} | {et.get(4, float('nan')):.2f} | {et.get(5, float('nan')):.2f} | {et.get(6, float('nan')):.2f} | {et[fmax]:.2f} (f{fmax}) | "
              f"{fr[4]['Eel'] if 4 in fr else float('nan'):.2f} | {elm:.2f} | {g(5,'dmg0')} · {g(8,'dmg0')} · {g(last,'dmg0')} (f{last}) | {pc} | {fr[last]['zc']:.3f} | {n0 - nl} |")
