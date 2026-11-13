"""절 CM127 linwhy 집계 (로컬·인스턴스 공용, numpy): 판 디렉토리들의 lw.jsonl -> 표 (마크다운) + 잔차 위치 (lw/fFFF_iII.npz).
사용: python exe/linwhy_an.py <루트> <판...>"""
import json
import os
import sys

import numpy as np


def at(h, i):
    return h[min(i, len(h) - 1)] if h else float("nan")


def loc(z):
    """실제 PCG 끝 잔차 |r|² 몫: 확장 주인 노드 · 장벽 노드 · 꼭짓점 몫 ≥ 0.5 · 고정·고착 (pm ≠ 0) · 나머지, 그리고 높이 (바닥 0.48 위) 구간."""
    rf = np.linalg.norm(z["rfin"], axis=1) ** 2
    r0 = np.linalg.norm(z["r0"], axis=1) ** 2
    tot, tot0 = rf.sum(), r0.sum()
    ext = z["w8b"] >= 0
    own = (z["own"] > 0) if "own" in z else np.zeros_like(ext)
    bar = z["bKn"] > 0
    fap = z["fap"] if "fap" in z else np.full(len(rf), np.nan)
    ap = np.nan_to_num(fap) >= 0.5
    pmn = z["pm"] != 0
    IP, A = z["IP"], z["A"]
    nb1, nb2 = int(IP[7]), int(IP[8])
    nid = z["nid"].astype(np.int64)
    g = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]
    hz = (g @ A.T)[:, 2] - 0.48
    out = {}
    for nm, m in (("주인", own & ~ext), ("장벽", bar), ("꼭짓점≥.5", ap), ("pm≠0", pmn), ("외부", ext)):
        out[nm] = (float(m.mean()), float(rf[m].sum() / max(tot, 1e-300)), float(r0[m].sum() / max(tot0, 1e-300)))
    for lo, hi in ((-1, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)):
        m = (hz >= lo) & (hz < hi)
        out[f"z[{lo},{hi})"] = (float(m.mean()), float(rf[m].sum() / max(tot, 1e-300)), float(r0[m].sum() / max(tot0, 1e-300)))
    top = np.argsort(-rf)[:200]
    out["top200"] = dict(주인=float(own[top].mean()), 장벽=float(bar[top].mean()), 꼭짓점=float(np.nanmean(fap[top])), z중앙=float(np.median(hz[top])), 몫=float(rf[top].sum() / max(tot, 1e-300)))
    return out


def main():
    root, runs = sys.argv[1], sys.argv[2:]
    print("| 판 | f | it | 자유도 | 확인 Ap · Mz | 대칭 A · M | λÃ min · max (Mi min) | λA(확장 없음) min · max | λM min · max | PCG(M) 5·20·100·300 | 야코비 100·300 | CG 100 | FGMRES(M) 50·100 | ω λmax 층 0~3 | 새 ω: λM · PCG 5·100·300 | 리츠 M⁻¹Ã | 실제 PCG Prel 끝 |")
    print("|" + "---|" * 19)
    for n in runs:
        p = os.path.join(root, n, "lw.jsonl")
        if not os.path.exists(p):
            print(f"| {n} | 없음 |"); continue
        for line in open(p):
            o = json.loads(line)
            M, J, I, G, Mf = o["pcg_M"], o["pcg_J"], o["cg_I"], o["gm_M"], o["pcg_Mf"]
            rz = o.get("ritz_M"); rp = o.get("real_prel") or []
            print(f"| {n} | {o['f']} | {o['it']} | {o['nd']} | {o['chk_Ap']:.1e} · {o['chk_Mz']:.1e} | {o['symA']:.1e} · {o['symM']:.1e} | {o['lzA'][0]:.3g} · {o['lzA'][1]:.3g} ({o['Mi'][0]:.3g}) | "
                  f"{o['lzA0'][0]:.3g} · {o['lzA0'][1]:.3g} | {o['lzM'][0]:.3g} · {o['lzM'][1]:.3g} | {at(M,5):.2g} · {at(M,20):.2g} · {at(M,100):.2g} · {at(M,300):.2g} {o['pcg_M_bad']} | "
                  f"{at(J,100):.2g} · {at(J,300):.2g} | {at(I,100):.2g} | {at(G,50):.2g} · {at(G,100):.2g} | {' '.join(f'{q:.2f}' for q in o['omega_lmax'])} | "
                  f"{o['lzM_fresh'][0]:.3g}·{o['lzM_fresh'][1]:.3g} · {at(Mf,5):.2g} · {at(Mf,100):.2g} · {at(Mf,300):.2g} | {('%.3g~%.3g' % tuple(rz)) if rz else '-'} | {rp[-1] if rp else float('nan'):.2g} ({len(rp)}) |")
    print()
    for n in runs:
        d = os.path.join(root, n, "lw")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            z = np.load(os.path.join(d, fn))
            o = loc(z)
            print(f"{n} {fn}: (노드 몫, 끝 |r|² 몫, 처음 |r|² 몫)")
            for k_, v in o.items():
                print(f"   {k_}: {v if isinstance(v, dict) else tuple(round(q, 3) for q in v)}")


if __name__ == "__main__":
    main()
