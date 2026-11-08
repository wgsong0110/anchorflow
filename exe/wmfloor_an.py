"""절 CM64 묶음 wmfloor 진단 (로컬, 결과 불변 기록만 읽음): 수박 이탈·바닥 뚫음 입자의 '튐 프레임' (D2 고리 버퍼에서 입자 속도가 한 프레임에 1 m/s 넘게 는 첫 프레임) 을
그 프레임의 걸음 기록 (hc.jsonl: 충돌·H4 뒤집힘·H5·몸 후퇴 오일러 적용), 장벽 기록 (e5.jsonl), 노드 (질량 비·고정·후퇴 오일러 집합·시작/끝 속도) 와 묶음.
사용: python exe/wmfloor_an.py <판 디렉토리...>   (출력: 판별 표 + 튐 사건 표, 마크다운)
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

FLOORZ = 0.02


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError:
        return []


def frame_of_t(t, h=0.03):
    return int(round(t / h + 1e-6)) + 1                                                  # 걸음 시작 t 가 [ (f-1) h, f h ) 에 있으면 프레임 f (1 부터)


def an_run(d):
    hc = jl(os.path.join(d, "hc.jsonl"))
    e5 = {q["f"]: q for q in jl(os.path.join(d, "e5.jsonl")) if "f" in q}
    e2 = {q["f"]: q for q in jl(os.path.join(d, "e2.jsonl")) if "f" in q}
    H = defaultdict(lambda: dict(nstep=0, new=0, ncol=0, nflip=0, nf5=0, napp=0))
    for q in hc:
        f = int(np.floor(q["t"] / 0.03 + 1e-6)) + 1
        for k in ("nstep", "new", "ncol", "nflip", "nf5", "napp"):
            H[f][k] += int(bool(q.get(k, 0))) if k in ("napp",) else q.get(k, 0)
    ev = []
    for z in sorted(glob.glob(os.path.join(d, "e2_f*.npz"))):
        Z = np.load(z)
        G = len([k for k in Z.files if k.endswith("_f") and k.startswith("g")])
        idx = Z["idx"]
        for j in range(len(idx)):
            bump = None
            for g in range(G):
                p = f"g{g}_"
                dv = np.linalg.norm(Z[p + "v1"][j]) - np.linalg.norm(Z[p + "v0"][j])
                if dv > 1.0:
                    bump = g
                    break
            zmin_end = min(float(Z[f"g{g}_x1"][j][2]) for g in range(G))
            r = dict(run=os.path.basename(d), esc=int(Z["f"]), pid=int(idx[j]), dist=float(Z["d"][j]), zend=float(Z[f"g{G-1}_x1"][j][2]), zmin=zmin_end)
            r["below"] = r["zend"] < FLOORZ - float(Z["cell"])
            if bump is None:
                r["bump"] = None
                ev.append(r)
                continue
            p = f"g{bump}_"
            f = int(Z[p + "f"])
            w = Z[p + "w"][j]
            r.update(bump=f, z0=float(Z[p + "x0"][j][2]), v0=Z[p + "v0"][j].tolist(), v1=Z[p + "v1"][j].tolist(), lj1=float(Z[p + "lj1"][j]), cmr=float(Z[p + "cmr"][j]),
                     wpin=float(Z[p + "wpin"][j]))
            if p + "mn" in Z.files:
                mn, pin, vs, vk = Z[p + "mn"][j], Z[p + "pin"][j], Z[p + "vs"][j], Z[p + "vk"][j]
                ulm = Z[p + "ulm"][j] if p + "ulm" in Z.files else np.zeros(27, int)
                big = w > 1e-3
                sp = np.where(big, np.linalg.norm(vk, axis=1), 0)
                k = int(np.argmax(sp))
                contrib = w[:, None] * vk                                                    # G2P 몫
                dvp = np.array(r["v1"]) - np.array(r["v0"])
                # 입자 속도 늘어난 몫을 노드별로: w (vk - vs) 의 dv 방향 성분
                dn = dvp / max(np.linalg.norm(dvp), 1e-12)
                share = (w[:, None] * (vk - vs)) @ dn
                kk = int(np.argmax(share))
                r.update(kf=dict(vk=float(sp[k]), vkz=float(vk[k][2]), vs=float(np.linalg.norm(vs[k])), mn=float(mn[k]), pin=bool(pin[k]), ulm=int(ulm[k]), w=float(w[k])),
                         ks=dict(share=float(share[kk]), tot=float(share.sum()), vk=float(np.linalg.norm(vk[kk])), mn=float(mn[kk]), pin=bool(pin[kk]), ulm=int(ulm[kk]), w=float(w[kk])),
                         mnmin=float(mn[big].min()), ulmw=float((w * (ulm != 0)).sum()), Ushare=float(np.linalg.norm(Z[p + "U"][j][kk])))
            r["hc"] = dict(H[f])
            q5 = e5.get(f, {})
            r["e5"] = dict(inact=q5.get("inact"), act_end_le0=q5.get("act_end_le0"), bar_pos_act_le0=q5.get("bar_pos_act_le0"), nv_act=q5.get("nv_act"), nbelow=q5.get("nbelow"), kcnt=q5.get("kcnt"))
            q2 = e2.get(f, {})
            r["step"] = q2.get("step")
            ev.append(r)
    return ev, H


def main():
    runs = sys.argv[1:]
    allev = []
    for d in runs:
        ev, H = an_run(d)
        allev += ev
    # 사건 표: (판, 튐 프레임) 으로 묶음
    grp = defaultdict(list)
    for r in allev:
        grp[(r["run"], r.get("bump"))].append(r)
    print("| 판 | 튐 프레임 | 입자 수 (바닥 밑 끝) | 입자 z 시작 m | 입자 속도 증가 m/s | 최대 몫 노드: 질량 비 · 고정 · 후퇴 오일러 집합 · 끝 속도 m/s · 시작 속도 | 가장 빠른 노드: 질량 비 · 끝 속도 | 입자 27 노드 최소 질량 비 | 후퇴 오일러 집합 무게 | 걸음: 덜 풀림 · ‖g‖²/F² | hc: 충돌 · H4 뒤집힘 · H5 · 몸 BE 적용 | e5: 비활성 · 끝 ≤0 · 시작 ≤0 · A2 옮김 · 바닥 밑 입자 |")
    print("|" + "---|" * 12)
    for (run, f), L in sorted(grp.items(), key=lambda a: (a[0][0], a[0][1] or 0)):
        if f is None:
            print(f"| {run} | 없음 | {len(L)} ({sum(r['below'] for r in L)}) | | | | | | | | | |")
            continue
        r = max(L, key=lambda q: np.linalg.norm(np.array(q["v1"])) - np.linalg.norm(np.array(q["v0"])))
        dv = [np.linalg.norm(np.array(q["v1"])) - np.linalg.norm(np.array(q["v0"])) for q in L]
        ks, kf = r.get("ks", {}), r.get("kf", {})
        st = r.get("step") or {}
        hcq, e5q = r["hc"], r["e5"]
        print(f"| {run} | f{f} | {len(L)} ({sum(q['below'] for q in L)}) | {min(q['z0'] for q in L):.3f}~{max(q['z0'] for q in L):.3f} | {min(dv):.1f}~{max(dv):.1f} | "
              f"{ks.get('mn', float('nan')):.1e} · {'고정' if ks.get('pin') else '풀림'} · {ks.get('ulm')} · {ks.get('vk', float('nan')):.1f} · {np.nan if not kf else kf['vs']:.2f} | "
              f"{kf.get('mn', float('nan')):.1e} · {kf.get('vk', float('nan')):.1f} | {r.get('mnmin', float('nan')):.1e} | {r.get('ulmw', float('nan')):.2f} | "
              f"{st.get('fail')} · {[round(g, 2) for g in st.get('g2', [])]} | {hcq['ncol']} · {hcq['nflip']} · {hcq['nf5']} · {hcq['napp']} | "
              f"{e5q['inact']} · {e5q['act_end_le0']} · {e5q['bar_pos_act_le0']} · {e5q['nv_act']} · {e5q['nbelow']} |")
    json.dump(allev, open(os.environ.get("WMF_JSON", "/dev/null"), "w"), default=float)


if __name__ == "__main__":
    main()
