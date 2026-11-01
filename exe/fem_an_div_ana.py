"""절 AN 발산 진단 분석: <루트>/<궤적>_<판>/fem.jsonl (--diag) -> 궤적마다 (1) 이상 표시가 처음 나온 프레임 (g·δ ≥ 0 = 하강 방향 아님, r·z ≤ 0 = 전처리 양이 아님,
보폭 0 = 받아들인 걸음 없음, 받아들인 걸음의 에너지 증가), (2) 발산 직전 5 프레임의 뉴턴 반복별 값, (3) 프레임별 고착 불일치 (걸음 끝 몫 > 0 과 시작 몫 > 0 이 다른 노드 수)."""
import glob
import json
import math
import os
import sys

R = sys.argv[1]
for d in sorted(glob.glob(os.path.join(R, "*", "fem.jsonl"))):
    nm = os.path.basename(os.path.dirname(d))
    rs = [json.loads(l) for l in open(d)]
    flags = []
    for r in rs:
        dg = r.get("diag")
        if not dg:
            continue
        f = []
        for i, it in enumerate(dg["it"]):
            st = it["st"]
            Esel = it["E0"] if st == 0 else it["Ec"][[0.25, 0.5, 1.0].index(st)] if st in (0.25, 0.5, 1.0) else float("nan")
            if not (it["gd"] < 0):
                f.append(f"뉴턴{i + 1} g·δ={it['gd']:.2e}")
            if not (it["rz0"] > 0):
                f.append(f"뉴턴{i + 1} r·Mr={it['rz0']:.2e}")
            if st == 0:
                f.append(f"뉴턴{i + 1} 보폭0")
            if math.isfinite(Esel) and Esel > it["E0"]:
                f.append(f"뉴턴{i + 1} E증가 {Esel - it['E0']:.2e}")
        flags.append((r["t"], r["err"], f, dg["stick_flip"], dg["stick_n0"]))
    first = next((x for x in flags if x[2]), None)
    last = rs[-1]
    print(f"## {nm}: 프레임 {len(rs)} (끝 오차 {100 * last['err']:.2f}%), 이상 표시 처음 프레임 {first[0] + 1 if first else '없음'} {first[2][:4] if first else ''}")
    nflag = sum(1 for x in flags if x[2])
    print(f"  이상 표시 프레임 수 {nflag}/{len(flags)}; 고착 불일치 노드 (프레임 평균) {sum(x[3] for x in flags) / max(len(flags), 1):.1f} (고착 노드 평균 {sum(x[4] for x in flags) / max(len(flags), 1):.1f})")
    for r in rs[-5:]:
        dg = r.get("diag") or {}
        its = "; ".join(f"|g|²={it['g2']:.2e} r·Mr={it['rz0']:.2e}->{it['rz1']:.2e} g·δ={it['gd']:.2e} E0={it['E0']:.4e} E(¼,½,1)=({it['Ec'][0]:.4e},{it['Ec'][1]:.4e},{it['Ec'][2]:.4e}) 보폭={it['st']}"
                        for it in dg.get("it", []))
        print(f"  프레임 {r['t'] + 1} 오차 {100 * r['err']:.2f}% 고착 불일치 {dg.get('stick_flip')}/{dg.get('stick_n0')} :: {its}")
