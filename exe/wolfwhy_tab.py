"""절 CM25 표: python exe/wolfwhy_tab.py <묶음 출력 디렉토리> <정답 wolf.pt>  -> 표 (stdout), 곡선 원자료 <출력>/wolfwhy_curves.json
판 디렉토리: base0 (기본 판 그대로, 진단 --nrec 없음 -- 결과 불변 대조), base (기본 판 + --nrec), a_noy0eb, b_nostep, e_n8p20, A_stick.
높이 = 바닥 법선 (z), 질량 중심 z 는 질량 가중 (실행기 --hgt hp; 정답 hg), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (덤프에서; 절 CM11 정의)."""
import json
import os
import sys

import numpy as np
import torch

O, GTP = sys.argv[1:3]
REF_K1 = "W_K1 (절 CM23 k1new)"
RUNS = [("base", "기본 판 (장벽 + --y0eb + 보간 교체, K 1, 2x5) + --nrec"),
        ("a_noy0eb", "(a) --y0eb 끔 (시작점 = U 에서 ŷ 쪽 실현 가능 최대 2^-k)"),
        ("b_nostep", "(b) 장벽 실현 가능 보폭 자름 끔 (--bar_nostep)"),
        ("e_n8p20", "(e) 반복 상한만 8x20"),
        ("A_stick", "(A) 가우스 점 고착 바닥 (장벽 없음), K 1")]


def jl(d):
    p = os.path.join(O, d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sm(d):
    p = os.path.join(O, d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def spread(x):
    x = x[np.isfinite(x).all(1)]
    if not len(x):
        return float("nan")
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


def f3(v, n=3):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def g2(v):
    return "-" if v is None else f"{v:.2g}"


G = torch.load(GTP, map_location="cpu", weights_only=False)["x"]
gtw = [spread(G[i].numpy()) for i in range(1, G.shape[0])]
cur = {"gt": {"w": gtw}}
info = {}
for d, nm in [("base0", "기본 판 진단 없음")] + RUNS:
    J, S = jl(d), sm(d)
    P = None
    pp = os.path.join(O, d, "pred.pt")
    if os.path.exists(pp):
        try:
            P = torch.load(pp, map_location="cpu", weights_only=False).get("pred")
        except Exception:
            P = None
    c = dict(err=[r.get("err") for r in J], com=[(r.get("hp") or [None])[0] for r in J], zmax=[(r.get("hp") or [None, None])[1] for r in J],
             w=[(spread(P[r["t"]].numpy()) if P is not None and r["t"] < P.shape[0] else None) for r in J], xh=[r.get("xh") for r in J],
             neng=[(r.get("bc") or {}).get("neng") for r in J], it=[r.get("it") for r in J], nr=[r.get("nr") for r in J], y0k=[r.get("y0k") for r in J], bk=[r.get("bk") for r in J])
    if J and "hg" in J[0]:
        cur["gt"].update(com=[r["hg"][0] for r in J], zmax=[r["hg"][1] for r in J])
    cur[d] = c
    info[d] = (nm, S, J)
json.dump(cur, open(os.path.join(O, "wolfwhy_curves.json"), "w"), ensure_ascii=False)
gt = cur["gt"]


def at(a, f):
    return a[f - 1] if a and len(a) >= f else None


print("# 절 CM25 wolf 가 장벽 바닥에서 무너지지 않는 원인 -- 묶음 wolfwhy (exe/stab_wolfwhy.sh)")
print()
print("모두 결정적 --det, 50 프레임, K 1 (새 기본 판 = 절 CM23 W_K1 명령: 장벽 --fl_bar --fl_bst 1 --y0eb + --fp_interp, HHT ρ 0.8, 뉴턴 2 x PCG 5, εN 1e-2 · εP 1e-1). 프레임 f = fem.jsonl t+1.")
print("재질 판 (모래, MatRT) 의 뉴턴 보폭을 자를 수 있는 것은 장벽 실현 가능 보폭 (rt_bar_kfe) 하나뿐 -- 에너지 선탐색·뒤집힘 필터는 이 경로에 없음 (fem_an_rt_sim: 재질 판 a.ls = 0; MatRT 요소 ao_elem 에 필터 없음). 그래서 대조 (c) 뒤집힘 필터 끔 = 해당 없음, (d) = (a).")
print()
print("## 0. 결과 불변 (진단 --nrec 켬·끔 프레임 해시)")
b0, b1 = cur.get("base0", {}).get("xh") or [], cur.get("base", {}).get("xh") or []
same = sum(1 for p, q in zip(b0, b1) if p == q and p)
print(f"- base0 (진단 없음) 대 base (--nrec): 프레임 해시 같음 {same} / {len(b0)} (끝 {b0[-1] if b0 else '-'} · {b1[-1] if b1 else '-'}); 절 CM23 W_K1 끝 해시 (R2 k1new/W_K1 마지막 xh) 와는 go 로그 참조")
s_ = info.get("b_nostep", (None, None, None))[2]
if s_ and b1:
    xb = [r.get("xh") for r in s_]
    print(f"- b_nostep 대 base: 프레임 해시 같음 {sum(1 for p, q in zip(b1, xb) if p == q and p)} / {len(xb)}")
print()
print("## 1. 판별 요약 (mm; 질량중심 z·최대 z·퍼짐 폭은 f30 · f50; 정답 질량중심 z 572.6 · 533.3, 최대 z 1112.3 · 660.8)")
print()
print("| 판 | 결과 · 평균 오차 % | 오차 f30 · f50 % | 질량중심 z f30 · f50 | 최대 z f30 · f50 | 퍼짐 폭 f30 · f50 | 끝 해시 | N·P /프레임 | f15~35 시작점 k 중앙 (실현 가능 k0 중앙) | f15~35 걸음 끝 ‖g‖/F 중앙 (최대) | 보폭 < 1 반복 수 | 장벽 걸린 점 합 · 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
print(f"| 정답 | - | - | {f3(at(gt.get('com'), 30) and 1e3 * at(gt['com'], 30), 1)} · {f3(at(gt.get('com'), 50) and 1e3 * at(gt['com'], 50), 1)} | {f3(at(gt.get('zmax'), 30) and 1e3 * at(gt['zmax'], 30), 1)} · {f3(at(gt.get('zmax'), 50) and 1e3 * at(gt['zmax'], 50), 1)} | {f3(1e3 * gtw[29], 1)} · {f3(1e3 * gtw[49], 1)} | | | | | | |")
for d, nm in [("base0", "기본 판 (진단 없음)")] + RUNS:
    c = cur.get(d)
    _, S, J = info.get(d, (None, None, None))
    if not J:
        print(f"| {nm} | 없음 | | | | | | | | | | |")
        continue
    dv = S.get("diverged") if S else None
    em = (S.get("err_mean") if S and S.get("err_mean") is not None else (S.get("err_mean_done") if S else None))
    res = (f"발산 f{dv}" if dv else f"완주 {len(J)}") + (f" · {100 * em:.2f}" if em is not None else "")
    nN = np.mean([r["it"]["nN"] for r in J if r.get("it")]); nP = np.mean([r["it"]["nP"] for r in J if r.get("it")])
    nr = [q for q in c["nr"][14:35] if q]
    y0 = [s["y0k"] for q in nr for s in q["steps"] if s.get("y0k") is not None]
    k0 = [s["k0f"] for q in nr for s in q["steps"] if s.get("k0f") is not None]
    ge = [s["gFend"] for q in nr for s in q["steps"] if s.get("gFend") is not None]
    nst = sum(1 for q in c["nr"] if q for s in q["steps"] for r in s["it"] if r.get("ran") and r.get("st") is not None and r["st"] < 1)
    ne = [q for q in c["neng"] if q is not None]
    hsh = (S or {}).get("x_sha1", "")[:10] if S else ""
    def mm(a, f):
        v = at(a, f)
        return f3(1e3 * v, 1) if v is not None else "-"
    print(f"| {nm} | {res} | {f3(at(c['err'], 30) and 100 * at(c['err'], 30), 2)} · {f3(at(c['err'], 50) and 100 * at(c['err'], 50), 2)} | {mm(c['com'], 30)} · {mm(c['com'], 50)} | {mm(c['zmax'], 30)} · {mm(c['zmax'], 50)} | {mm(c['w'], 30)} · {mm(c['w'], 50)} | {hsh} | {nN:.1f}·{nP:.1f} | "
          f"{(f'{np.median(y0):.0f} ({np.median(k0):.0f})' if y0 else '-')} | {(f'{np.median(ge):.3g} ({max(ge):.3g})' if ge else '-')} | {nst if nr else '-'} | {(f'{sum(ne)} · {max(ne)}' if ne else '-')} |")
print()
for d, nm in RUNS:
    c = cur.get(d)
    if not c or not any(c["nr"]):
        continue
    print(f"## 2. 프레임별 -- {nm} [{d}]")
    print()
    print("시작점 k = 뉴턴 시작점 y0 = U + 2^-k (ŷ − U) 의 k (0 = 관성 예측 ŷ 그대로, 클수록 걸음 시작 위치 U); k0 = 장벽 실현 가능 k (--y0eb 전), k = 최종 (--y0eb 장벽 에너지 규칙 뒤).")
    print("‖g‖/F = 뉴턴 반복 시작 잔차 / 걸음 첫 반복의 탄성·접촉 힘 몫 (허용치 εN 1e-2 와 같은 척도), 끝 = 마지막 뉴턴 뒤 y 에서 다시 잰 값. PCG = 반복 수 (상한 도달 *). 보폭 = 2^-bk (장벽 실현 가능 자름; 이 판의 유일한 자름 주체).")
    print("ΣG 장벽 = 걸음 끝 장벽 노드 기울기 법선 합 (N, 장벽이 받치는 힘 크기 -- 몸 무게 참고).")
    print()
    print("| f | 정답 질량중심 z | 우리 | 오차 % | k0 · k | 반복 1 ‖g‖/F · PCG · 보폭 | 반복 2 ‖g‖/F · PCG · 보폭 | 끝 ‖g‖/F (허용치 대비 배) | 장벽 걸린 점 | ΣG 장벽 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in range(1, len(c["err"]) + 1):
        q = c["nr"][f - 1]
        if not q:
            continue
        s = q["steps"][0]
        def itc(i):
            if i >= len(s["it"]):
                return "-"
            r = s["it"][i]
            if not r["ran"]:
                return f"{g2(r['gF'])} 멈춤" if r["stop"] else "-"
            return f"{g2(r['gF'])} · {r.get('nP')}{'*' if r.get('cap') else ''} · {('2^-' + str(r['bk'])) if r.get('bk') else f3(r.get('st'), 3)}"
        extra = ""
        if len(s["it"]) > 2:
            nst = sum(1 for r in s["it"] if r["ran"])
            extra = f" (+{nst - 2} 반복)" if nst > 2 else ""
        gE = s.get("gFend")
        print(f"| {f} | {f3(at(gt.get('com'), f), 3)} | {f3(at(c['com'], f), 3)} | {f3(100 * c['err'][f - 1], 2)} | {s.get('k0f', '-')} · {s.get('y0k', '-')} | {itc(0)} | {itc(1)}{extra} | "
              f"{g2(gE)} ({g2(gE / q['tolN']) if gE is not None and q['tolN'] > 0 else '-'}) | {at(c['neng'], f) if at(c['neng'], f) is not None else '-'} | {g2(s.get('Gbar'))} |")
    print()
