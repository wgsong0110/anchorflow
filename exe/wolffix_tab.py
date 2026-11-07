"""절 CM27 표: python exe/wolffix_tab.py <묶음 출력 디렉토리> <정답 wolf.pt> [판 목록 쉼표]  -> 마크다운 (표준 출력), 곡선 원자료 <출력>/wolffix_curves.json
판 디렉토리 = 판 이름 (결정적 판정 판), z_<판>_a·b (단독 시간). 높이 = 바닥 법선 z (m -> mm), 질량중심·최대 z = 실행기 --hgt (hp; 정답 hg),
퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (덤프; 절 CM11 정의). 속도 반전 = --prec dv.rest (k_구속 ≥ ½ 노드, 질량 가중 법선 속도) 가 v0 < −0.02 m/s 에서 v1 > 0 으로 (아래 -> 위)."""
import json
import os
import sys

import numpy as np
import torch

O, GTP = sys.argv[1:3]
NAMES = {"base": "기본 판 (장벽 바닥 + 보간 교체)", "a_ube": "(a) 덜 풀린 걸음 후퇴 오일러 갱신 (--ue be, 판정 2)", "b_ua0": "(b) 덜 풀린 걸음 가속도 기억 0 (--ue a0, 판정 2)",
         "c_loc": "(c) 국소 시작점 섞기 (--y0loc 3)", "d_locube": "(d) (a)+(c)", "e_phi": "(e) 구속 이동도 φ (--pbc_phi)", "e_phiube": "(e)+(a)",
         "e_philoc": "(e)+(c)", "e_all": "(e)+(a)+(c)", "f_ks": "적응 쪼개기 --ksplit 3", "f_ksloc": "--ksplit 3 + (c)", "f_ksphi": "--ksplit 3 + (e)", "f_ksall": "--ksplit 3 + (c)+(e)", "g_dks": "--ksplit 3 (판정 2) + (d)"}
RUNS = sys.argv[3].split(",") if len(sys.argv) > 3 else list(NAMES)
REF_END = "6d716537dd"


def rd(p):
    out = []
    if os.path.exists(p):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def sm(p):
    return json.load(open(p)) if os.path.exists(p) else None


def spread(x):
    x = x[np.isfinite(x).all(1)]
    if not len(x):
        return float("nan")
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


def mm(v, n=1):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 1000:.{n}f}"


def fm(v, n=2):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def at(a, f):
    return a[f - 1] if a and len(a) >= f else None


G = torch.load(GTP, map_location="cpu", weights_only=False)["x"]
gtw = [spread(G[i].numpy()) for i in range(1, G.shape[0])]
cur = {"gt": {"w": gtw}}
rows = {}
for n in RUNS:
    J = rd(os.path.join(O, n, "fem.jsonl"))
    if not J:
        continue
    S = sm(os.path.join(O, n, "summary.json"))
    PR = rd(os.path.join(O, n, "prec.jsonl"))
    P = None
    pp = os.path.join(O, n, "pred.pt")
    if os.path.exists(pp):
        try:
            P = torch.load(pp, map_location="cpu", weights_only=False).get("pred")
        except Exception:
            P = None
    com = [(r.get("hp") or [None])[0] for r in J]
    zmx = [(r.get("hp") or [None, None])[1] for r in J]
    w = [(spread(P[r["t"]].numpy()) if P is not None and r["t"] < P.shape[0] else None) for r in J]
    if all("hg" in r for r in J) and len(J) > len(cur["gt"].get("com") or []):
        cur["gt"].update(com=[r["hg"][0] for r in J], zmax=[r["hg"][1] for r in J])
    uf = [sum((r.get("uf") or {}).get("fail") or []) for r in J]
    neng = [(r.get("bc") or {}).get("neng") for r in J]
    rev, revf = 0, []
    vr = {}
    drop = {}
    for r in PR:
        q = (r.get("dv") or {}).get("rest")
        if q and "v0" in q:
            vr[r["f"]] = (q["v0"], q["v1"])
            if q["v0"] < -0.02 and q["v1"] > 0:
                rev += 1; revf.append(r["f"])
        rl = (r.get("p") or {}).get("rel") or {}
        if "drop" in rl:
            drop[r["f"]] = (rl.get("drop"), rl.get("drop_gt"))
    yl = [r.get("yl") for r in J]
    ylfb = sum(1 for y in yl if y for st in y if st and len(st) > 4 and st[-1] > 0)
    yln = [st[0] for y in yl if y for st in y if st]
    yl0 = [st[2] for y in yl if y for st in y if st]
    xh = [r.get("xh") for r in J]
    zt = [sm(os.path.join(O, f"z_{n}_{r}", "summary.json")) for r in "ab"]
    cur[n] = dict(err=[r.get("err") for r in J], com=com, zmax=zmx, w=w, uf=uf, neng=neng, vrest=vr, drop=drop, yl=yl)
    rows[n] = dict(S=S, com=com, zmx=zmx, w=w, uf=uf, neng=neng, rev=rev, revf=revf, drop=drop, ylfb=ylfb, yln=yln, yl0=yl0, xh=xh, zt=zt, nprerr=sum(1 for r in PR if "err" in r))
json.dump(cur, open(os.path.join(O, "wolffix_curves.json"), "w"), ensure_ascii=False)
gt = cur["gt"]
print("# 절 CM27 wolf 무너짐 고치기 -- 묶음 wolffix (exe/stab_wolffix.sh)\n")
print("모두 결정적 --det, 50 프레임, K 1, 장벽 바닥 (--fl_bar --fl_bst 1 --y0eb) + --fp_interp, HHT ρ 0.8, 뉴턴 2 x PCG 5, εN 1e-2 · εP 1e-1 (절 CM25 base 명령) + 판 플래그. "
      "진단 (결과 불변): --hgt --fhash --bar_cnt --nrec --prec 10:50, 판정 커널 (--ue 없는 판은 --ufrec --uf_res). 단독 시간 = 비결정·진단 없음, 카드 하나에 차례로 두 번 (프레임당 재생 GPU ms 평균/최대, 첫 2 프레임 뺌).\n")
b = rows.get("base")
if b:
    href = [r.get("xh") for r in rd(os.path.join(O, "ref_base_fem.jsonl"))]
    same = sum(1 for x, y in zip(href, b["xh"]) if x and x == y)
    print(f"끄면 비트 같음: base (새 코드, 새 플래그 없음) 프레임 해시 절 CM25 base 와 같음 {same} / {len(b['xh'])} (끝 {b['xh'][-1] if b['xh'] else '-'}, 절 CM25 {REF_END})\n")
print(f"정답 (mm): 질량중심 z f15 · f22 · f30 · f50 = {mm(at(gt.get('com'), 15))} · {mm(at(gt.get('com'), 22))} · {mm(at(gt.get('com'), 30))} · {mm(at(gt.get('com'), 50))}, "
      f"최대 z f30 · f50 = {mm(at(gt.get('zmax'), 30))} · {mm(at(gt.get('zmax'), 50))}, 퍼짐 폭 f30 · f50 = {mm(at(gtw, 30))} · {mm(at(gtw, 50))}\n")
print("## 1. 판별 결과 (mm)\n")
print("| 판 | 오차 % | 발산 | 질량중심 z f15 · f22 · f30 · f50 | 최대 z f30 · f50 | 퍼짐 폭 f30 · f50 | 덜 풀린 걸음 (프레임 합) | 속도 반전 프레임 수 (f10~50: 프레임) | 장벽 걸린 표본점 합 · 최대 | 풀린 입자 내려감 f22 · f30 (정답) | 단독 ms a · b (평균/최대) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n in RUNS:
    r = rows.get(n)
    if not r:
        print(f"| {NAMES.get(n, n)} | 없음 |||||||||| ")
        continue
    S = r["S"] or {}
    ng = [q for q in r["neng"] if q is not None]
    d22, d30 = r["drop"].get(22, (None, None)), r["drop"].get(30, (None, None))
    zt = " · ".join(f"{fm(z.get('ms_mean'))}/{fm(z.get('ms_max'))}" if z else "-" for z in r["zt"])
    print(f"| {NAMES.get(n, n)} | {fm((S.get('err_mean') or float('nan')) * 100)} | {S.get('diverged') or '없음'} | {' · '.join(mm(at(r['com'], f)) for f in (15, 22, 30, 50))} | "
          f"{mm(at(r['zmx'], 30))} · {mm(at(r['zmx'], 50))} | {mm(at(r['w'], 30))} · {mm(at(r['w'], 50))} | {sum(r['uf'])} | {r['rev']} ({', '.join(map(str, r['revf'][:12]))}) | "
          f"{sum(ng) if ng else '-'} · {max(ng) if ng else '-'} | {mm(d22[0])} · {mm(d30[0])} ({mm(d22[1])} · {mm(d30[1])}) | {zt} |")
print()
print("## 2. 국소 시작점 (--y0loc) 진단: 걸음별 k > 0 노드 수 (중앙 · 최대), β 0 노드 수 최대, 전역 k 로 되돌린 걸음 수\n")
for n in RUNS:
    r = rows.get(n)
    if r and r["yln"]:
        print(f"- {NAMES.get(n, n)}: k > 0 노드 {int(np.median(r['yln']))} · {max(r['yln'])}, β 0 노드 최대 {max(r['yl0'])}, 되돌림 {r['ylfb']} 걸음")
print()
print("## 3. 프레임별 질량중심 z (mm; 10 프레임마다 + f15·f22)\n")
FR = [5, 10, 15, 20, 22, 25, 30, 35, 40, 45, 50]
print("| 판 | " + " | ".join(f"f{f}" for f in FR) + " |")
print("|---|" + "---|" * len(FR))
print("| 정답 | " + " | ".join(mm(at(gt.get("com"), f)) for f in FR) + " |")
for n in RUNS:
    r = rows.get(n)
    if r:
        print(f"| {NAMES.get(n, n)} | " + " | ".join(mm(at(r["com"], f)) for f in FR) + " |")
print()
print("## 4. 풀린 쪽 (k_구속 ≥ ½ 노드) 질량 가중 법선 속도 v0 -> v1 (m/s, f10~30)\n")
FV = list(range(10, 31))
print("| 판 | " + " | ".join(f"f{f}" for f in FV) + " |")
print("|---|" + "---|" * len(FV))
for n in RUNS:
    if n in cur:
        v = cur[n]["vrest"]
        print(f"| {n} | " + " | ".join((f"{v[f][0]:+.2f}→{v[f][1]:+.2f}" if f in v else "-") for f in FV) + " |")
print()
for n in RUNS:
    r = rows.get(n)
    if r and r["nprerr"]:
        print(f"- {n}: --prec 진단 실패 프레임 {r['nprerr']}")
