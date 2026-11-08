"""절 CM58 묶음 hhtc 표: python exe/hhtc_tab.py <결과 디렉토리> -> markdown (stdout). 판 목록 = exe/hhtc_spec.txt.
바탕 = 공통 후보 F (E + W̄ + R2 + A2) + P1X (@g4 1e-1 @g4t 1 @g4x 1, 절 CM53). @hc M = 절 CM58 정의 (0 진단만 · 1 H1 충돌 걸음 몸 전체 후퇴 오일러 · 2 H2 새 접촉 노드 후퇴 오일러 · 3 H3 걸음 끝 접촉 노드 가속도 기억 0).
오차 = 끝난 프레임 평균 %, 60 fps 문턱 16.7 ms. 이탈 = 몸 상자 (0.1~99.9 백분위) 4 칸 밖 입자 (e1.jsonl), 이탈 판 = > 20 개. 튐 프레임 = 진단 D2 이탈 입자의 앞 5 프레임 중 입자 속도가 한 프레임에 1 m/s 넘게 는 프레임.
되튐 판 = 첫 접촉 프레임 (hc.jsonl 의 첫 충돌 걸음 프레임, 없으면 f4) 끝 몸 질량중심 vz > 0 (절 CM57.4 의 '같은 꼴 판' 판정 그대로). K 1 받음 = 그 프레임 시도 목록이 K 1 한 번 (무너짐 0) 뿐, 수렴 사건 = 그중 뉴턴이 허용치로 멈추고 (capN 0) 모든 보폭 1 인 판 (절 CM57.4 의 되튐 판 꼴).
수박 손상 = logJp > 0 몫 (정답 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %). 단독 = 계산 프로세스 0 인 카드 하나에 a · b 차례로, 렌더 포함 (rtrender1, 시뮬 + 가우시안 갱신 + 래스터화, 프레임 3~), 진단 끔.
"""
import collections
import json
import math
import os
import sys

O = sys.argv[1]
spec = [l.split(None, 4) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hhtc_spec.txt")) if l.strip() and not l.startswith("#")]
spec = [s + [""] * (5 - len(s)) for s in spec]
LIM = 1000.0 / 60.0
SCN = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}
ZF = 0.02


def jl(n, f):
    p = os.path.join(O, n, f)
    try:
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else None
    except Exception:
        return None


def rd(n):
    return jl(n, "fem.jsonl")


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def oom(n):
    p = os.path.join(O, n, "log.txt")
    return os.path.exists(p) and "OutOfMemoryError" in open(p, errors="replace").read()


def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def res(n):
    s = summ(n)
    if s is None:
        return "발산 (메모리)" if oom(n) else ("도는 중/실패" if rd(n) else "없음")
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def div(n):
    return res(n).startswith("발산")


def err(n):
    s = summ(n)
    if s is None:
        return None
    e = s.get("err_mean_done", s.get("err_mean"))
    return 100 * e if e is not None else None


def msl(r):
    return [(x["t"], x["ms"]) for x in (r or []) if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]


def tz(n):
    """단독 판 한 개: '평균 / 최악 (> 16.7 프레임 수)' 또는 발산."""
    r = rd(n)
    if r is None and not oom(n):
        return None
    if oom(n):
        return "메모리 발산"
    m = msl(r)
    if div(n):
        return res(n)
    if not m:
        return "-"
    q = [b for _, b in m]
    return f"{sum(q) / len(q):.2f} / {max(q):.1f} ({sum(1 for b in q if b > LIM)})"


def com(r, t):
    return r[t]["hp"][0] * 1000 if r and t < len(r) and "hp" in r[t] else None


def lj_at(r, t):
    if not r or t >= len(r):
        return None
    v = (r[t].get("en") or {}).get("lj0")
    return None if v is None or r[t].get("err") is None or not math.isfinite(r[t]["err"]) else 100 * v


def lj_last(r):
    for q in reversed(r or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return 100 * v
    return None


def e1s(n):
    e = jl(n, "e1.jsonl") or []
    if not e:
        return None
    return dict(nfar=max(q.get("nfar", 0) for q in e), dmax=max((q.get("dmax") or 0) for q in e))


def e2s(n):
    L = jl(n, "e2.jsonl") or []
    F = [q for q in L if "esc" not in q]
    E = [q for q in L if "esc" in q]
    if not F:
        return None
    seen, kk = set(), []
    for e in E:
        for fr in e["frames"]:
            if fr.get("v1") is not None and fr.get("v0") is not None and fr["v1"] - fr["v0"] > 1.0 and "vn_end_max_sol" in fr and fr["f"] not in seen:
                seen.add(fr["f"]); kk.append(fr)
    zb = [q for q in F if "nbelow" in q]
    return dict(nk=len(kk), vsol=max((q["vn_end_max_sol"] for q in kk), default=None),
                below=max((q["nfar_below"] for q in zb), default=None) if zb else None)


def e5zmin(n):
    L = [q for q in (jl(n, "e5.jsonl") or []) if "nq" in q and q.get("zmin") is not None]
    return min((q["zmin"] for q in L), default=None)


def bbox(n):
    p = os.path.join(O, n, "bbox.txt")
    if os.path.exists(p):
        for l in open(p):
            if "pred.pt" in l:
                try:
                    d_ = json.loads(l.split(" ", 1)[1])
                    return max(z["nfar"] for z in d_.values()), max(z["dmax"] for z in d_.values())
                except Exception:
                    pass
    return None


def period(n):
    p = os.path.join(O, n, "period.json")
    try:
        d = json.load(open(p))
        r = d.get("ratio")
        return (r if r is not None and math.isfinite(r) else None), d.get("amp_ratio"), d.get("nzc")
    except Exception:
        return None


def rng(v, d=2):
    v = [q for q in v if q is not None]
    return f"{min(v):.{d}f}~{max(v):.{d}f}" if v else "-"


def mean(v):
    v = [q for q in v if q is not None]
    return sum(v) / len(v) if v else None




def hcl(n):
    return jl(n, "hc.jsonl") or []


def contact_t(n):
    """첫 충돌 걸음이 있는 시도의 프레임 번호 t (0 부터; hc.jsonl, 없으면 3 = f4)."""
    for q in hcl(n):
        if q.get("ncol", 0) > 0 and q.get("h", 0) > 0:
            return int(round(q["t"] / q["h"]))
    return 3


def ev(n):
    r = rd(n)
    if not r:
        return None
    t = contact_t(n)
    if t >= len(r) or t < 1:
        return None
    tr = (r[t].get("ks") or {}).get("tries") or []
    acc1 = len(tr) == 1 and int(tr[0][0]) == 1 and int(tr[0][1]) == 0
    it, uf = r[t].get("it") or {}, r[t].get("uf") or {}
    conv = acc1 and int(it.get("capN", 1)) == 0 and min(uf.get("smin") or [0]) >= 1.0
    en = lambda i: (r[i].get("en") or {}) if 0 <= i < len(r) else {}
    zc = [en(i).get("zc") for i in range(t, len(r))]
    zc = [z for z in zc if z is not None and math.isfinite(z)]
    E = lambda i: (en(i).get("KE", 0) + en(i).get("PE", 0) + en(i).get("Eel", 0)) if en(i) else None
    return dict(t=t, tries=[[int(a), int(b)] for a, b, *_ in tr], acc1=acc1, vz0=en(t - 1).get("vz"), vz1=en(t).get("vz"), vz2=en(t + 1).get("vz"), vz3=en(t + 2).get("vz"),
                E0=E(t - 1), E1=E(t), E2=E(t + 1), conv=conv, g2=(uf.get("g2") or [None])[-1], rise=(max(zc) - min(zc)) if zc else None, bounce=(en(t).get("vz") or 0) > 0)


def hcsum(n):
    L = hcl(n)
    if not L:
        return None
    return dict(nstep=sum(q["nstep"] for q in L), ncol=sum(q["ncol"] for q in L), new=sum(q["new"] for q in L), eng=sum(q["eng"] for q in L), napp=sum(q["napp"] for q in L))


def hcof(fl):
    import re
    m = re.search(r"@hc (\d)", fl)
    return int(m.group(1)) if m else None


HCN = {None: "P1X (훅 없음)", 0: "P1X (대조, 진단만)", 1: "P1X + H1 (충돌 걸음 몸 전체 후퇴 오일러)", 2: "P1X + H2 (새 접촉 노드 후퇴 오일러)", 3: "P1X + H3 (접촉 노드 가속도 기억 0)"}


def rfof(fl):
    return " @tc 8" in " " + fl + " "


def lab(fl):
    return HCN.get(hcof(fl), fl) + (" + Rf (예산 16.7 − 앞 프레임 렌더)" if rfof(fl) else "")


def e1f(n):
    e = jl(n, "e1.jsonl") or []
    if not e:
        return None, None
    return max((q.get("nfast", 0) or 0) for q in e), max((q.get("nfar", 0) or 0) for q in e)


def rrs(n):
    R = jl(n, "rr.jsonl")
    if not R:
        return None
    q3 = [q for q in R if q.get("t", 0) >= 2]
    tot = [q["tot"] for q in q3 if q.get("tot") is not None and math.isfinite(q["tot"])]
    a = [q["a"] for q in q3 if q.get("a") is not None and math.isfinite(q["a"])]
    bc = [q["b"] + q["c"] for q in q3 if q.get("b") is not None and q.get("c") is not None]
    if not tot:
        return None
    return dict(tm=sum(tot) / len(tot), tx=max(tot), to=sum(1 for m in tot if m > LIM), am=sum(a) / len(a) if a else None, ax=max(a) if a else None,
                ao=sum(1 for m in a if m > LIM), rm=sum(bc) / len(bc) if bc else None, nf=len(tot))


def tzr(n):
    if oom(n):
        return "메모리 발산"
    if rd(n) is None:
        return None
    if div(n):
        return res(n)
    q = rrs(n)
    if q is None:
        return tz(n)
    return f"{q['tm']:.2f} / {q['tx']:.1f} ({q['to']})"


def zruns(sc, h, rf=False):
    for st, n, s_, md, f_ in spec:
        if md == "z" and s_ == sc and not st.endswith("k") and hcof(f_) == h and rfof(f_) == rf:
            return n
    return None


def zcol2(sc, h, rf=False):
    n = zruns(sc, h, rf)
    if not n:
        return "-"
    v = [tzr(f"{n}_{k_}") for k_ in ("a", "b")]
    v = [q for q in v if q is not None]
    return " · ".join(v) if v else "-"


print("# 절 CM58 묶음 hhtc -- 수박 첫 접촉 뒤 몸 전체 튀어오름: 충돌 걸음 HHT 갱신 정의\n")
print(__doc__.split("\n", 1)[1])

# ---- 재현: 정의별 K 1 수렴 받음 사건과 속도
G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc.startswith("M") and md in ("r", "g"):
        G.setdefault((hcof(fl), ("짧은 판 8 프레임" if "--frames 8" in fl else "100 프레임") + (" Rf" if rfof(fl) else "")), []).append(n)
rows, det = [], []


def rg(L, k, d=2):
    v = [e[k] for _, e in L if e.get(k) is not None]
    return f"{min(v):+.{d}f}~{max(v):+.{d}f}" if v else "-"


for (h, ln), ns in G.items():
    E_ = [(n, ev(n)) for n in ns]
    E_ = [(n, e) for n, e in E_ if e]
    if not E_:
        continue
    acc = [(n, e) for n, e in E_ if e["acc1"]]
    cv = [(n, e) for n, e in E_ if e["conv"]]
    oth = [(n, e) for n, e in E_ if not e["acc1"]]
    nb = sum(1 for _, e in E_ if e["bounce"])
    rows.append(f"| {HCN.get(h, h)} | {ln} | {len(E_)} | {len(acc)} | {len(cv)} | {nb} | {rg(cv, 'vz1')} · {rg(cv, 'vz2')} | {rg(acc, 'vz1')} · {rg(acc, 'vz2')} | {rg(acc, 'rise', 3)} | "
                f"{rg(acc, 'E0')} → {rg(acc, 'E1')} → {rg(acc, 'E2')} | {rg(oth, 'vz1')} · {rg(oth, 'vz2')} | {rg(oth, 'rise', 3)} |")
    for n, e in acc:
        det.append(f"| {n} | {HCN.get(h, h)} | f{e['t'] + 1} | {e['tries']} | {'수렴' if e['conv'] else '덜 풀림 (S 후퇴 오일러)'} · {fmt(e['g2'], 3)} | {fmt(e['vz0'])} → {fmt(e['vz1'])} → {fmt(e['vz2'])} → {fmt(e['vz3'])} | "
                   f"{fmt(e['E0'])} → {fmt(e['E1'])} → {fmt(e['E2'])} | {fmt(e['rise'], 3)} | {res(n)} |")
if rows:
    print("## 재현 -- 첫 접촉 프레임 K 1 수렴 받음 사건 (절 CM57.4 의 갈림 자리)\n")
    print("vz = 몸 질량중심 법선 속도 m/s (첫 접촉 프레임 끝 · 다음 프레임 끝), 오름 = 첫 접촉 프레임부터 질량중심 z 최대 − 최소 m, E = 운동 + 중력 위치 + SNH 탄성 J (앞 프레임 끝 → 접촉 프레임 끝 → 다음 프레임 끝).\n")
    print("| 정의 | 길이 | 판 | K 1 받음 판 | 그중 수렴 사건 (허용치 멈춤·보폭 1) | 되튐 판 (접촉 프레임 끝 vz > 0) | 수렴 사건 vz 접촉 · 다음 | 받음 판 vz 접촉 · 다음 | 받음 판 오름 m | 받음 판 E J | 다시 하거나 쪼갠 판 vz 접촉 · 다음 | 다시 하거나 쪼갠 판 오름 m |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
if det:
    print("### K 1 수렴 받음 판 (판마다)\n")
    print("| 판 | 정의 | 접촉 프레임 | 시도 [K, 무너짐] | 풀이 · 끝 ‖g‖²/F² | vz 앞 → 접촉 → 다음 → 그다음 | E 앞 → 접촉 → 다음 J | 오름 m | 결과 |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(det))
    print()

# ---- 수박 (절 CM55 표 형식; Rf 판 = 렌더를 같은 프로세스에서 돌리며 예산 16.7 − 앞 프레임 렌더)
G = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc.startswith("M") and md == "g" and not st.endswith("k"):
        G.setdefault((hcof(fl), rfof(fl)), []).append(n)
rows = []
for (h, rf), ns in G.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    zn = zruns("M3", h, rf)
    zt = zd = 0
    if zn:
        for k_ in ("a", "b"):
            nz = f"{zn}_{k_}"
            if rd(nz) is not None or oom(nz):
                zt += 1
                zd += int(div(nz))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    es = [e for e in (e1s(n) for n, _ in rs) if e]
    far = [e["nfar"] for e in es]
    box = [max(x.get("nn", 0) for x in r) for _, r in rs if r]
    e2 = [e2s(n) for n, _ in rs]
    bel = [q["below"] for q in e2 if q and q.get("below") is not None]
    nk = sum(q["nk"] for q in e2 if q)
    vs = [q["vsol"] for q in e2 if q and q.get("vsol") is not None]
    zm = [z for z in (e5zmin(n) for n, _ in rs) if z is not None]
    lv = [lj_last(r) for _, r in rs if r]
    lv = [q for q in lv if q is not None]
    f5 = mean([lj_at(r, 4) for _, r in rs if r])
    f6 = mean([lj_at(r, 5) for _, r in rs if r])
    f8 = mean([lj_at(r, 7) for _, r in rs if r])
    evs = [ev(n) for n, _ in rs]
    nb = sum(1 for e in evs if e and e["bounce"])
    na = sum(1 for e in evs if e and e["acc1"])
    ff = [e1f(n) for n, _ in rs]
    fmx = max((a for a, _ in ff if a is not None), default=None)
    rows.append(f"| {HCN.get(h, h)}{' + Rf' if rf else ''} | {len(rs)} | {dv}/{len(rs)} + {zd}/{zt} | {nb}/{len(rs)} ({na}) | {fmx} | {sum(1 for b in bel if b > 0)}/{len(bel)} · {fmt(min(zm) + ZF if zm else None, 3)} | "
                f"{(str(sum(1 for q in far if q > 20)) + '/' + str(len(far)) + ' · ' + str(max(far)) + ' · ' + fmt(max(e['dmax'] for e in es))) if far else '-'} | "
                f"{(max(box) / 1e6) if box else 0:.2f} 백만 ({sum(1 for b in box if b > 5e6)}/{len(box)}) | {fmt(f5)} · {fmt(f6)} · {fmt(f8)} · {fmt(mean(lv))} [{rng(lv)}] | {rng(E_)} | "
                f"{nk} / {nk / len(rs):.1f} | {('%.0f~%.0f' % (min(vs), max(vs))) if vs else '-'} | {zcol2('M3', h, rf)} |")
if rows:
    print("## 수박 (절 CM55 표 형식; 비결정, 진단 D1·D2·D5 켬, 100 프레임)\n")
    print("| 후보 | 판 | 발산 (판정 + 단독) | 되튐 판 (K 1 수렴 받음) | 빠른 노드 최대 | 바닥 뚫음 판 · z 최소 m | 이탈 판 · 최대 수 · 거리 m | 노드 상자 최대 (> 5 백만 판) | 손상 f5 · f6 · f8 (평균) · 끝 평균 [범위] % | 오차 % | 튐 프레임 (판 합 / 판당) | 튐 노드 끝 속도 최대 m/s | 단독 렌더 포함 평균 / 최악 ms (> 16.7 프레임) a · b |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()

# ---- 다른 씬
RG = collections.OrderedDict()
for st, n, sc, md, fl in spec:
    if sc in ("L", "W", "F", "B") and md == "nd" and not st.endswith("k"):
        RG.setdefault((sc, hcof(fl), rfof(fl)), []).append(n)
rows = []
for (sc, h, rf), ns in RG.items():
    rs = [(n, rd(n)) for n in ns if rd(n) is not None or oom(n)]
    if not rs:
        continue
    dv = sum(1 for n, _ in rs if div(n))
    E_ = [err(n) for n, _ in rs if err(n) is not None and not div(n)]
    met = "-"
    if sc == "W":
        f50 = [com(r, 49) for _, r in rs if r and com(r, 49) is not None]
        bb = [b for b in (bbox(n) for n, _ in rs) if b]
        met = (f"f50 {min(f50):.0f}~{max(f50):.0f} mm" if f50 else "f50 -") + (f", 이탈 {sum(1 for b in bb if b[0] > 20)}/{len(bb)} (최대 {max(b[0] for b in bb)} 개·{max(b[1] for b in bb):.2f} m)" if bb else "")
    elif sc == "F":
        pp = [p for p in (period(n) for n, _ in rs) if p]
        met = ("주기 비 " + rng([p[0] for p in pp]) + ", 진폭 비 " + rng([p[1] for p in pp]) + ", 영점 " + ",".join(str(p[2]) for p in pp)) if pp else "-"
    elif sc == "L":
        es = [e for e in (e1s(n) for n, _ in rs) if e]
        met = f"이탈 {sum(1 for e in es if e['nfar'] > 20)}/{len(es)} (최대 {max(e['nfar'] for e in es)} 개)" if es else "-"
    hs = [q for q in (hcsum(n) for n, _ in rs) if q]
    hcc = (f"{min(q['ncol'] for q in hs)}~{max(q['ncol'] for q in hs)} / {min(q['nstep'] for q in hs)}~{max(q['nstep'] for q in hs)} · {max(q['napp'] for q in hs)}") if hs else "-"
    ff = [e1f(n) for n, _ in rs]
    fmx = max((a for a, _ in ff if a is not None), default=None)
    fr = [b for _, b in ff if b is not None]
    rows.append(f"| {SCN[sc]} | {HCN.get(h, h)}{' + Rf' if rf else ''} | {len(rs)} | {dv} | {fmt(mean(E_))} [{rng(E_)}] | {met} | {fmx} · {sum(1 for b in fr if b > 20)}/{len(fr)} ({max(fr) if fr else '-'}) | {hcc} | {zcol2(sc, h, rf)} |")
if rows:
    print("## 다른 씬 (비결정 판 + 단독 렌더 포함 a · b)\n")
    print("| 씬 | 정의 | 판 | 발산 | 오차 % 평균 [범위] | 씬 지표 | 빠른 노드 (> 3 m/s) 최대 · 이탈 판 (최대 개) | 충돌 걸음 / 걸음 · 바꾼 노드 합 최대 | 단독 렌더 포함 평균 / 최악 ms (> 16.7 프레임) a · b |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()

# ---- 단독 렌더 포함 자세히
out = []
for st, n, sc, md, fl in spec:
    if md != "z":
        continue
    for k_ in ("a", "b"):
        nz = f"{n}_{k_}"
        if rd(nz) is None and not oom(nz):
            continue
        q = rrs(nz)
        ap = ""
        try:
            ap = open(os.path.join(O, nz, "apps_at_start.txt")).read().split("\n")
            ap = [l for l in ap if l.startswith("카드")][0]
        except Exception:
            ap = "-"
        if q:
            out.append(f"| {nz} | {SCN.get(sc, sc)} | {lab(fl)} | {res(nz)} | {fmt(err(nz))} | {q['am']:.2f} / {q['ax']:.1f} ({q['ao']}) | {fmt(q['rm'])} | {q['tm']:.2f} / {q['tx']:.1f} ({q['to']}) / {q['nf']} | {ap} | {e1f(nz)[0]} · {e1f(nz)[1]} |")
        else:
            out.append(f"| {nz} | {SCN.get(sc, sc)} | {lab(fl)} | {res(nz)} | {fmt(err(nz))} | - | - | - | {ap} | {e1f(nz)[0]} · {e1f(nz)[1]} |")
if out:
    print("## 단독 (렌더 포함, 프레임 3~; 시뮬 = 재생 GPU 시간, 렌더 = 가우시안 갱신 + 래스터화 CUDA 이벤트)\n")
    print("| 판 | 씬 | 정의 | 결과 | 오차 % | 시뮬 평균 / 최악 (> 16.7) | 렌더 평균 | 합계 평균 / 최악 (> 16.7) / 프레임 | 시작 때 그 카드 | 빠른 노드 최대 · 이탈 최대 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print()

# ---- 판별
print("## 판별 (판마다; 시간은 동시 판이라 참고)\n")
print("| 판 | 씬 | 플래그 | 결과 | 오차 % | 손상 f5 · f6 · f8 · 끝 % | wolf f50 mm · ficus 주기 비 | 이탈 최대 수 · 거리 m | 튐 프레임 · 바닥 뚫음 · z 최소 m | 노드 상자 최대 | 충돌 걸음 / 걸음 · 새 접촉 노드 · 바꾼 노드 | 첫 접촉 시도 · vz |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for st, n, sc, md, fl in spec:
    if md in ("z",):
        continue
    r = rd(n)
    if r is None and not oom(n):
        continue
    e = e1s(n)
    e2 = e2s(n)
    zm = e5zmin(n)
    dmg = " · ".join(fmt(lj_at(r, t)) for t in (4, 5, 7)) + " · " + fmt(lj_last(r)) if sc.startswith("M") else "-"
    pp = period(n) if sc == "F" else None
    met = fmt(com(r, 49), 0) if sc == "W" else ((fmt(pp[0]) if pp else "-") if sc == "F" else "-")
    hs = hcsum(n)
    ee = ev(n) if sc.startswith("M") else None
    print(f"| {n} | {SCN.get(sc, sc)} | `{fl.strip()}` | {res(n)} | {fmt(err(n))} | {dmg} | {met} | {(str(e['nfar']) + ' · ' + fmt(e['dmax'])) if e else '-'} | "
          f"{(str(e2['nk']) + ' · ' + str(e2['below'])) if e2 else '-'} · {fmt(zm + ZF if zm is not None else None, 3)} | {(max(x.get('nn', 0) for x in r) / 1e6) if r else 0:.2f} 백만 | "
          f"{(str(hs['ncol']) + '/' + str(hs['nstep']) + ' · ' + str(hs['new']) + ' · ' + str(hs['napp'])) if hs else '-'} | {(str(ee['tries']) + ' · ' + fmt(ee['vz1']) + ' → ' + fmt(ee['vz2'])) if ee else '-'} |")
print()
