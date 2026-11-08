"""절 CM60 묶음 wmseed1 표 (exe/wmcrack1_tab.py 복사 = 절 CM59 와 같은 정의 + E. 씨 입자 −inf 몫): python exe/wmseed1_tab.py <결과 디렉토리> <gt_jp 디렉토리> [씨 마스크 npz] -> markdown (stdout) + <결과>/wmseed1_curves.json.
판 = exe/wmseed1_spec.txt 의 수박 방식 r, 묶음 = 이름의 끝 _숫자 뺀 것 (비결정 반복). 뒤에 절 CM40 표 (exe/wmshape_tab.py, 같은 정의) 를 이 판 목록으로.
정답 = <결과>/gt/gt_wmshape.json (절 CM40) · gt_wmshape2.json (절 CM41, 손상 입자 J·Jp 분위). 손상 = logJp > 0 몫 (정답 끝 6.51 %), 생긴 E = 전 구간 최대 (운동 + 중력 위치 + SNH 탄성) − 처음 7.13 J,
IoU = 같은 입자 번호에서 손상 집합 교집합 / 합집합, 첫 손상 높이 = f5 손상 입자 중 바닥 위 27~53 cm 몫 (정답 57 %),
손상 입자 J·Jp = 탄성 det F · exp(logJp + 0.04) 의 10·50·90 % 분위 (wsh/fs_*.npz, exe/wmcrack1_wrap.py --wc1sv; 정답 f8 1.19·1.92·5.64),
첫 접촉 K = 프레임 4 (t 3) 에 받아들인 걸음 수 (쪼개기 시도 목록 ks), 다시 하기 = X3c (무너짐 다시 하기) 를 탄 판 수, 미룬 걸음 = fem_wmshape3 진단 ws3."""
import contextlib
import io
import json
import math
import os
import runpy
import sys

import numpy as np

O, JP = sys.argv[1], sys.argv[2]
SEEDF = sys.argv[3] if len(sys.argv) > 3 else ""
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["WSH_SPEC"] = os.path.join(HERE, "wmseed1_spec.txt")
os.environ["WSH_CUR"] = "wmseed1_curves.json"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = [os.path.join(HERE, "wmshape_tab.py"), O, JP]  # noqa
    T = runpy.run_path(os.path.join(HERE, "wmshape_tab.py"), run_name="__main__")
base = buf.getvalue()
W, grp, G, rng, val, dE, load_sv, gt_lj = T["W"], T["grp"], T["G"], T["rng"], T["val"], T["dE"], T["load_sv"], T["gt_lj"]
try:
    G2 = json.load(open(os.path.join(O, "gt", "gt_wmshape2.json")))
except Exception:
    G2 = {}


def summ(n):
    try:
        return json.load(open(os.path.join(O, n, "summary.json")))
    except Exception:
        return None


def zms(g):
    o = []
    for r_ in ("a", "b"):
        n = f"z{g}_{r_}"
        f = os.path.join(O, n, "fem.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        m = [x["ms"] for x in rows if x["t"] >= 2 and x.get("ms") is not None and math.isfinite(x["ms"])]
        s = summ(n)
        o.append((r_, float(np.mean(m)) if m else None, float(np.max(m)) if m else None, (s or {}).get("diverged"), s is None, sum(1 for q in m if q > 16.7)))
    return o


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def iou(n, t):
    z = load_sv(n, t)
    if z is None:
        return None
    gl = gt_lj(t) > 0
    dm = z[1] > 0
    un = (dm | gl).sum()
    return float((dm & gl).sum() / un) if un else None


def jjp(n, t):
    z = load_sv(n, t)
    p = os.path.join(O, n, "wsh", f"fs_{t:03d}.npz")
    if z is None or not os.path.exists(p):
        return None
    det = np.load(p)["det"].astype(np.float64)
    lj = z[1].astype(np.float64)
    m = (lj > 0) & np.isfinite(det)
    if m.sum() < 10:
        return None
    return [float(q) for q in np.quantile(det[m] * np.exp(lj[m] + 0.04), [0.1, 0.5, 0.9])]


def jq(v, i):
    v = [q[i] for q in v if q is not None]
    return f"{np.mean(v):.2f}" if v else "-"


def k4(n):
    for q in W[n]["rows"]:
        if q["t"] == 3:
            ks = q.get("ks") or {}
            tr = ks.get("tries") or []
            return ks.get("K", q.get("K")), any(len(t_) > 3 and t_[3] for t_ in tr) or (len(tr) > 1 and any(t_[1] for t_ in tr[:-1]))
    return None, None


def ws3(n):
    a = b = 0
    for q in W[n]["rows"]:
        w = q.get("ws3")
        if w and w.get("on"):
            a += w["ok"]; b += w["defer"]
    return (b / (a + b)) if a + b else None


def e2rows(n):
    p = os.path.join(O, n, "e2.jsonl")
    if not os.path.exists(p):
        return []
    o = []
    for l in open(p):
        try:
            q = json.loads(l)
        except Exception:
            continue
        if "f" in q:
            o.append(q)
    return o


def cause(n):
    s = summ(n)
    lg = os.path.join(O, n, "log.txt")
    t = open(lg, errors="ignore").read()[-6000:] if os.path.exists(lg) else ""
    if "OutOfMemoryError" in t:
        return "메모리"
    if s is None:
        return "죽음"
    return f"발산 f{s.get('diverged')}" if s.get("diverged") else "완주"


gj = lambda t, i: (G2.get(str(t)) or {}).get("dJt", [None] * 3)[i]
print("# 절 CM60 묶음 wmseed1 -- 수박 씨 영역 (β 3e9, 첫 접촉 −inf = 영원히 탄성) 을 입자 수준으로\n")
print(__doc__.split("\n", 1)[1].strip() + "\n")
print("## A. 후보 요약 (절 CM46 표 형식; 비결정 판 + 단독 z a·b; 정답 손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51 %, 끝 r95 0.895 · 질량중심 0.166 m)\n")
print("| 묶음 | 판 | 발산 (판정 + 단독) | 끝 손상 % [범위] | f5 · f6 · f8 손상 % | 판 사이 표준편차 %p | 손상 입자 J·Jp 10·50·90 % f8 | f100 | f100 r95 · 질량중심 m | 정답 IoU f5 · f10 · f100 | f5 손상 27~53 cm 몫 % | 생긴 E J | 오차 % | 첫 접촉 K (다시 하기 판) | 바닥 뚫음 판 · 이탈 판 | 단독 ms 평균/최대 (> 16.7 프레임) a · b | 미룬 걸음 몫 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print(f"| 정답 | - | - | {100 * G[100]['d0']:.2f} | {100 * G[5]['d0']:.2f} · {100 * G[6]['d0']:.2f} · {100 * G[8]['d0']:.2f} | - | {fmt(gj(8, 0), 2)} · {fmt(gj(8, 1), 2)} · {fmt(gj(8, 2), 2)} | "
      f"{fmt(gj(100, 0), 2)} · {fmt(gj(100, 1), 2)} · {fmt(gj(100, 2), 2)} | {G[100]['r95']:.3f} · {G[100]['zc']:.3f} | 1 | {100 * G[5]['dh'][2]:.0f} | (+16.7 첫 접촉) | - | - | - | - | - |")
CUR, ROWS = {}, {}
for g, ns in grp.items():
    last = []
    for n in ns:
        by = W[n]["by"]
        last.append(by[max(by)]["d0"] if by else None)
    bad = sum(1 for n in ns if (summ(n) is None) or summ(n).get("diverged"))
    z = zms(g)
    zbad = sum(1 for q in z if q[3] or q[4])
    er = [100 * s["err_mean_done"] for s in (summ(n) for n in ns) if s]
    lv = [q for q in last if q is not None]
    sd = 100 * float(np.std(lv)) if len(lv) > 1 else None
    zs = " · ".join(f"{fmt(q[1], 1)}/{fmt(q[2], 0)} ({q[5]})" + (f" (발산 f{q[3]})" if q[3] else "") for q in z) or "-"
    dh = [val(n, 5, "dh", 2) for n in ns]
    wd = [ws3(n) for n in ns]
    j8 = [jjp(n, 8) for n in ns]; j100 = [jjp(n, 100) for n in ns]
    kk = [k4(n) for n in ns]
    kset = sorted({q[0] for q in kk if q[0] is not None})
    nre = sum(1 for q in kk if q[1])
    e2 = [e2rows(n) for n in ns]
    nbel = sum(1 for r in e2 if any(q.get("nbelow", 0) > 0 for q in r))
    nfar = sum(1 for r in e2 if max((q.get("nfar", 0) for q in r), default=0) > 20)
    row = (f"| {g} | {len(ns)} | {bad}/{len(ns)} + {zbad}/{len(z)} | {rng(last, 2, True)} | {rng([val(n, 5, 'd0') for n in ns], 2, True)} · {rng([val(n, 6, 'd0') for n in ns], 2, True)} · {rng([val(n, 8, 'd0') for n in ns], 2, True)} | {fmt(sd, 2)} | "
           f"{jq(j8, 0)} · {jq(j8, 1)} · {jq(j8, 2)} | {jq(j100, 0)} · {jq(j100, 1)} · {jq(j100, 2)} | "
           f"{rng([val(n, 100, 'r95') for n in ns])} · {rng([val(n, 100, 'zc') for n in ns])} | {rng([iou(n, 5) for n in ns], 2)} · {rng([iou(n, 10) for n in ns], 2)} · {rng([iou(n, 100) for n in ns], 2)} | "
           f"{rng(dh, 0, True)} | {rng([dE(n) for n in ns], 2)} | {rng(er, 2)} | {'·'.join(str(k) for k in kset) or '-'} ({nre}) | {nbel}/{len(ns)} · {nfar}/{len(ns)} | {zs} | {rng(wd, 2) if any(q is not None for q in wd) else '-'} |")
    print(row)
    ROWS[g] = row
    CUR[g] = {n: {str(t): W[n]["by"][t].get("d0") for t in W[n]["by"]} for n in ns}

FR = (5, 6, 8, 10, 15, 20, 30, 50, 100)
print("\n## B. 손상 곡선 (묶음 평균 %; 정답 맨 위)\n")
print("| 묶음 | " + " | ".join(f"f{t}" for t in FR) + " |")
print("|---|" + "---|" * len(FR))
print("| 정답 | " + " | ".join(f"{100 * G[t]['d0']:.2f}" for t in FR) + " |")
for g, ns in grp.items():
    print(f"| {g} | " + " | ".join(rng([val(n, t, 'd0') for n in ns], 2, True).split(" [")[0] for t in FR) + " |")

print("\n## C. 높이 H · 퍼짐 r95 · 질량중심 z 곡선 (묶음 평균 m; f4 · f5 · f6 · f10 · f20 · f50 · f100)\n")
FC = (4, 5, 6, 10, 20, 50, 100)
print("| 묶음 | H | r95 | 질량중심 z |")
print("|---|---|---|---|")
print("| 정답 | " + " | ".join(" · ".join(f"{G[t][k]:.3f}" for t in FC) for k in ("H", "r95", "zc")) + " |")
for g, ns in grp.items():
    print(f"| {g} | " + " | ".join(" · ".join(rng([val(n, t, k) for n in ns]).split(" [")[0] for t in FC) for k in ("H", "r95", "zc")) + " |")

print("\n## D. 판별 첫 접촉·바닥 뚫음·이탈 (진단 D2: 바닥 밑 = z < 0.02 − 한 칸, 이탈 = 몸 상자 4 칸 밖)\n")
print("| 판 | 끝 | 프레임 | f4 쪼개기 시도 [K, 무너진 걸음, ms, X3c] | f4 끝 KE · 탄성 J | 바닥 밑 입자 첫 프레임 · 최대 수 | z 최소 m | 이탈 최대 수 | 노드 수 최대 (백만) | 미룬 걸음 몫 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for g, ns in grp.items():
    for n in ns:
        r = e2rows(n)
        nb = [q for q in r if q.get("nbelow", 0) > 0]
        nnm = max((q.get("nn", 0) for q in W[n]["rows"]), default=0)
        nnm = max([nnm] + [q.get("nn", 0) for q in r])
        q3 = [q for q in W[n]["rows"] if q["t"] == 3]
        tr = (q3[0].get("ks") or {}).get("tries") if q3 else None
        en = (q3[0].get("en") or {}) if q3 else {}
        print(f"| {n} | {cause(n)} | {len(W[n]['rows'])} | {json.dumps([[round(v, 1) if isinstance(v, float) else v for v in t_] for t_ in tr]) if tr else '-'} | {fmt(en.get('KE'), 3)} · {fmt(en.get('Eel'), 3)} | "
              f"{('f%d' % nb[0]['f']) if nb else '-'} · {max((q['nbelow'] for q in r), default=0)} | {min((q['zmin'] for q in r), default=float('nan')):.3f} | {max((q.get('nfar', 0) for q in r), default=0)} | {nnm / 1e6:.2f} | {fmt(ws3(n), 2)} |")

print("\n## E. 씨 입자 logJp −inf 몫 (원래 순번 입자, wsh/xs_*.npz float16; 씨 = wm_seed.npz (GF 러너 판정) 16.09 %; 묶음 평균 [범위])\n")
try:
    SEED = np.load(SEEDF)["seed"].astype(bool) if SEEDF else None
except Exception as ex:
    SEED = None
    print(f"(씨 마스크 못 읽음 {ex!r})")
if SEED is not None:
    FE = (3, 4, 5, 6, 10, 30, 100)
    print("| 묶음 | " + " | ".join(f"씨 −inf % f{t}" for t in FE) + " | 비씨 −inf 입자 최대 | 비유한 (NaN·+inf) 입자 최대 |")
    print("|---|" + "---|" * (len(FE) + 2))
    gr = []
    for t in FE:
        try:
            l = gt_lj(t)
            gr.append(f"{100 * float(np.isneginf(l[SEED]).mean()):.1f}")
        except Exception:
            gr.append("-")
    print("| 정답 | " + " | ".join(gr) + " | - | - |")
    for g, ns in grp.items():
        cells, nsd, nnf = [], 0, 0
        for t in FE:
            v = []
            for n in ns:
                z = load_sv(n, t)
                if z is None:
                    continue
                l = np.asarray(z[1]).astype(np.float32)
                v.append(100 * float(np.isneginf(l[SEED]).mean()))
                nsd = max(nsd, int(np.isneginf(l[~SEED]).sum()))
                nnf = max(nnf, int((np.isnan(l) | np.isposinf(l)).sum()))
            cells.append(rng(v, 1, False) if v else "-")
        print(f"| {g} | " + " | ".join(cells) + f" | {nsd} | {nnf} |")

print("\n# 절 CM40 과 같은 표 (이 판 목록)\n")
print(base.split("\n", 1)[1])
try:
    cur = json.load(open(os.path.join(O, "wmseed1_curves.json")))
    cur["d0"] = CUR
    json.dump(cur, open(os.path.join(O, "wmseed1_curves.json"), "w"))
except Exception as ex:
    print(f"(곡선 json 덧붙이기 실패 {ex!r})", file=sys.stderr)
