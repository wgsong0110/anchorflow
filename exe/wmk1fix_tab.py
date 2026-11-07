"""절 CM24 묶음 wmk1fix 표: 수박 기본 판 K 1 첫 접촉 걸음 덜 풀림 고치기 -- (1a) 보폭 막는 원인 끄기, (1b) 적응 걸음 쪼개기, (2) 덜 풀린 걸음 후퇴 오일러 갱신, (3) 덜 풀린 걸음 가속도 기억 0.
사용: python exe/wmk1fix_tab.py <출력 디렉토리> [정답 wm.pt]
튐 = 질량중심 z 가 그때까지 최저점보다 오른 최대 (m; 정답은 진단 nrec 의 정답 질량중심 gz 로 같은 식).
Δv = 프레임 끝 입자 질량 가중 법선 속도 차 (erec vz; 정답은 정답 질량중심 z 의 중심 차분 대신 정답 v 가 있으면 그것).
에너지 E = 운동 + 중력 위치 (−Σ m g·x) + SNH 탄성 (Σ V Ψ(F_입자)), 단위 J. 손상 = 위치 유한 입자 중 logJp > 0 몫 (%).
덜 풀린 걸음 = 판정 커널 (허용치 εN 으로 안 멈췄고 걸음 안 뉴턴 보폭 최소 < 1).
"""
import json
import math
import os
import sys

O = sys.argv[1]
WM = sys.argv[2] if len(sys.argv) > 2 else None
VAR = [("base", "기본 판"),
       ("noeb", "(1a) 시작점 --y0eb 끔 (실현 가능 k0 만)"), ("noy0u", "(1a) 시작점 뒤집힘 필터 --y0u 끔"),
       ("noend", "(1a) --ls_end 끔"), ("noinv", "(1a) --ls_inv 끔"), ("noie", "(1a) --ls_inv·--ls_end 끔"),
       ("ks3", "(1b) 적응 걸음 쪼개기 (최대 K 8)"),
       ("ube", "(2) 덜 풀린 걸음 후퇴 오일러 갱신"), ("ubeall", "(2) 비교: 모든 걸음 후퇴 오일러 갱신 (HHT 풀이)"), ("be", "(2) 비교: 후퇴 오일러 적분 (HHT 없음)"),
       ("ua0", "(3) 덜 풀린 걸음 가속도 기억 0")]
H = 0.03


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None, None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows, js(os.path.join(O, n, "summary.json"))


def bounce(zs):
    lo, b, at = float("inf"), 0.0, None
    for i, z in enumerate(zs):
        if z is None:
            continue
        lo = min(lo, z)
        if z - lo > b:
            b, at = z - lo, i + 1
    return b, at


def fm(x, f="{:.3g}"):
    return "-" if (x is None or (isinstance(x, float) and not math.isfinite(x))) else f.format(x)


def res(s, rows):
    if s is None:
        return "도는 중" if rows else "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def energy(r):
    e = r.get("en")
    if not e or e.get("Eel") is None:
        return None
    return e["KE"] + e["PE"] + e["Eel"]


def gt_energy():
    p = os.path.join(O, "gt_energy.json")
    g = js(p)
    if g is not None or WM is None:
        return g
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
        import torch
        from anchorflow import deform_model as DM
        from anchorflow import fem_wmkick_diag as FRD
        from anchorflow import phys_resid as PR
        d = torch.load(WM, map_location="cpu", weights_only=False)
        cfg = d["cfg"]; mu, lam = PR.lame(cfg["E"], cfg["nu"])
        m = DM.traj_mass(d, "cpu").double(); vol = m / float(cfg["density"])
        Fg = DM.traj_F(d, float(cfg["frame_dt"]), "cpu", keep_gpu=False)
        g_ = torch.tensor([float(q) for q in cfg["g"]], dtype=torch.float64)
        out = []
        for f in range(0, min(16, d["x"].shape[0])):
            x = d["x"][f].double(); ok = torch.isfinite(x).all(-1)
            r = dict(t=f, zc=float((m[ok] * x[ok][:, 2]).sum() / m[ok].sum()), PE=float(-(m[ok] * (x[ok] @ g_)).sum()))
            if "v" in d and d["v"].shape[0] > f:
                v = d["v"][f].double(); ok2 = ok & torch.isfinite(v).all(-1)
                r.update(KE=float(0.5 * (m[ok2] * (v[ok2] ** 2).sum(1)).sum()), vz=float((m[ok2] * v[ok2][:, 2]).sum() / m[ok2].sum()))
            r["Eel"] = FRD._snh(Fg[f].double().reshape(-1, 9)[ok], vol[ok], float(mu), float(lam))[0]
            out.append(r)
        g = dict(frames=out)
        json.dump(g, open(p, "w"))
        return g
    except Exception as e:
        return dict(err=repr(e))


def zms(n):
    rows, s = load("z_" + n)
    if not rows:
        return "-"
    ms = [q["ms"] for q in rows][2:]
    ms = [q for q in ms if q == q]
    dv = s.get("diverged") if s else None
    return (f"{sum(ms) / len(ms):.1f} · {max(ms):.1f}" if ms else "-") + (f" (x{dv})" if dv else "") + ("" if s else " (도는 중)")


def row(n):
    rows, s = load(n)
    if not rows:
        return None
    err = {q["t"] + 1: q["err"] for q in rows}
    zs = [q.get("en", {}).get("zc") for q in rows]
    gz = [q.get("com", {}).get("gz") for q in rows]
    b, bat = bounce(zs)
    vz = {q["t"] + 1: q.get("en", {}).get("vz") for q in rows}
    E = {q["t"] + 1: energy(q) for q in rows}
    dE = [E[f] - E[f - 1] for f in range(2, max(E) + 1) if E.get(f) is not None and E.get(f - 1) is not None]
    lj = [q.get("en", {}).get("lj0") for q in rows if q.get("en")]
    uf = [(q["t"] + 1, sum(q["uf"]["fail"]), len(q["uf"]["fail"])) for q in rows if q.get("uf")]
    ks = [q["ks"] for q in rows if q.get("ks")]
    return dict(rows=rows, s=s, err=err, b=b, bat=bat, gz=gz, vz=vz, E=E, dEmax=(max(dE) if dE else None), lj=(lj[-1] if lj else None), uf=uf, ks=ks,
                em=(100 * s["err_mean_done"]) if s else None)


G = gt_energy()
print("# 절 CM24 수박 K 1 첫 접촉 걸음 덜 풀림 고치기 -- 묶음 wmk1fix (exe/stab_wmk1fix.sh)\n")
print(__doc__.split("\n", 2)[2].strip().replace("\n", "  \n"), "\n")
R = {}
for v, _ in VAR:
    for r in "ab":
        R[f"{v}_{r}"] = row(f"{v}_{r}")
# 정답 줄
gz = next((x["gz"] for x in R.values() if x and any(q is not None for q in x["gz"])), None)
GB, GBat = bounce(gz) if gz else (None, None)
gE = {}
gvz = {}
if G and G.get("frames"):
    for q in G["frames"]:
        if q.get("KE") is not None:
            gE[q["t"]] = q["KE"] + q["PE"] + q["Eel"]
        if q.get("vz") is not None:
            gvz[q["t"]] = q["vz"]
print("## 1. 판별 요약 (각 판 비결정 두 번 a · b)\n")
print("| 판 | 결과 a · b | 평균 오차 % a · b | 오차 % f4 · f8 · f12 · f20 · f50 · f100 (a) | 튐 m a · b (프레임) | Δv m/s f4 · f5 · f6 (a) | E f3 → f4 → f6 J (a) | 프레임당 E 최대 증가 J a · b | 손상 % a · b | 덜 풀린 걸음 프레임 (a: 수, 처음 셋) | 단독 ms 평균 · 최대 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for v, nm in VAR:
    A, B = R.get(v + "_a"), R.get(v + "_b")
    if A is None and B is None:
        print(f"| {nm} | 없음 | | | | | | | | | {zms(v)} |")
        continue
    rs = " · ".join(res(x["s"], x["rows"]) if x else "없음" for x in (A, B))
    em = " · ".join(fm(x["em"], "{:.2f}") if x else "-" for x in (A, B))
    cur = " · ".join(fm(100 * A["err"][f], "{:.1f}") if (A and f in A["err"]) else "x" for f in (4, 8, 12, 20, 50, 100)) if A else "-"
    bb = " · ".join((f"{x['b']:.3f} (f{x['bat']})" if x and x["bat"] else ("0" if x else "-")) for x in (A, B))
    dv = " · ".join(fm((A["vz"].get(f) - A["vz"].get(f - 1)) if (A and A["vz"].get(f) is not None and A["vz"].get(f - 1) is not None) else None, "{:+.2f}") for f in (4, 5, 6)) if A else "-"
    ee = " → ".join(fm(A["E"].get(f), "{:.2f}") for f in (3, 4, 6)) if A else "-"
    dm = " · ".join(fm(x["dEmax"], "{:+.2f}") if x else "-" for x in (A, B))
    lj = " · ".join(fm(100 * x["lj"], "{:.2f}") if (x and x["lj"] is not None) else "-" for x in (A, B))
    ufa = "-"
    if A and A["uf"]:
        fl = [f for f, k, _ in A["uf"] if k > 0]
        ufa = f"{len(fl)} ({', '.join('f%d' % f for f in fl[:3])})" if fl else "0"
    print(f"| {nm} | {rs} | {em} | {cur} | {bb} | {dv} | {ee} | {dm} | {lj} | {ufa} | {zms(v)} |")
gdv = " · ".join(fm((gvz[f] - gvz[f - 1]) if (f in gvz and f - 1 in gvz) else None, "{:+.2f}") for f in (4, 5, 6))
gee = " → ".join(fm(gE.get(f), "{:.2f}") for f in (3, 4, 6))
print(f"| 정답 | - | - | - | {fm(GB, '{:.3f}')} (f{GBat}) | {gdv} | {gee} | - | 6.51 (f100) | - | - |")
print(f"\n정답 에너지 계산: {('오류 ' + G['err']) if (G and G.get('err')) else ('프레임 0~15, wm.pt x·v·F' if G else '없음')}\n")

# ---- (1b) 걸음 수
for v in ("ks3",):
    for r in "ab":
        x = R.get(f"{v}_{r}")
        if not x or not x["ks"]:
            continue
        from collections import Counter
        c = Counter(k["K"] for k in x["ks"])
        tr = Counter(len(k["tries"]) for k in x["ks"])
        fr = [(q["t"] + 1, q["ks"]["K"], q["ks"]["tries"]) for q in x["rows"] if q.get("ks") and q["ks"]["K"] > 1]
        print(f"### (1b) {v}_{r}: 프레임별 최종 K 분포 {dict(sorted(c.items()))}, 시도 수 분포 {dict(sorted(tr.items()))}, 최대 K 에서도 덜 풀림 {sum(1 for q in x['rows'] if q.get('ks') and q['ks']['tries'][-1][1] > 0)} 프레임")
        print("K > 1 프레임 (프레임: 최종 K, 시도 [K, 덜 풀린 걸음, ms]): " + "; ".join(f"f{f}: K {k} {[[t[0], t[1], round(t[2], 1)] for t in tr_]}" for f, k, tr_ in fr[:30]) + ("" if len(fr) <= 30 else f" … (모두 {len(fr)})") + "\n")

# ---- (1a) f4 보폭 기록
print("## 2. 첫 접촉 프레임 f4 (와 f3·f5) 보폭 기록 (a 판, --nrec)\n")
print("시작점: y0k = 쓴 시작점 k (β = 2^-k, 1000 = U), k0 = 실현 가능 k0 (0 = 관성 예측 ŷ 실현 가능), y0b = 시작점 뒤집힘 필터 β. 반복마다: 뒤집힘 필터 보폭 si · 장벽 실현 가능 보폭 2^-kb · 쓴 보폭 st · ‖g‖/F (반복 시작).\n")
print("| 판 | 프레임 | y0k · k0 · y0b | 반복별 si · 2^-kb · st · ‖g‖/F | 끝 ‖g‖/F | 판정 (덜 풀림) | 질량중심 z 우리 · 정답 |")
print("|---|---|---|---|---|---|---|")
for v, nm in VAR:
    x = R.get(v + "_a")
    if not x:
        continue
    for q in x["rows"]:
        f = q["t"] + 1
        if f not in (3, 4, 5) or not q.get("nr"):
            continue
        nr = q["nr"]
        st = []
        for it in nr["it"]:
            if not it.get("ran"):
                st.append("(멈춤)" if it.get("stop") else "-")
                continue
            kb = it.get("kb", -1)
            st.append(f"{fm(it.get('si') if it.get('si', -1) >= 0 else None, '{:.3g}')} · {fm(2.0 ** -kb if kb >= 0 else None, '{:.3g}')} · {fm(it.get('st'), '{:.3g}')} · {fm(it.get('gF'), '{:.2g}')}")
        y0 = f"{q.get('y0k')} · {nr.get('k0')} · {[round(b_, 4) for b_ in nr.get('y0b', [])]}"
        uf = q.get("uf", {}).get("fail")
        com = q.get("com", {})
        print(f"| {nm} | {f}{(' (K %d)' % q['K']) if q.get('K', 1) > 1 else ''} | {y0} | {' / '.join(st)} | {', '.join(fm(g_, '{:.2g}') for g_ in nr.get('gend', []))} | {uf} | {fm(com.get('z'), '{:.4f}')} · {fm(com.get('gz'), '{:.4f}')} |")
print()

# ---- f4~6 속도 변화 분해 (--frec, K 1 HHT 판)
print("## 3. 프레임 4~6 속도 변화 분해 (몸 전체, m/s, a 판, --frec; HHT 식 -- (2)·(3) 판의 덜 풀린 걸음은 갱신식이 달라 대조 차가 큼)\n")
print("| 판 | 프레임 | v 시작 → 끝 | 측정 Δv | = 가속도 기억 | + 중력 | + 탄성 | + 장벽 | + 잔차 | 대조 차 | 끝 가속도 a1 m/s² |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for v, nm in VAR:
    x = R.get(v + "_a")
    if not x:
        continue
    for q in x["rows"]:
        f = q["t"] + 1
        if f not in (4, 5, 6) or not q.get("fr") or "dv" not in q["fr"]:
            continue
        d = q["fr"]["dv"].get("all")
        if not d:
            continue
        print(f"| {nm} | {f} | {d['v0']:.3f} → {d['v1']:.3f} | {d['meas']:+.3f} | {d['mem']:+.3f} | {d['grav']:+.3f} | {d['el']:+.3f} | {d['bar']:+.3f} | {d['res']:+.3f} | {d['chk']:+.1e} | {d['a1']:.4g} |")
print(f"\nh = {H} s")
