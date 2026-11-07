"""절 CM33 묶음 wmen 표 (수박 (2) 판 에너지 생성: 몫 분해 진단 --enrec + 고침 후보). 사용: python exe/wmen_tab.py <출력 디렉토리>
오차 = 끝난 프레임 평균. 손상 = 위치 유한 입자 중 logJp > 0 몫 (%, 마지막 프레임; 정답 6.51 %). E = 입자 운동 + 중력 위치 + SNH 탄성 (J; 처음 7.13).
에너지 생성 = 전 구간 최대 E − 처음 E (> 0 이면 생김). 시간 = b 판 (--erec 만) 프레임당 재생 GPU ms (첫 2 프레임 뺌), 한 줄 (카드 하나에 차례로).
덜 풀린 걸음 = 판정 커널 (uf.fail) 합. 뉴턴 = 허용치 제어 CT 의 뉴턴 반복 수 평균 (프레임당).
"""
import glob
import json
import math
import os
import sys

O = sys.argv[1]
NAME = {"p2": "(2) 판", "dn": "(2) 판 − --y0loc", "ro": "(a) 판정 잔차만", "nl": "(b) 자른 노드 실제 이동 갱신", "x6": "(c) 덜 풀린 걸음 뉴턴 +4", "mo": "(d) Φ 늘면 되돌림", "cm": "공통 후보 (--ue_loc 2 --ls0_keep)"}
CELLS = {"c25": "2.5", "c3": "3"}
E0 = 7.13


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load(n):
    p = os.path.join(O, n, "fem.jsonl")
    if not os.path.exists(p):
        return None, None
    return [json.loads(l) for l in open(p) if l.strip()], js(os.path.join(O, n, "summary.json"))


def cell(v):
    return next((CELLS[x] for x in v.split("+") if x in CELLS), "?")


def nm(v):
    x = [NAME.get(x, x) for x in v.split("+") if x not in CELLS]
    return " + ".join(x)


def fm(x, f="{:.3g}"):
    return "-" if (x is None or (isinstance(x, float) and not math.isfinite(x))) else f.format(x)


def res(s, rows):
    if s is None:
        return "도는 중" if rows else "없음"
    return ("발산 f%d" % s["diverged"]) if s.get("diverged") else ("완주 %d" % s["frames_done"])


def E(q):
    e = (q or {}).get("en") or {}
    return None if e.get("Eel") is None else e["KE"] + e["PE"] + e["Eel"]


def emax(rows):
    v = [(E(q), q["t"] + 1) for q in rows or [] if E(q) is not None and math.isfinite(E(q))]
    return max(v) if v else (None, None)


def lj_last(rows):
    for q in reversed(rows or []):
        v = (q.get("en") or {}).get("lj0")
        if v is not None and q.get("err") is not None and math.isfinite(q["err"]):
            return v
    return None


def ms(rows):
    v = [q["ms"] for q in rows[2:] if q.get("ms") is not None and math.isfinite(q["ms"])]
    return (sum(v) / len(v), max(v)) if v else (None, None)


def nN(rows):
    v = [(q.get("it") or {}).get("nN") for q in rows or []]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


ORD = ["p2", "dn", "ro", "nl", "x6", "mo", "cm"]
vs = sorted({os.path.basename(p)[:-2] for p in glob.glob(os.path.join(O, "*_a")) if not os.path.basename(p).startswith("vid_")},
            key=lambda v: (float(cell(v)) if cell(v) != "?" else 9, [ORD.index(x) if x in ORD else 99 for x in v.split("+") if x not in CELLS]))
print("# 절 CM33 묶음 wmen -- 수박 (2) 판 (절 CM32 --gq_oct 0 --ws_bar 4 --ws_fi + CM27 (d) --y0loc 3 --ue be --uf_res, 쪼개기 끔) 의 에너지 생성: 몫 분해 + 고침 후보\n")
print(__doc__.split("\n", 1)[1])
print("## 1. 판별 (각 판 비결정 두 번 a · b)\n")
print("| 셀 | 판 | 결과 a · b | 오차 % a · b | 손상 % a · b | 최대 E J (프레임) a · b | 생긴 E J a · b | E f100 J a · b | 덜 풀린 걸음 a · b | 뉴턴 평균 a | (d) 되돌림 · 버림 a | ms 평균 · 최대 (b) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for v in vs:
    R = {r: load(f"{v}_{r}") for r in "ab"}
    if not R["a"][0] and not R["b"][0]:
        continue
    c = [" · ".join(res(R[r][1], R[r][0]) for r in "ab"),
         " · ".join(fm(100 * R[r][1]["err_mean_done"], "{:.2f}") if R[r][1] else "-" for r in "ab"),
         " · ".join(fm(100 * lj_last(R[r][0]) if lj_last(R[r][0]) is not None else None, "{:.2f}") for r in "ab")]
    mx = {r: emax(R[r][0]) for r in "ab"}
    c.append(" · ".join(f"{fm(mx[r][0], '{:.2f}')} ({mx[r][1] or '-'})" for r in "ab"))
    c.append(" · ".join(fm(mx[r][0] - E0 if mx[r][0] is not None else None, "{:+.2f}") for r in "ab"))
    c.append(" · ".join(fm(E((R[r][0] or [{}])[-1]) if R[r][0] else None, "{:.2f}") for r in "ab"))
    c.append(" · ".join(str(sum(sum((q.get("uf") or {}).get("fail") or []) for q in R[r][0])) if R[r][0] else "-" for r in "ab"))
    c.append(fm(nN(R["a"][0]), "{:.2f}"))
    c.append(" · ".join(str(sum((q.get("enmo") or [0, 0])[i] for q in R["a"][0] or [])) for i in (0, 1)) if any("enmo" in q for q in R["a"][0] or []) else "-")
    m_, x_ = ms(R["b"][0]) if R["b"][0] else (None, None)
    c.append(f"{fm(m_, '{:.1f}')} · {fm(x_, '{:.0f}')}")
    print(f"| {cell(v)} | {nm(v)} | " + " | ".join(c) + " |")
print("| - | 정답 | - | - | 6.51 | 23.79 (4) | - | - | - | - | - | - |")

KEYS = [("P2G 운동", lambda e: e["p2g"]["KE"]), ("P2G 위치", lambda e: e["p2g"]["PE"]), ("P2G 탄성", lambda e: e["p2g"]["El"]),
        ("잔차 W_res", lambda e: e["step"]["Wres"]), ("고착·고정 W_con", lambda e: e["step"]["Wcon"]), ("탄성 C_el", lambda e: e["step"]["Cel"]),
        ("장벽 일 W_bar", lambda e: e["step"]["Wbar"]), ("갱신식 R_upd", lambda e: e["step"]["Rupd"]),
        ("  ⊃ 불일치 Σ Δx·g_upd", lambda e: e["step"]["Wupd"] if e.get("uf") else None), ("  ⊃ 수치 감쇠 −½Σm|Δv|²", lambda e: -e["step"]["Dnum"] if e.get("uf") else None),
        ("커밋 (RM)", lambda e: e["commit"]), ("G2P 운동", lambda e: e["g2p"]["KE"]), ("G2P 위치", lambda e: e["g2p"]["PE"]), ("G2P 탄성 (F 보간)", lambda e: e["g2p"]["El"]),
        ("입자 ΔE (측정)", lambda e: e["dEp"]), ("대조 (측정 − 몫 합)", lambda e: e["sum_chk"])]
SEG = [(1, 3), (4, 4), (5, 10), (11, 30), (31, 100), (1, 100)]


def segsum(rows, a, b, f):
    t, n = 0.0, 0
    for q in rows:
        e = q.get("enb")
        if not e or "err" in e or not (a <= q["t"] + 1 <= b):
            continue
        x = f(e)
        if x is None or not math.isfinite(x):
            continue
        t += x; n += 1
    return t if n else None


print("\n## 2. 에너지 몫 분해 (a 판 --enrec; 프레임 구간 합, J; 양수 = 에너지를 만든 몫)\n")
print("걸음 몫 (Δx = 걸음 끝 노드 변위 U1, 노드 합): 잔차 = Σ_자유 Δx·g (g = m/τ²(y − ŷ) + f_el(y) + G n, 마지막 뉴턴 뒤), 고착·고정 = 같은 식의 띠 고착·거의 빈 노드, "
      "탄성 C_el = ΔE_el(가우스 점, U1) − Σ Δx·f_el(y), 장벽 일 = −Σ Δx·G n, 갱신식 R_upd = ΔKE − Σ Δx·(m g + m/τ²(y − ŷ)) (덜 풀린 걸음 = 후퇴 오일러 갱신이면 = Σ Δx·g_upd − ½Σ m|Δv|²; ⊃ 줄은 덜 풀린 걸음만 합). "
      "P2G = 격자 시작 − 입자 시작, 커밋 = RM 뒤 − 앞 (가우스 점 탄성), G2P = 입자 끝 − 격자 끝.\n")
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows or not any("enb" in q for q in rows):
        continue
    print(f"### 셀 {cell(v)} {nm(v)} ({res(s, rows)})\n")
    print("| 몫 | " + " | ".join(f"f{a}~{b}" if a != b else f"f{a}" for a, b in SEG) + " |")
    print("|---|" + "---|" * len(SEG))
    for k, f in KEYS:
        print(f"| {k} | " + " | ".join(fm(segsum(rows, a, b, f), "{:+.3f}") for a, b in SEG) + " |")
    # 자른 노드 / 바닥 근처 나눔
    for rg, lab in (("cut", "자른 노드"), ("uncut", "안 자른 노드"), ("near", "바닥 3 칸 안"), ("far", "바닥 3 칸 밖")):
        print(f"| {lab}: 잔차 · 갱신식 | " + " | ".join(f"{fm(segsum(rows, a, b, lambda e, rg=rg: e['split'][rg]['Wres']), '{:+.2f}')} · {fm(segsum(rows, a, b, lambda e, rg=rg: e['split'][rg]['Rupd']), '{:+.2f}')}" for a, b in SEG) + " |")
    print()

print("\n## 3. 뉴턴 반복별 증분 포텐셜 Φ(y) = ½Σ m/τ²|y − ŷ|² + Σ V Ψ(X(y)) + B (a 판; y_0 = 시작점, y_end = 끝; 늘면 풀이가 포텐셜을 올린 것)\n")
print("| 셀 | 판 | Φ 오른 걸음 수 (y_end > y_0) | Σ (Φ_end − Φ_0) J | Σ (Φ_0 − Φ(ŷ)) J | Σ 탄성 (y_end − y_0) J | 프레임 예 f5·f10·f15 (Φ(ŷ) / Φ_0 / Φ_end) |")
print("|---|---|---|---|---|---|---|")
for v in vs:
    rows, s = load(f"{v}_a")
    if not rows or not any("enb" in q for q in rows):
        continue
    n_up = 0; d_it = 0.0; d_st = 0.0; d_el = 0.0; ex = []
    for q in rows:
        e = q.get("enb")
        if not e or "phi" not in e:
            continue
        ph = e["phi"]; p0, p1 = ph[0]["tot"], ph[-1]["tot"]
        n_up += int(p1 > p0); d_it += p1 - p0; d_st += p0 - e["phi_hat"]["tot"]; d_el += ph[-1]["el"] - ph[0]["el"]
        if q["t"] + 1 in (5, 10, 15):
            ex.append(f"f{q['t'] + 1} {e['phi_hat']['tot']:.2f}/{p0:.2f}/{p1:.2f}")
    print(f"| {cell(v)} | {nm(v)} | {n_up} | {d_it:+.2f} | {d_st:+.2f} | {d_el:+.2f} | {' · '.join(ex)} |")

print("\n## 4. 프레임별 (셀 2.5·3 (2) 판 a, f3~30): 입자 E, ΔE, 잔차·탄성·갱신식·커밋·P2G 탄성·G2P 탄성, 덜 풀림, 자른 노드 수·질량 몫\n")
for v in vs:
    if [x for x in v.split("+") if x not in CELLS] != ["p2"]:
        continue
    rows, s = load(f"{v}_a")
    if not rows:
        continue
    print(f"### 셀 {cell(v)} {nm(v)}\n")
    print("| f | E | ΔE | 잔차 | 고착 | 탄성 C_el | 갱신식 (Δx·g_upd · 감쇠) | 커밋 | P2G 탄성 | G2P 탄성 | 덜 풀림 | 자른 노드 (질량 %) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q in rows:
        f = q["t"] + 1
        e = q.get("enb")
        if not e or "err" in e or not (3 <= f <= 30):
            continue
        st = e["step"]
        print(f"| {f} | {fm(E(q), '{:.2f}')} | {e['dEp']:+.2f} | {st['Wres']:+.2f} | {st['Wcon']:+.2f} | {st['Cel']:+.2f} | {st['Rupd']:+.2f} ({st['Wupd']:+.2f} · {-st['Dnum']:+.2f}) | {e['commit']:+.2f} | "
              f"{e['p2g']['El']:+.2f} | {e['g2p']['El']:+.2f} | {e.get('uf')} | {e['ncut']} ({100 * e['mcut']:.1f}) |")
    print()
