"""절 AK.14 표 (R2 로컬 마운트): 정답 MPM 민감도 (재현 2 회, 격자 141, 서브스텝 416, 격자 71, PIC) 와 우리 절 AK 판 (pqf3, 비측정 3 회 평균) 오차를 한 표에,
그리고 정답 FLIP 섞기 판의 충격 뒤 몸 높이·입자 탄성 곡선을 원래 정답·우리 판과 비교. 사용: python exe/fem_ak_gtsum.py"""
import json
import os

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
T = ["L000"] + ["%04d" % k for k in range(20)]
FAIL = {"0001", "0003", "0016", "0017", "0018"}


def ge(t, v):
    p = f"{L}/fem_ak_gt/{t}_{v}/err.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def ours(t):
    es = []
    for tg in (f"{t}_pqf3_z_r1", f"{t}_pqf3_z_r2", f"{t}_pqf3_z_r3"):
        p = f"{L}/fem_ak_lego/{tg}/summary.json"
        if os.path.exists(p):
            s = json.load(open(p))
            if not s.get("diverged"):
                es.append(100 * s["err_mean"])
    return sum(es) / len(es) if es else None


f_ = lambda q: "-" if q is None else f"{100 * q['err_mean']:.2f}"
print("| 궤적 | 우리 (pqf3) | 정답 재현 1 / 2 | 서브스텝 416 | 격자 141 | 격자 71 | PIC | 우리 ÷ 정답 민감도 최대 (재현·416·141) | 기준 |")
print("|---|---|---|---|---|---|---|---|---|")
for t in T:
    o = ours(t)
    vs = {v: ge(t, v) for v in ("rep", "rep2", "ns416", "ng141", "ng71", "pic")}
    sm = max([100 * vs[v]["err_mean"] for v in ("rep", "rep2", "ns416", "ng141") if vs[v]] or [float("nan")])
    print(f"| {t} | {o:.2f} | {f_(vs['rep'])} / {f_(vs['rep2'])} | {f_(vs['ns416'])} | {f_(vs['ng141'])} | {f_(vs['ng71'])} | {f_(vs['pic'])} | {o / sm:.1f} | {'미달' if t in FAIL else '만족'} |")

print("\n정답 FLIP 섞기 (α 0.95) vs 원래 정답 (재현 2, 진단) vs 우리 (pqf3 진단 _dg): 정답 몸 높이 최저 프레임부터 15 프레임 평균")
print("| 궤적 | FLIP 정답 대비 오차 % | 몸 높이 (m) 원래 정답 / FLIP 정답 / 우리 | 입자 탄성 (J) 원래 정답 / FLIP 정답 / 우리 | 끝 잔류 변형 % 원래 / FLIP / 우리 |")
print("|---|---|---|---|---|")
for t in ("0001", "0003", "0016", "0017", "0018", "0005", "0013"):
    g, fl = ge(t, "rep2"), ge(t, "flip95")
    if not g or not fl or not g.get("diag") or not fl.get("diag"):
        print(f"| {t} | 없음 |")
        continue
    dg, df = g["diag"], fl["diag"]
    i0 = min(range(len(dg)), key=lambda i: dg[i]["height"])
    w = range(i0, min(i0 + 15, len(dg), len(df)))
    av = lambda D, k: sum(D[i][k] for i in w) / len(w)
    p = f"{L}/fem_ak_lego/{t}_pqf3_z_dg_r1/fem.jsonl"
    oh = oe = ores = None
    if os.path.exists(p):
        rs = [json.loads(q) for q in open(p)]
        oh = sum(rs[i]["rigid"][0][1] for i in w if i < len(rs)) / len(w)
        oe = sum(rs[i]["ak"]["ep1"] for i in w if i < len(rs) and isinstance(rs[i].get("ak"), dict)) / len(w)
        ores = rs[-1]["rigid"][0][0]
    print(f"| {t} | {100 * fl['err_mean']:.2f} | {av(dg, 'height'):.3f} / {av(df, 'height'):.3f} / {oh:.3f} | {av(dg, 'ep'):.0f} / {av(df, 'ep'):.0f} / {oe:.0f} | {dg[-1]['res']:.1f} / {df[-1]['res']:.1f} / {ores:.1f} |")
