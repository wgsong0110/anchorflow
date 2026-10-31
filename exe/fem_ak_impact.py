"""절 AK.7 충격 구간 진단 (R2 로컬 마운트): 진단 실행 (--ak_en 1 --ai_rigid 1 --rbdiag) 의 프레임별 기록에서 우리 / 정답 비교.
첫 접촉 (정답 최저 입자 < 바닥 + 2 mm) 부터 20 프레임 창에서: 질량중심 수직 속도 곡선 (감속 지연 = 정답이 −0.5 m/s 위로 올라온 프레임과 우리 차),
몸 좌표 높이 최저 (눌림), 입자 탄성 최대 (V_p·ψ, 정답과 같은 식), 에너지 손실 경로 합 (P2G, G2P, 서브스텝, 프레임 경계 -- 입자 적분점 셀 에너지 eq 로).
사용: python exe/fem_ak_impact.py <root> <태그 접미 예 _pqf3_z_dg_r1> 궤적,..."""
import json
import os
import sys

L = os.path.expanduser("~/r2/result/anchorflow/lbours/")
root, sfx = sys.argv[1], sys.argv[2]
ts = sys.argv[3].split(",")
FL = 0.48
print("| 궤적 | 첫 접촉 | 감속 (vz > −0.5) 정답 / 우리 프레임 | 몸 높이 최저 정답 / 우리 | 입자 탄성 최대 정답 / 우리 J | 창 안 KE 손실 P2G / G2P | 서브스텝 (바닥 등) | 프레임 경계 | 창 끝 오차 % | 전체 오차 % |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t in ts:
    d = f"{L}{root}/{t}{sfx}/"
    if not os.path.exists(d + "fem.jsonl"):
        print(f"| {t} | 없음 |")
        continue
    rs = [json.loads(q) for q in open(d + "fem.jsonl")]
    rb = [q.get("rb") for q in rs]
    ak = [q.get("ak") if isinstance(q.get("ak"), dict) else None for q in rs]
    rg = [q.get("rigid") for q in rs]
    c0 = next((i for i, r in enumerate(rb) if r and r["g"]["zmin"] < FL + 0.002), None)
    if c0 is None:
        print(f"| {t} | 접촉 없음 |")
        continue
    w = range(c0, min(c0 + 20, len(rs)))
    dec = lambda k: next((i + 1 for i in w if rb[i] and rb[i][k]["vc"][2] > -0.5), None)
    hmin = lambda k: min(rg[i][k][1] for i in w if rg[i])
    epm_g = max(ak[i]["ep_gt"] for i in w if ak[i])
    epm_o = max(ak[i]["ep1"] for i in w if ak[i])
    p2g = sum(ak[i]["ke_p0"] - ak[i]["ke_n0"] for i in w if ak[i])
    g2p = sum(ak[i]["ke_n1"] - ak[i]["ke_p1"] for i in w if ak[i])
    sub = -sum((ak[i]["ke_n1"] + ak[i]["eq1"] + ak[i]["pe_n1"]) - (ak[i]["ke_n0"] + ak[i]["eq0"] + ak[i]["pe_n0"]) for i in w if ak[i])
    bd = sum(ak[i]["eq1"] - ak[i + 1]["eq0"] for i in w if ak[i] and i + 1 < len(ak) and ak[i + 1])
    ew = 100 * rs[w[-1]]["err"]
    ea = 100 * sum(q["err"] for q in rs) / len(rs)
    print(f"| {t} | {c0 + 1} | {dec('g')} / {dec('o')} | {hmin(1):.3f} / {hmin(0):.3f} | {epm_g:.0f} / {epm_o:.0f} | {p2g:.0f} / {g2p:.0f} | {sub:.0f} | {bd:.0f} | {ew:.1f} | {ea:.2f} |")
