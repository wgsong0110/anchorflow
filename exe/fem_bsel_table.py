"""절 AK.20 표: B-스플라인 요소 판 vs 기본 판 (cms, 사면체) -- 오차, 정답 최대 압축 (몸 높이 최저) 뒤 15 프레임 평균 몸 높이 차·입자 탄성 (우리 / 정답).
기본 판 몸 높이·탄성은 절 AK 진단 실행 (pqf3_z_dg, θ 없음) 대신 오차만 비교 (cms 진단 실행은 없음). 사용: python exe/fem_bsel_table.py"""
import json
import os

L = os.path.expanduser("~/r2/result/anchorflow/lbours")
TS = ["0001", "0003", "0015", "0016", "0017", "0018", "0005", "0013"]
print("| 궤적 | 기본 판 (사면체, cms) 오차 % | B-스플라인 요소 오차 % | 발산 | 최대 압축 뒤 몸 높이 차 (cm) | 입자 탄성 우리 / 정답 (J) |")
print("|---|---|---|---|---|---|")
for t in TS:
    s0 = f"{L}/fem_ak_lego/{t}_cms_z_r1/summary.json"
    e0 = 100 * json.load(open(s0))["err_mean_done"] if os.path.exists(s0) else float("nan")
    d = f"{L}/fem_ak_bsel/{t}_b05"
    if not os.path.exists(d + "/summary.json"):
        print(f"| {t} | {e0:.2f} | 진행 중 | | | |")
        continue
    s = json.load(open(d + "/summary.json"))
    rs = [json.loads(q) for q in open(d + "/fem.jsonl")]
    bh = [(r["t"], r["bh"], r["ep"]) for r in rs if "bh" in r]
    i0 = min(range(len(bh)), key=lambda i: bh[i][1][1]) if bh else 0
    w = bh[i0:i0 + 8]                                                                  # 2 프레임 간격 기록 -> 8 개 = 15 프레임
    dh = 100 * sum(q[1][0] - q[1][1] for q in w) / max(len(w), 1)
    eo = sum(q[2][0] for q in w) / max(len(w), 1)
    eg = sum(q[2][1] for q in w) / max(len(w), 1)
    print(f"| {t} | {e0:.2f} | {100 * s['err_mean_done']:.2f} | {s['diverged'] or '-'} | {dh:+.1f} | {eo:.0f} / {eg:.0f} |")
