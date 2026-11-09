"""절 CM90 묶음 wmconv 표 (로컬, R2 lbours/wmconv/<판>/wmconv_stat.json): python exe/wmconv_tab.py <자료 디렉토리> cA_1 cC_1 cC_2 -> markdown.
정답 값은 절 CM80 표 (손상 f5 4.41 · f6 6.94 · f8 7.08 · 끝 6.51, 손상 입자 J·Jp f8 1.19·1.92·5.64 · f100 1.16·1.84·5.53); 정답 손상 몫은 gt_jp 로 다시 셈 (같은 값)."""
import json
import sys

D, runs = sys.argv[1], sys.argv[2:]
NM = {"cA": "H9 그대로", "cC": "수렴 판 (예산 끔 · xit 문 끔 · 뉴턴 20 · PCG 40 · tolP 1e-2)"}
q = lambda r: " · ".join(f"{v:.2f}" for v in r["JJp_q"]) if r.get("JJp_q") else "-"
print("| 판 | 오차 % | 손상 f4 · f5 · f6 · f8 · 끝 % | 정답 IoU f5 · f8 · 끝 | 손상 입자 J·Jp 10·50·90 % f5 | f8 | 끝 | f5 손상 입자 높이 중앙 m |")
print("|---|---|---|---|---|---|---|---|")
print("| 정답 (GF) | - | 0.03 · 4.41 · 6.94 · 7.08 · 6.51 | 1 | - | 1.19 · 1.92 · 5.64 | 1.16 · 1.84 · 5.53 | - |")
for n in runs:
    s = json.load(open(f"{D}/{n}/wmconv_stat.json")); f = s["fr"]; L = str(s["L"])
    d = " · ".join(f"{100 * f[t]['dmg']:.2f}" for t in ("4", "5", "6", "8", L))
    iou = " · ".join(f"{f[t].get('iou') or 0:.3f}" for t in ("5", "8", L))
    z = f["5"].get("z_med"); z = f"{z:.2f}" if z is not None else "-"
    print(f"| {NM[n[:2]]} {n} | {100 * s['err']:.2f} | {d} | {iou} | {q(f['5'])} | {q(f['8'])} | {q(f[L])} | {z} |")
