"""절 CM54 묶음 rtrender1 표: 판마다 (a) 시뮬 · (b) 가우시안 갱신 · (c) 3DGS 래스터화 · 합계 프레임 ms (프레임 3~, 절 CM50 과 같게), 확인 판 화면 대조.

  python exe/rtrender1_tab.py <출력 디렉토리>
"""
import glob
import json
import os
import sys

import numpy as np

O = sys.argv[1]
B = 16.7
SC = {"L": "레고", "W": "wolf", "F": "ficus", "B": "bread", "M3": "수박"}


def rd(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def st(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():.2f} / {v.max():.1f} / {int((v > B).sum())}" if v.size else "-"


rows, chk = [], []
for d in sorted(glob.glob(os.path.join(O, "*"))):
    if not os.path.isfile(os.path.join(d, "rr.jsonl")):
        continue
    n = os.path.basename(d)
    L = [q for q in rd(os.path.join(d, "rr.jsonl")) if q["t"] >= 2]
    try:
        M = json.load(open(os.path.join(d, "rr_meta.json")))
    except Exception:
        M = {}
    try:
        S = json.load(open(os.path.join(d, "summary.json")))
    except Exception:
        S = {}
    try:
        ap = open(os.path.join(d, "apps_at_start.txt")).read()
        oth = [l for l in ap.splitlines() if l.startswith("카드")]
        oth = oth[0].split()[-1] if oth else "?"
    except Exception:
        oth = "?"
    a = [q["a"] for q in L]; b = [q["b"] for q in L]; c = [q["c"] for q in L]; tt = [q["tot"] for q in L]; bc = [q["b"] + q["c"] for q in L]
    rows.append((n, M, S, a, b, c, bc, tt, oth))
    if os.path.isfile(os.path.join(d, "cmp.json")) or os.path.isfile(os.path.join(d, "rr_chk.jsonl")):
        chk.append((n, M, d))

print("# 절 CM54 rtrender1 -- 시뮬 + 가우시안 갱신 + 3DGS 래스터화 (같은 GPU 루프, CUDA 이벤트)\n")
print("## 판별 (프레임 3~; 칸 = 평균 / 최악 / 16.7 ms 넘는 프레임 수; a 시뮬 = 실행기 재생 GPU ms (fem.jsonl ms 와 같음), b 갱신, c 래스터화, b+c 렌더, 합계 = a + b + c)\n")
print("| 판 | 씬 | 갱신 방식 | 가우시안 | 해상도 | 프레임 (발산) | 오차 % | (a) 시뮬 | (b) 갱신 | (c) 래스터화 | (b)+(c) | 합계 | 시작 카드 다른 프로세스 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n, M, S, a, b, c, bc, tt, oth in rows:
    em = S.get("err_mean_done")
    print(f"| {n} | {M.get('shape', '?')} | {M.get('fast', '?')} | {M.get('NG', '?')} | {M.get('Wd', '?')}x{M.get('H', '?')} | {S.get('frames_done', '?')} ({S.get('diverged')}) | "
          f"{'-' if em is None else round(100 * em, 2)} | {st(a)} | {st(b)} | {st(c)} | {st(bc)} | {st(tt)} | {oth} |")

print("\n## 확인 판 (화면 대조: 실시간 경로 PNG 대 영상 렌더 경로 PNG, 같은 덤프 · 같은 프레임; 갱신 1 은 같은 프레임 갱신 0 과도 비교)\n")
print("| 판 | 대조 프레임 | 영상 경로 대비 최대 픽셀 차 (0~255) | PSNR 최소 dB | 차 있는 픽셀 최대 | 차 > 2 픽셀 최대 | 갱신 0 대비 (위치·공분산 상대·색 최대 차, 화면 u8 최대 차·픽셀) | 극분해 대 CPU SVD 최대 차 (> 1e-3 수) |")
print("|---|---|---|---|---|---|---|---|")
for n, M, d in chk:
    try:
        cm = json.load(open(os.path.join(d, "cmp.json")))
    except Exception:
        cm = {}
    ck = rd(os.path.join(d, "rr_chk.jsonl"))
    f0 = [q for q in ck if "img_max" in q]
    s0 = (f"위치 {max(q['dP'] for q in f0):.1e} · 공분산 상대 {max(q['rC'] for q in f0):.1e}" + (f" · 색 {max(q['dK'] for q in f0):.1e}" if "dK" in f0[0] else "") +
          f" · 화면 {max(q['u8_max'] for q in f0)} / {max(q['u8_ndiff'] for q in f0)} 픽셀") if f0 else "-"
    pp = [q for q in ck if "polar_max" in q]
    s1 = f"{max(q['polar_max'] for q in pp):.1e} ({sum(q['polar_n1e3'] for q in pp)})" if pp else "-"
    print(f"| {n} | {cm.get('n_cmp', '-')}/{cm.get('n', '-')} | {cm.get('max', '-')} | {cm.get('psnr_min', '-') if cm.get('psnr_min') is None else round(cm['psnr_min'], 1)} | "
          f"{cm.get('px_diff_max', '-')} | {cm.get('px_gt2_max', '-')} | {s0} | {s1} |")

print("\n## 16.7 ms 넘는 프레임 (단독 판, 합계 기준) -- 몫\n")
for n, M, S, a, b, c, bc, tt, oth in rows:
    if not n.endswith(("_a", "_b")):
        continue
    ov = [(i + 2, a[i], b[i], c[i], tt[i]) for i in range(len(tt)) if tt[i] > B]
    if not ov:
        continue
    na = sum(1 for q in ov if q[1] > B)
    nr = sum(1 for q in ov if q[1] <= B)
    print(f"- {n}: {len(ov)} 프레임 -- 시뮬 혼자 넘음 {na}, 시뮬은 안 넘는데 렌더를 더해 넘음 {nr}; 렌더 몫 (b+c) 중앙 {np.median([q[2] + q[3] for q in ov]):.2f} ms; "
          f"앞 10: " + ", ".join(f"f{q[0] + 1} {q[1]:.1f}+{q[2]:.2f}+{q[3]:.2f}" for q in ov[:10]))
