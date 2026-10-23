"""attnring_bench.sh 결과 표 (사용자 10-23). python exe/attnring_report.py <출력디렉토리>

반복당 시간 = Adam 한 반복 (순전파 + 증분 포텐셜 + 역전파 + 자르기 + 걸음), cuda 동기화로 잰 것.
  워밍업 버림: 프레임 0 의 앞 10 회 (첫 반복은 K 링 이웃 목록 구성·할당 포함), 프레임 1 의 앞 2 회 (새 격자) -> 나머지 모아 중앙값.
프레임당 추정 = 실제 설정 ov_roll 2000 × 중앙값 + 프레임 1 의 반복 밖 시간 (맞춤 시간 - 반복 합).
"""
import json
import os
import re
import sys

import numpy as np

O = sys.argv[1]
ROLL_REAL = 2000
CFGS = ["sgnn4", "sgnn8", "lattn4_k1", "lattn4_k2", "lattn4_k3", "lattn8_k1", "lattn8_k2", "lattn8_k3"]
print(f"{'구성':<11} {'층':>2} {'K':>2} {'노드':>6} {'이웃/노드':>9} {'간선':>9} {'반복 ms(중앙)':>13} {'p10-p90':>13} {'n':>3} "
      f"{'최대메모리 MiB':>14} {'파라미터':>9} {'맞춤 s/프레임(측정 반복 수)':>26} {'프레임당 추정 s (2000회)':>22} {'이웃구성 ms':>10} {'단독':>4}")
for k in CFGS:
    d = os.path.join(O, k)
    f = os.path.join(d, "iter_time.jsonl")
    if not os.path.exists(f):
        print(f"{k:<11} 없음")
        continue
    rows = [json.loads(l) for l in open(f)]
    rows.sort(key=lambda r: r["t"])
    skip = [10, 2]
    its = []
    for j, r in enumerate(rows):
        its += r["iters"][skip[min(j, 1)]:]
    its = np.array(its) * 1e3
    med = float(np.median(its))
    rec = json.load(open(os.path.join(d, "records.json")))
    r1 = rows[-1]
    over = rec[-1]["sec_fit"] - sum(r1["iters"])
    est = ROLL_REAL * med / 1e3 + over
    rs = r1.get("ring_stats") or {}
    nb = f"{rs['avg']:.2f}" if rs else "-"
    E = rs.get("E", "-") if rs else "-"
    Mn = rs.get("M", "-") if rs else "-"
    if not rs:      # sgnn: 메시지 패싱 이웃 = 시작 로그의 1 링
        txt = open(os.path.join(d, "run.log"), errors="ignore").read()
        mm = re.search(r"\[어텐션 이웃 수\] 노드당 최소 \d+ 평균 ([\d.]+) 최대 \d+ \(간선 (\d+)\)", txt)
        if mm:
            nb, E = mm.group(1), mm.group(2)
    ex = open(os.path.join(d, "exclusive.txt")).read().split("pids_on_card=")[1].split()[0] if os.path.exists(os.path.join(d, "exclusive.txt")) else "?"
    print(f"{k:<11} {r1['layers']:>2} {r1['attn_ring'] if r1['gnn_layer'] == 'lattn' else '-':>2} {Mn:>6} {nb:>9} {E:>9} {med:>13.1f} "
          f"{np.percentile(its, 10):>6.1f}-{np.percentile(its, 90):<6.1f} {its.size:>3} {max(r['peak_alloc_mib'] for r in rows):>14.0f} "
          f"{r1['n_param']:>9} {rec[-1]['sec_fit']:>18.1f}({len(r1['iters'])}) {est:>22.0f} "
          f"{(rows[0].get('ring_stats') or {}).get('build_ms', float('nan')):>10.1f} {ex:>4}")
    print("    프레임별 중앙값 ms: " + ", ".join(f"t{r['t']}: {1e3 * float(np.median(r['iters'][skip[min(j, 1)]:])):.1f}" for j, r in enumerate(rows))
          + f"; 첫 반복 ms {1e3 * rows[0]['iters'][0]:.0f}; 예약 메모리 최대 {max(r['peak_reserved_mib'] for r in rows):.0f} MiB")
