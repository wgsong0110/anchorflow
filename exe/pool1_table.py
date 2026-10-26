"""pool1 표: 여섯 실행(다섯 궤적 상태 적합 2, 한 궤적 상태 적합 sf1 2, 한 궤적 풀 pool1 2)의 위치 오차·탄성 에너지·뒤집힘.
사용: python3 exe/pool1_table.py <내려받은 디렉토리 T> <출력 txt> <오차 12 개: ni L000 0004, gu, g1, u1, p1, q1>"""
import json
import sys

T, out = sys.argv[1], sys.argv[2]
E = sys.argv[3:]
# (이름, defstats 칸, defstats 경로 형식, inv 경로 형식, 오차 L000, 0004)
rows = [("일반 GNN · 다섯 궤적 상태 적합 (noinertial)", "ni", T + "/ni/defstats_{t}.txt", T + "/ni/inv_{t}.jsonl", E[0], E[1]),
        ("Graph U-Net · 다섯 궤적 상태 적합 (gunet5_near)", "gu", T + "/g5/defstats_{t}.txt", T + "/g5/inv_{t}.jsonl", E[2], E[3]),
        ("일반 GNN · 한 궤적 상태 적합 (sf1_gnn)", "g1", T + "/sf1_gnn/defstats_{t}.txt", T + "/sf1_gnn/inv_sf1_gnn_{t}.jsonl", E[4], E[5]),
        ("Graph U-Net · 한 궤적 상태 적합 (sf1_gunet)", "u1", T + "/sf1_gunet/defstats_{t}.txt", T + "/sf1_gunet/inv_sf1_gunet_{t}.jsonl", E[6], E[7]),
        ("일반 GNN · 한 궤적 풀 (pool1_gnn)", "p1", T + "/gnn/psi/{t}/defstats_{t}.txt", T + "/gnn/inv_pool1_gnn_{t}.jsonl", E[8], E[9]),
        ("Graph U-Net · 한 궤적 풀 (pool1_gunet)", "q1", T + "/gunet/psi/{t}/defstats_{t}.txt", T + "/gunet/inv_pool1_gunet_{t}.jsonl", E[10], E[11])]
# 덧붙일 행 (pool1 변형, 10-26): 환경 변수 POOL1_EXTRA_ROWS = json [[이름, 칸, defstats 형식, inv 형식, 오차 L000, 0004], ...] ({t} = 궤적)
import os
rows += [tuple(r) for r in json.loads(os.environ.get("POOL1_EXTRA_ROWS", "[]"))]


def ds(p, k):
    r = {}
    for ln in open(p):
        w = ln.split()
        if len(w) == 7 and w[0] in (k, "gt"):
            r[(w[0], int(w[1]))] = (float(w[2]), float(w[3]), int(w[4]), float(w[5]), float(w[6]))
    return r


def inv(p):
    L = [json.loads(l) for l in open(p) if l.strip()]
    return max(q["inv_tets"] for q in L), min(q["det_min"] for q in L)


lines = ["# lego 롤아웃 80 프레임, 정답 시작. 위치 오차 = 평균 |x-x_gt| / EXT (롤아웃 로그 평균). ψ = 고정 FCR 탄성 에너지 밀도 평균 [Pa] (psi_color_lb)",
         "# 뒤집힘 = AF_INV_LOG: 프레임별 뒤집힌 사면체 수 최대, det 최소; det<0 = 프레임 80 입자 수 (psi_color_lb)",
         f"{'실행':<44} {'위치오차 L000':>12} {'위치오차 0004':>12} | {'ψ평균40 L000':>12} {'ψ평균80 L000':>12} {'ψ평균80 0004':>12} | "
         f"{'뒤집힌사면체최대 L000/0004':>22} {'det최소 L000/0004':>18} {'det<0(80) L000/0004':>18}"]
gt = None
for nm, k, dp, ip, eL, e4 in rows:
    try:
        sL, s4 = ds(dp.format(t="L000"), k), ds(dp.format(t="0004"), k)
        gt = gt or (sL.get(("gt", 40)), sL.get(("gt", 80)), s4.get(("gt", 80)))
        iL, i4 = inv(ip.format(t="L000")), inv(ip.format(t="0004"))
        lines.append(f"{nm:<44} {eL:>12} {e4:>12} | {sL[(k, 40)][3]:>12.3e} {sL[(k, 80)][3]:>12.3e} {s4[(k, 80)][3]:>12.3e} | "
                     f"{iL[0]:>10d}/{i4[0]:<10d} {iL[1]:>8.3f}/{i4[1]:<8.3f} {sL[(k, 80)][2]:>8d}/{s4[(k, 80)][2]:<8d}")
    except Exception as ex:
        lines.append(f"{nm:<44} {eL:>12} {e4:>12} | 통계 없음 ({ex})")
if gt and all(gt):
    lines.append(f"{'정답 (MPM)':<44} {'-':>12} {'-':>12} | {gt[0][3]:>12.3e} {gt[1][3]:>12.3e} {gt[2][3]:>12.3e} |")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
