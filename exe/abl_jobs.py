#!/usr/bin/env python3
"""절 CM183 작업 목록: 장면별 (판 목록 x 반복 3, 반복이 바깥 고리라 부분 결과가 일찍 나옴). 사용: abl_jobs.py <장면,...>  -> 쉼표 목록 (장면:판:반복)"""
import sys
V = dict(ficus="def q1 rho0 rho1 q27 q64 c075 c15 c20 ak", bread="def q1 rho0 rho1 q27 q64 c075 c15 c20 ak",
         vasedeck="def q1 q27 q64 c075 c15 c20", plane="def q1 q27 q64 c075 c15 c20", pillow2sofa="def q1 rho0 rho1 q27 q64 c075 c15 c20", wolf="def q1 rho0 rho1 q27 q64 c075 c15 c20",
         pillow2sofa_bc="defb q27 q64 c075 c15 c20", toast="def q1 rho0 rho1", sand_castle="def q1 rho0 rho1")
if len(sys.argv) > 2 and sys.argv[2] == "E":   # 서브스텝 강인성: 장면 x (ss, ssf) x N, 반복 2 번호 (덤프·모드 분석 없음, 한 번)
    print(",".join(f"{s}:{p}{n}:2" for s in sys.argv[1].split(",") for n in (1, 2, 4, 8, 16, 32, 64, 128) for p in ("ss", "ssf")))
    sys.exit(0)
sc = sys.argv[1].split(",")
NAME = lambda s: s.replace("_bc", "")
print(",".join(f"{NAME(s)}:{v}:{r}" for r in (1, 2, 3) for s in sc for v in V[s].split()))
