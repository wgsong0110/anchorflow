#!/usr/bin/env python3
"""절 CM183 작업 목록: 장면별 (판 목록 x 반복 3, 반복이 바깥 고리라 부분 결과가 일찍 나옴). 사용: abl_jobs.py <장면,...>  -> 쉼표 목록 (장면:판:반복)"""
import sys
V = dict(ficus="def q1 rho0 rho1 q27 q64 c075 c15 c20 ak", bread="def q1 rho0 rho1 q27 q64 c075 c15 c20 ak",
         vasedeck="def q1 q27 q64 c075 c15 c20", plane="def q1 q27 q64 c075 c15 c20", pillow2sofa="def q1 rho0 rho1 q27 q64 c075 c15 c20", wolf="def q1 rho0 rho1 q27 q64 c075 c15 c20",
         toast="def q1 rho0 rho1", sand_castle="def q1 rho0 rho1")
sc = sys.argv[1].split(",")
print(",".join(f"{s}:{v}:{r}" for r in (1, 2, 3) for s in sc for v in V[s].split()))
