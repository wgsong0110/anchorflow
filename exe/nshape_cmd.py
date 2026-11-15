#!/usr/bin/env python3
"""절 CM186 (PG 경로): 새 형상 PG 장면의 우리 판 명령 = PG 장면용 기본 판 (bread BA_1 명령 = 레고·wolf·bread 와 같은 H9 사슬, R2 lbours/wolfcv/cm157/BA_1/cmd.txt)
에서 장면 의존 부분만 바꾼다 (exe/pr2_grid.py 의 plane <- bread 명령과 같은 방식). GF 장면용 감싸개 (hexff, 칸·상자 상한) 는 넣지 않는다.
바꾸는 것: AF_SP_FR, --rr_shape/--rr_traj/--rr_cfg (렌더 훅 --rr pg 그대로), --data, --frames, --tag, --out,
바닥 (--floor none -> 레고 LA_1 의 --floor vel --fl_bar --fl_bst 1 --y0eb --hgt --bar_cnt; 이 장면들은 sticky 바닥으로 떨어진다),
K > 1 이면 --ksplit 3 --ks_big 뺌 (gfs_cmd 와 같은 규칙), 덤프 (--dump --dump_F, 영상·위상 지표용; NS_NODUMP=1 이면 안 붙임), NS_DROP (단독 시간 판: 진단 플래그 빼기).
  nshape_cmd.py BA_1_cmd.txt DATA FRAMES TAG OUT RR_TRAJ RR_CFG K"""
import os, re, sys
src, data, fr, tag, out, rtraj, rcfg, K = sys.argv[1:9]
c = " ".join(open(src).read().split())
c = re.sub(r"AF_SP_FR=1,\d+", f"AF_SP_FR=1,{fr}", c)
assert "--rr pg" in c
c = re.sub(r"--rr_shape \S+", f"--rr_shape {tag}", c); c = re.sub(r"--rr_traj \S+", f"--rr_traj {rtraj}", c); c = re.sub(r"--rr_cfg \S+", f"--rr_cfg {rcfg}", c)
c = re.sub(r"--data \S+", f"--data {data}", c); c = re.sub(r"--frames \d+", f"--frames {fr}", c); c = re.sub(r"--tag \S+", f"--tag {tag}", c)
assert "--floor none" in c; c = c.replace("--floor none", "--floor vel --fl_bar --fl_bst 1 --y0eb --hgt --bar_cnt")
c = re.sub(r"--K \d+", f"--K {int(K)}", c)
if os.environ.get("NS_CG4", "1") == "1":                                         # 절 CM189 기본 판: 걸음마다 PCG 2 -> 4 (레고 세팅 chair 다시 풀기 71 -> 0·25.2 -> 10.3 ms; 레고·wolf·plane·vasedeck·pillow2sofa 회귀 없음); NS_CG4=0 이면 옛 판
    assert re.search(r"--cg \d+", c), "BA_1 명령에 --cg 없음"
    c = re.sub(r"--cg \d+", "--cg 4", c)
if int(K) > 1:
    c = c.replace(" --ksplit 3", "").replace(" --ks_big", "")
for f in os.environ.get("NS_DROP", "").split():
    c = re.sub(r"(^| )" + re.escape(f) + r"( (?!--)\S+)?(?= |$)", " ", c)
c = re.sub(r"--out \S+$", "", c).strip()
if os.environ.get("NS_NODUMP", "0") != "1":
    c += " --dump --dump_F"
print(" ".join(c.split()) + f" --out {out}")
