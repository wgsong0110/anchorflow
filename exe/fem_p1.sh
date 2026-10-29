#!/bin/bash
# 절 P 1 차 (사용자 10-29 "해"): lag 명시적 판 + 서브스텝 속도 바닥 (--floor vel, 질량 비율 고착 + 비관통) 에 K 서브스텝마다 재평균 (--rep K)
#   full = G2P (입자 x·v·F) -> 재배치 P2G (노드 질량·속도·셀 F̄), vel = 노드 속도만 재평균. 측정은 GPU0 단독 순차 (fem_ex_run.sh 가 7 장 빌 때까지 대기)
#   lego L000 80 프레임: N 64·96·128·192·240 x K 1·2·4·8·16 (full, vel)
#   ficus 125 프레임 (같은 코드 경로, 바닥 없음): N 512·640·768 x K 끔·128·64·32 (full), 640·768 x 32·64·128 (vel), 주기 (pred.pt)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
LB="--mode lag --opt --mass tet --no_dump --floor vel"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
# 0. 연기 시험 (3 프레임, 비측정)
export EX_ROOT=fem_p_lego
for M in full vel; do EX_NOWAIT=1 $R p_smoke_$M 0 $LB --N 96 --rep 8 --rep_mode $M --frames 3; tail -4 $W/lbours/fem_p_lego/p_smoke_$M/v_fem.log; done
# 1. lego
for M in full vel; do for N in 64 96 128 192 240; do for K in 1 2 4 8 16; do
  $R p_l_n${N}_k${K}_$M 0 $LB --N $N --rep $K --rep_mode $M
done; done; done
python exe/fem_p_table.py $W/lbours/fem_p_lego/p_l_* | tee $W/lbours/fem_p_lego/table1.md
rclone copy $W/lbours/fem_p_lego r2:storage/result/anchorflow/lbours/fem_p_lego -q
# 2. ficus
export EX_ROOT=fem_p_ficus
for N in 512 640 768; do
  $R p_f_n${N}_k0 0 $FB --N $N
  for K in 128 64 32; do $R p_f_n${N}_k${K}_full 0 $FB --N $N --rep $K --rep_mode full; done
done
for N in 640 768; do for K in 128 64 32; do $R p_f_n${N}_k${K}_vel 0 $FB --N $N --rep $K --rep_mode vel; done; done
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_p_ficus/p_f_*/pred.pt | tee $W/lbours/fem_p_ficus/period1.txt
python exe/fem_p_table.py $W/lbours/fem_p_ficus/p_f_* | tee $W/lbours/fem_p_ficus/table1.md
rclone copy $W/lbours/fem_p_ficus r2:storage/result/anchorflow/lbours/fem_p_ficus -q
echo P1_DONE
