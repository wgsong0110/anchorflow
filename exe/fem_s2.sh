#!/bin/bash
# 절 S 2 차: (a) 이동 제한 끈 판의 큰 N (1024·2048·4096, lego 3 회·ficus 2 회, 5 프레임) GPU 1~6 동시에 (비측정) -> 표
#            (b) 단독 측정 (GPU0, 7 장 모두 빌 때까지 대기 -- fem_ex_run.sh 기본): 제한 켠 판 lego N192·ficus N1024 x 3 (발산 전 프레임 ms, ficus 는 앞 1 프레임만 뺀다),
#                커널별 시간 (--tprof, 그래프 끔) lego 프레임 3·ficus 프레임 2, 절 R 판 같은 N 대조
#            (c) 질량·운동량 보존 요약
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --p2g bs --no_dump --floor vel --rstat --bsdiag -1"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --p2g bs --no_dump --data $FD --tag F_pg_v0 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1"
export EX_NOWAIT=1
G=1
for N in 1024 2048 4096; do
  for v in "gb --g2p bary" "gs --g2p bs"; do
    set -- $v; t=$1; shift
    ( for r in 1 2 3; do EX_ROOT=fem_s_lego $R s2_l_${t}_n${N}_r$r $G $LB "$@" --frames 5 --N $N; done
      for r in 1 2; do EX_ROOT=fem_s_ficus $R s2_f_${t}_n${N}_r$r $G $FB "$@" --frames 5 --N $N; done ) < /dev/null > /dev/null 2>&1 &
    G=$((G + 1))
  done
done
wait
unset EX_NOWAIT
python exe/fem_r_table.py $W/lbours/fem_s_lego/s2_l_* > $W/lbours/fem_s_lego/table2.md
python exe/fem_r_table.py $W/lbours/fem_s_ficus/s2_f_* > $W/lbours/fem_s_ficus/table2.md
echo "[2a 끝] $(date +%T)"; cat $W/lbours/fem_s_lego/table2.md $W/lbours/fem_s_ficus/table2.md | cut -d'|' -f2,5,6,7,8
for r in 1 2 3; do
  EX_ROOT=fem_s_lego $R t_l_gbL_n192_r$r 0 $LB --g2p bary --lim 0.3 --lim_mode box --N 192
  EX_ROOT=fem_s_lego $R t_l_gsL_n192_r$r 0 $LB --g2p bs --lim 0.3 --lim_mode box --N 192
  EX_ROOT=fem_s_ficus $R t_f_gsL_n1024_r$r 0 $FB --g2p bs --lim 0.3 --lim_mode box --N 1024 --frames 125 --warm 1
done
EX_ROOT=fem_s_lego $R p_l_gsL_n192 0 $LB --g2p bs --lim 0.3 --lim_mode box --N 192 --frames 4 --no_graph --tprof 2
EX_ROOT=fem_s_lego $R p_l_R_n192 0 --mode lag --opt --mass tet --no_dump --floor vel --rstat --lim 0.3 --lim_mode box --N 192 --frames 4 --no_graph --tprof 2
EX_ROOT=fem_s_ficus $R p_f_gsL_n1024 0 $FB --g2p bs --lim 0.3 --lim_mode box --N 1024 --frames 3 --no_graph --tprof 1
EX_ROOT=fem_s_ficus $R p_f_R_n1024 0 --mode lag --opt --mass tet --no_dump --data $FD --tag F_pg_v0 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --lim 0.3 --lim_mode box --N 1024 --frames 3 --no_graph --tprof 1
python exe/fem_r_table.py $W/lbours/fem_s_lego/t_l_* > $W/lbours/fem_s_lego/table3.md
python exe/fem_r_table.py $W/lbours/fem_s_ficus/t_f_* > $W/lbours/fem_s_ficus/table3.md
python exe/fem_s_cons.py $W/lbours/fem_s_lego/s_smoke_gs* $W/lbours/fem_s_lego/s_l_g*L_n*_r* $W/lbours/fem_s_lego/t_l_* $W/lbours/fem_s_ficus/s_f_g*L_n* $W/lbours/fem_s_ficus/t_f_* > $W/lbours/fem_s_lego/cons.txt 2>&1
rclone copy $W/lbours/fem_s_lego r2:storage/result/anchorflow/lbours/fem_s_lego -q
rclone copy $W/lbours/fem_s_ficus r2:storage/result/anchorflow/lbours/fem_s_ficus -q
echo S2_DONE $(date +%T)
