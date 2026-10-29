#!/bin/bash
# 절 C 3 차: (a) 결과 불변 대조 (비측정, GPU5·6 동시, --dump_pred): 기존 / --cfast / --cfuse 를 c5 N256·c2 N192 각 5 회 -> 완주 수·오차·궤적 차 (같은 설정 반복 바닥 대비)
#            (b) 단독 측정 (GPU0 flock): --cfuse lego c2 N192·c5 N256 각 2 회 (+ 구간 이벤트 1 회), ficus N640 기존·--cfast·--cfuse
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat --no_dump --dump_pred"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --no_dump"
R="env EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
RF="env EX_ROOT=fem_c_ficus bash exe/fem_c_run.sh"
T="env EX_TIMED=1 EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
TF="env EX_TIMED=1 EX_ROOT=fem_c_ficus bash exe/fem_c_run.sh"
# 연기 시험 (GPU5): --cfuse 3 프레임
$R s_c2_fu 5 $LB --lim_sr 2 --N 192 --cfuse --frames 3 < /dev/null > /dev/null 2>&1
echo "[연기] $(tail -1 $W/lbours/fem_c_lego/s_c2_fu/v_fem.log | cut -c1-200)"; [ -f $W/lbours/fem_c_lego/s_c2_fu/EX_DONE ] || { tail -20 $W/lbours/fem_c_lego/s_c2_fu/v_fem.log; echo C3_FAIL; exit 1; }
( for r in 1 2 3 4 5; do for v in base:"" cf:--cfast; do k=${v%%:*}; f=${v#*:}; $R q_c5_n256_${k}_r$r 5 $LB --lim_sr 5 --N 256 $f < /dev/null > /dev/null 2>&1; done; done; echo "[a5] 끝 $(date +%T)" ) &
( for r in 1 2 3 4 5; do $R q_c5_n256_fu_r$r 6 $LB --lim_sr 5 --N 256 --cfuse < /dev/null > /dev/null 2>&1; for v in base:"" cf:--cfast fu:--cfuse; do k=${v%%:*}; f=${v#*:}; $R q_c2_n192_${k}_r$r 6 $LB --lim_sr 2 --N 192 $f < /dev/null > /dev/null 2>&1; done; done; echo "[a6] 끝 $(date +%T)" ) &
for r in 1 2; do
  $T u_c2_n192_fu_r$r 0 $LB --lim_sr 2 --N 192 --cfuse < /dev/null > /dev/null 2>&1
  $T u_c5_n256_fu_r$r 0 $LB --lim_sr 5 --N 256 --cfuse < /dev/null > /dev/null 2>&1
done
$T u_c5_n256_fu_sec 0 $LB --lim_sr 5 --N 256 --cfuse --secprof < /dev/null > /dev/null 2>&1
$T u_c5_n256_cf_sec 0 $LB --lim_sr 5 --N 256 --cfast --secprof < /dev/null > /dev/null 2>&1
$TF u_f_c2_n640_base 0 $FB --lim_sr 2 --N 640 < /dev/null > /dev/null 2>&1
$TF u_f_c2_n640_cf 0 $FB --lim_sr 2 --N 640 --cfast < /dev/null > /dev/null 2>&1
$TF u_f_c2_n640_fu 0 $FB --lim_sr 2 --N 640 --cfuse < /dev/null > /dev/null 2>&1
$TF u_f_c2_n640_fu_r2 0 $FB --lim_sr 2 --N 640 --cfuse < /dev/null > /dev/null 2>&1
wait
for d in $W/lbours/fem_c_lego/[qu]_* $W/lbours/fem_c_ficus/u_*; do echo "$(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-170)"; done
echo C3_DONE $(date +%T)
