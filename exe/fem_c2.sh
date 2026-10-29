#!/bin/bash
# 절 C 2 차: (a) 검산 (GPU5, 비측정, 그래프 끔 --cchk): 기존 커널과 클램프 α 비트·접촉 대조 -- c2 N192 41 프레임
#            (b) 오차 대조 (GPU6, 비측정): --cfast c5 N256 80 프레임
#            (c) 단독 측정 (GPU0 flock): 기존 / --cfast / 항목별, c2 N192·c5 N256 80 프레임 + 프로파일 (--cfast, 프레임 41)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat --no_dump"
R="env EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
# 연기 시험 (GPU5, 3 프레임): 그래프 끔 + 검산 / 그래프 켬 -- 실패하면 멈춘다
$R s_c2_chk 5 $LB --lim_sr 2 --N 192 --cfast --no_graph --cchk --frames 3 < /dev/null > /dev/null 2>&1
$R s_c2_g 5 $LB --lim_sr 2 --N 192 --cfast --frames 3 < /dev/null > /dev/null 2>&1
for t in s_c2_chk s_c2_g; do echo "[연기] $t: $(tail -2 $W/lbours/fem_c_lego/$t/v_fem.log | cut -c1-300)"; [ -f $W/lbours/fem_c_lego/$t/EX_DONE ] || { echo C2_FAIL; exit 1; }; done
T="env EX_TIMED=1 EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
( $R k_c2_n192_chk 5 $LB --lim_sr 2 --N 192 --cfast --no_graph --cchk --frames 41 < /dev/null > /dev/null 2>&1; echo "[a] 끝 $(date +%T)" ) &
( $R e_c5_n256_cf 6 $LB --lim_sr 5 --N 256 --cfast < /dev/null > /dev/null 2>&1; $R e_c2_n192_cf 6 $LB --lim_sr 2 --N 192 --cfast < /dev/null > /dev/null 2>&1; echo "[b] 끝 $(date +%T)" ) &
$T p1_c2_n192_cf 0 $LB --lim_sr 2 --N 192 --cfast --no_graph --tprof 40 --frames 41 < /dev/null > /dev/null 2>&1
$T p1_c5_n256_cf 0 $LB --lim_sr 5 --N 256 --cfast --no_graph --tprof 40 --frames 41 < /dev/null > /dev/null 2>&1
for r in 1 2; do
  $T t_c2_n192_base_r$r 0 $LB --lim_sr 2 --N 192 < /dev/null > /dev/null 2>&1
  $T t_c2_n192_cf_r$r 0 $LB --lim_sr 2 --N 192 --cfast < /dev/null > /dev/null 2>&1
  $T t_c5_n256_base_r$r 0 $LB --lim_sr 5 --N 256 < /dev/null > /dev/null 2>&1
  $T t_c5_n256_cf_r$r 0 $LB --lim_sr 5 --N 256 --cfast < /dev/null > /dev/null 2>&1
done
for f in vfl_cull sr_fast tlist fpolar; do
  $T t_c2_n192_$f 0 $LB --lim_sr 2 --N 192 --$f < /dev/null > /dev/null 2>&1
done
wait
for d in $W/lbours/fem_c_lego/[ekpt]_*; do echo "$(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-200)"; done
echo C2_DONE $(date +%T)
