#!/bin/bash
# 절 C 4 차: (a) 단독 측정 (GPU0 flock): --cfuse_mode 2 (힘+접촉만 융합) c2 N192·c5 N256, --cfast --fpolar_off c5 N256, 커널 프로파일 (--cfuse 1·2, 프레임 41)
#            (b) 비측정 스윕 (GPU5·6, 3 회씩, --cfast): c 4·6·8 x N256, c5 x N 288·320·384, 적응 서브스텝 (--cfuse_mode 2 --adapt act/stab)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat --no_dump"
R="env EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
T="env EX_TIMED=1 EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
$R s_c5_fu2 5 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --frames 3 < /dev/null > /dev/null 2>&1
$R s_c5_ad 5 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --adapt stab --frames 4 < /dev/null > /dev/null 2>&1
for t in s_c5_fu2 s_c5_ad; do echo "[연기] $t: $(tail -1 $W/lbours/fem_c_lego/$t/v_fem.log | cut -c1-200)"; [ -f $W/lbours/fem_c_lego/$t/EX_DONE ] || { tail -20 $W/lbours/fem_c_lego/$t/v_fem.log; echo C4_FAIL; exit 1; }; done
( for r in 1 2 3; do
    for c in 4 6 8; do $R w_c${c}_n256_cf_r$r 5 $LB --lim_sr $c --N 256 --cfast < /dev/null > /dev/null 2>&1; done
    $R w_c5_n256_adact192_320_r$r 5 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --adapt act --ad_lo 192 --ad_hi 320 < /dev/null > /dev/null 2>&1
    $R w_c5_n256_adact160_256_r$r 5 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --adapt act --ad_lo 160 --ad_hi 256 < /dev/null > /dev/null 2>&1
  done; echo "[b5] 끝 $(date +%T)" ) &
( for r in 1 2 3; do
    for n in 288 320 384; do $R w_c5_n${n}_cf_r$r 6 $LB --lim_sr 5 --N $n --cfast < /dev/null > /dev/null 2>&1; done
    $R w_c5_n256_adstab_r$r 6 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --adapt stab --ad_lo 192 --ad_hi 384 < /dev/null > /dev/null 2>&1
    $R w_c5_n192_adstab_r$r 6 $LB --lim_sr 5 --N 192 --cfuse --cfuse_mode 2 --adapt stab --ad_lo 128 --ad_hi 384 < /dev/null > /dev/null 2>&1
  done; echo "[b6] 끝 $(date +%T)" ) &
$T v_c5_n256_fu2_r1 0 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 < /dev/null > /dev/null 2>&1
$T v_c2_n192_fu2_r1 0 $LB --lim_sr 2 --N 192 --cfuse --cfuse_mode 2 < /dev/null > /dev/null 2>&1
$T v_c5_n256_cfnp_r1 0 $LB --lim_sr 5 --N 256 --cfast --fpolar_off < /dev/null > /dev/null 2>&1
$T p2_c5_n256_fu1 0 $LB --lim_sr 5 --N 256 --cfuse --no_graph --tprof 40 --frames 41 < /dev/null > /dev/null 2>&1
$T p2_c5_n256_fu2 0 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 --no_graph --tprof 40 --frames 41 < /dev/null > /dev/null 2>&1
$T p2_c5_n256_cfnp 0 $LB --lim_sr 5 --N 256 --cfast --fpolar_off --no_graph --tprof 40 --frames 41 < /dev/null > /dev/null 2>&1
wait
python exe/fem_c_table.py $W/lbours/fem_c_lego/[tuvw]_* > $W/lbours/fem_c_lego/table4.md 2>&1; cat $W/lbours/fem_c_lego/table4.md
rclone copy $W/lbours/fem_c_lego/table4.md r2:storage/result/anchorflow/lbours/fem_c_lego -q
echo C4_DONE $(date +%T)
