#!/bin/bash
# 절 AK.30 칸 평균 + 칸 안 분산 분리 에너지 (비측정): 블록 아핀 시험 (γ*, γ 1) -> ficus 주기 -> lego 21 -> h_s 0.5·2 배.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block; SC=L000,$(seq -f %04g 0 19 | paste -sd,)
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_sfcr --dump_f32 --sp vcap --sp_vcap 1e9 --ak_fstep"
G=2
for g in -1 1; do for k in stretch shear; do
  EX_ROOT=fem_ak_block bash exe/fem_ab_run.sh ${k}_cs$g $G --data $B/$k.pt --tag ${k}_cs$g $BASE --ak_cs $g < /dev/null > /dev/null 2>&1 &
  G=$((G + 1)); done; done
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS='5 6' bash exe/fem_ak_sweep.sh v:cs,cs1::1:: ficus > /dev/null 2>&1 &
wait
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt nocap:/tmp/dkta/lbours/fem_ak_block/${k}_nocap/roll_fem.pt cs:/tmp/dkta/lbours/fem_ak_block/${k}_cs-1/roll_fem.pt cs1:/tmp/dkta/lbours/fem_ak_block/${k}_cs1/roll_fem.pt 2>&1 | grep -v "^ \|Traceback"
  for g in -1 1; do echo "${k}_cs$g $(grep -a 'AK.30\]' /tmp/dkta/lbours/fem_ak_block/${k}_cs$g/v_fem.log | head -1) $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_block/${k}_cs$g/v_fem.log | cut -c1-120) $(grep -a '프레임 80/80' /tmp/dkta/lbours/fem_ak_block/${k}_cs$g/v_fem.log | grep -o 'det A 최소 [^ ]*\|σ1/σ3(A) 최대 [^ ]*' | paste -sd' ')"; done; done
bash exe/fem_ak_period.sh fem_ak_ficus/ficus_cs_F1 fem_ak_ficus/ficus_cs1_F1
for d in ficus_cs_F1 ficus_cs1_F1; do echo "$d $(grep -a 'AK.30\]' /tmp/dkta/lbours/fem_ak_ficus/$d/v_fem.log | head -1) $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_ficus/$d/v_fem.log | cut -c1-140)"; done
echo "[블록·ficus 끝] $(date +%T)"
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:cs:z:1:: $SC > /dev/null 2>&1
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:cs:z:1:--N@133:_x2 $SC > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:cs:z:1:--N@534:_x0.5 $SC > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:cs::1:--N@320:_x2 ficus > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:cs::1:--N@1280:_x0.5 ficus > /dev/null 2>&1 &
wait
for v in cs_z_r1 cs_z_x2_r1 cs_z_x0.5_r1; do for t in L000 $(seq -f %04g 0 19); do echo "$t $v $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_lego/${t}_${v}/v_fem.log | cut -c1-140)"; done; done
for v in cs_x2_r1 cs_x0.5_r1; do echo "ficus $v $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_ficus/ficus_${v}/v_fem.log | cut -c1-140)"; done
echo "[끝] $(date +%T)"
