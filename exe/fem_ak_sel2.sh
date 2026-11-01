#!/bin/bash
# 절 AK.25 섞은 선택 적분 (α 0.1·0.25·0.5) 시험 (비측정): ficus 주기, 블록 아핀 시험, lego 21.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS='2 3 4' bash exe/fem_ak_sweep.sh v:sel1,sel25,sel5::1:: ficus > /dev/null 2>&1 &
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_sfcr --dump_f32 --sp vcap --sp_vcap 1e9 --ak_fstep --ak_sel"
G=5
for al in 0.1 0.25 0.5; do for k in stretch shear; do
  EX_ROOT=fem_ak_block bash exe/fem_ab_run.sh ${k}_sel$al $G --data $B/$k.pt --tag ${k}_sel$al $BASE --ak_sela $al < /dev/null > /dev/null 2>&1 &
  G=$([ $G = 5 ] && echo 6 || echo 5)
done; wait; done
wait
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $(for al in 0.1 0.25 0.5; do echo "sel$al:/tmp/dkta/lbours/fem_ak_block/${k}_sel$al/roll_fem.pt"; done) 2>&1 | grep -v Traceback; done
bash exe/fem_ak_period.sh fem_ak_ficus/ficus_sel1_F1 fem_ak_ficus/ficus_sel25_F1 fem_ak_ficus/ficus_sel5_F1
AK_EN=0 AK_W=2 bash exe/fem_ak_sweep.sh v:sel1,sel25,sel5:z:1:: L000,$(seq -f %04g 0 19 | paste -sd,) | grep "스윕 끝"
echo "[끝] $(date +%T)"
