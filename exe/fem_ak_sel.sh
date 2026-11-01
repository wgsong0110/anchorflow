#!/bin/bash
# 절 AK.25 선택 적분 판 시험 (비측정): 블록 아핀 시험 (신축·전단), ficus 진동 주기, lego 21.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_sfcr --dump_f32 --sp vcap --sp_vcap 1e9 --ak_fstep --ak_sel"
for k in stretch shear; do EX_ROOT=fem_ak_block bash exe/fem_ab_run.sh ${k}_sel $([ $k = stretch ] && echo 2 || echo 3) --data $B/$k.pt --tag ${k}_sel $BASE < /dev/null > /dev/null 2>&1 & done
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS=4 bash exe/fem_ak_sweep.sh v:sel::1:: ficus > /dev/null 2>&1 &
wait
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt sel:/tmp/dkta/lbours/fem_ak_block/${k}_sel/roll_fem.pt; done
bash exe/fem_ak_period.sh fem_ak_ficus/ficus_sel_F1
AK_EN=0 AK_W=2 bash exe/fem_ak_sweep.sh v:sel:z:1:: L000,$(seq -f %04g 0 19 | paste -sd,)
echo "[끝] $(date +%T)"
