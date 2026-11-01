#!/bin/bash
# 절 AK.27 변 기반 평활 FEM (ES-FEM-T4) 시험 (비측정): 블록 아핀 시험 (신축·전단), ficus 진동 주기, lego 21.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_sfcr --dump_f32 --sp vcap --sp_vcap 1e9 --ak_fstep --ak_es"
for k in stretch shear; do EX_ROOT=fem_ak_block bash exe/fem_ab_run.sh ${k}_es $([ $k = stretch ] && echo 2 || echo 3) --data $B/$k.pt --tag ${k}_es $BASE < /dev/null > $B/${k}_es.out 2>&1 & done
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS=4 bash exe/fem_ak_sweep.sh v:es::1:: ficus > /dev/null 2>&1 &
wait
for k in stretch shear; do tail -3 $B/${k}_es.out; python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt nocap:/tmp/dkta/lbours/fem_ak_block/${k}_nocap/roll_fem.pt es:/tmp/dkta/lbours/fem_ak_block/${k}_es/roll_fem.pt; done
bash exe/fem_ak_period.sh fem_ak_ficus/ficus_es_F1
AK_EN=0 AK_W=2 bash exe/fem_ak_sweep.sh v:es:z:1:: L000,$(seq -f %04g 0 19 | paste -sd,)
echo "[끝] $(date +%T)"
