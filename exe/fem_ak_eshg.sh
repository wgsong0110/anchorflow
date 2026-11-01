#!/bin/bash
# 절 AK.28 (비측정): (가) ES-FEM + 모래시계 억제 γ 1 -- 블록 아핀 시험, ficus 주기, lego 21.
#                    (나) 상한 없는 판 (nocs) 의 node_h 1.5 배·2 배 -- lego 21, ficus 주기 (셀당 입자 수 진단).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block; SC=L000,$(seq -f %04g 0 19 | paste -sd,)
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_sfcr --dump_f32 --sp vcap --sp_vcap 1e9 --ak_fstep --ak_es --ak_eshg 1"
for k in stretch shear; do EX_ROOT=fem_ak_block bash exe/fem_ab_run.sh ${k}_eshg $([ $k = stretch ] && echo 2 || echo 3) --data $B/$k.pt --tag ${k}_eshg $BASE < /dev/null > $B/${k}_eshg.out 2>&1 & done
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS=4 bash exe/fem_ak_sweep.sh v:eshg::1:: ficus > /dev/null 2>&1 &
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS=5 bash exe/fem_ak_sweep.sh v:nocs::1:--node_h@0.084853:_h15 ficus > /dev/null 2>&1 &
AK_KIND=F AK_EN=0 AK_W=1 AK_CARDS=6 bash exe/fem_ak_sweep.sh v:nocs::1:--node_h@0.113137:_h2 ficus > /dev/null 2>&1 &
wait
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt nocap:/tmp/dkta/lbours/fem_ak_block/${k}_nocap/roll_fem.pt eshg:/tmp/dkta/lbours/fem_ak_block/${k}_eshg/roll_fem.pt 2>&1 | grep -v "^ \|Traceback"; grep -a "롤아웃\]" /tmp/dkta/lbours/fem_ak_block/${k}_eshg/v_fem.log | cut -c1-200; grep -a "프레임 80/80\|프레임 40/80" /tmp/dkta/lbours/fem_ak_block/${k}_eshg/v_fem.log | grep -o "det A 최소 [^ ]*\|σ1/σ3(A) 최대 [^ ]*" | paste -sd' '; done
bash exe/fem_ak_period.sh fem_ak_ficus/ficus_eshg_F1 fem_ak_ficus/ficus_nocs_h15_F1 fem_ak_ficus/ficus_nocs_h2_F1
for d in ficus_eshg_F1 ficus_nocs_h15_F1 ficus_nocs_h2_F1; do echo "$d $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_ficus/$d/v_fem.log | cut -c1-160)"; done
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:eshg:z:1:: $SC > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:nocs:z:1:--node_h@0.042426:_h15 $SC > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:nocs:z:1:--node_h@0.056569:_h2 $SC > /dev/null 2>&1 &
wait
for v in eshg_z nocs_z_h15 nocs_z_h2; do for t in L000 $(seq -f %04g 0 19); do echo "$t $v $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_lego/${t}_${v}_r1/v_fem.log | cut -c1-160)"; done; done
echo "[끝] $(date +%T)"
