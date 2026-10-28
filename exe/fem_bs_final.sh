#!/bin/bash
# 절 M 마무리: (1) 국소 안정 걸음 진단 (입자 / 셀 대표) (2) 셀 대표 큰 N 시험 (3) 최종 판 F 덤프 + 3DGS 렌더 (lego 입자 적분점 N96, ficus N512 f32 + ψ 색)
# (4) ficus 고정 경계 띠 ψ (절 L 과 같은 스크립트). GPU0 단독 순차.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
PHYS="--data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
D=$W/lbours/fem_bs_lego/diag; mkdir -p $D
[ -f $D/diag_lego.txt ] || CUDA_VISIBLE_DEVICES=0 python -u exe/fem_bs_diag.py < /dev/null > $D/diag_lego.txt 2>&1
[ -f $D/diag_ficus.txt ] || CUDA_VISIBLE_DEVICES=0 python -u exe/fem_bs_diag.py --data $FD --node_h 0.056569 --region_density --cfg_bc < /dev/null > $D/diag_ficus.txt 2>&1
cat $D/diag_lego.txt $D/diag_ficus.txt; rclone copy $D r2:storage/result/anchorflow/lbours/fem_bs_lego/diag -q
export EX_ROOT=fem_bs_lego
for N in 1024 4096; do $R s3_c8_nofl_n$N 0 $B --no_dump --quad cell --K 8 --N $N --floor none --frames 5; done
SHAPE=lego EX_ROOT=fem_bs_lego bash exe/fem_ex_final.sh m_final_F 0 $B --N 96
echo FINAL_LEGO_DONE
export EX_ROOT=fem_bs_ficus
for N in 4096 16384; do $R s3_c8_n$N 0 $B --no_dump --quad cell --K 8 --N $N $PHYS --frames 3; done
SHAPE=ficus EX_ROOT=fem_bs_ficus bash exe/fem_ex_final.sh mf_final_F 0 $B --N 512 --dump_f32 $PHYS
O=$W/lbours/fem_bs_ficus; mkdir -p $O/band
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_rt_psi_band.py --data $FD --t 1,2,3,40,125 --out $O/band/band.json \
   --run "명시적 B-스플라인 (N512)=$O/mf_final_F/roll_fem.pt" --run "명시적 사면체 등분 (lag N768)=$W/lbours/fem_ex_ficus/f_final_F/roll_fem.pt" \
   --run "암시적 실시간+수정판정=$W/lbours/fem_rt_ficus/rt_tmix_fix/roll_fem.pt" < /dev/null > $O/band/band.log 2>&1
tail -8 $O/band/band.log | cut -c1-900
rclone copy $O/band r2:storage/result/anchorflow/lbours/fem_bs_ficus/band -q
cat $W/lbours/fem_bs_lego/cmp2.txt
for d in $W/lbours/fem_bs_lego/s3_* $W/lbours/fem_bs_ficus/s3_* $W/lbours/fem_bs_lego/m_final_F $W/lbours/fem_bs_ficus/mf_final_F; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c30-200) $(grep -a "발산\]" $d/v_fem.log|head -1|cut -c1-80)"; done
ls $W/lbours/fem_bs_lego/m_final_F/vj $W/lbours/fem_bs_ficus/mf_final_F/vj $W/lbours/fem_bs_ficus/mf_final_F/psi
echo FINAL_DONE
