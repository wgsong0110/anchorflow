#!/bin/bash
# 절 AN 실시간 판 단위 시험 예산 탐색 (비측정, 카드 AN_GPUS): 블록 신축·전단 · 떨어지는 블록 · ficus, 셀 AN_CELL(1), 판 = 이름:플래그 목록 (AN_VS, 플래그 안 빈칸은 _).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-ru1}; B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
C=${AN_CELL:-1}
VS=${AN_VS:-"n1c5:--newton_1_--cg_5 n2c10:--newton_2_--cg_10 n2c20:--newton_2_--cg_20 n2c50:--newton_2_--cg_50 n3c50:--newton_3_--cg_50 bn2c20:--newton_2_--cg_20_--integ_be"}
GL=(${AN_GPUS:-4 5 6}); j=0
run1() { G=$1; t=$2; v=$3; A="$4"; O=$R/${t}_$v; mkdir -p $O
  case $t in ficus) DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp";;
    stretch|shear) DA="--data $B/$t.pt --frames 80 --floor none";; drop1) DA="--data $D/drop_f1.pt --frames 80 --floor vel";; esac
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --out $O --tag $t $DA --cell $C --K 1 --integ mid $A --dump < /dev/null > $O/log.txt 2>&1
  rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/$NM/${t}_$v -q; }
for e in $VS; do v=${e%%:*}; A=$(echo ${e#*:} | tr '_' ' ')
  for t in stretch shear drop1 ficus; do ( run1 ${GL[$((j % ${#GL[@]}))]} $t $v "$A" ) & j=$((j + 1)); sleep 3; done
  wait
done
VL=$(for e in $VS; do echo ${e%%:*}; done)
{
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $(for v in $VL; do [ -f $R/${k}_$v/pred.pt ] && echo "$v:$R/${k}_$v/pred.pt"; done) 2>&1 | grep -v "^ \|Traceback\|Warning\|처음 10"; done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $(for v in $VL; do [ -f $R/ficus_$v/pred.pt ] && echo $R/ficus_$v/pred.pt; done) 2>&1 | grep -v Warning
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 $(for v in $VL; do [ -f $R/drop1_$v/pred.pt ] && echo "$v:$R/drop1_$v/pred.pt:0.0166666667"; done) 2>&1 | grep -v Warning | cut -c1-200
for t in stretch shear drop1 ficus; do for v in $VL; do echo "${t}_$v $(grep -a '롤아웃\|rror' $R/${t}_$v/log.txt | tail -1 | cut -c1-170)"; done; done
} > $M/${NM}_unit.md 2>&1
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
