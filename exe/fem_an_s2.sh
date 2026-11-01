#!/bin/bash
# 절 AN 2 단계 (비측정): lego L000 + 홀드아웃 20 (0000 에서 0019) x 셀 배수. 사용: bash exe/fem_an_s2.sh <판 접두> "<셀 배수 목록>" [추가 인자]
# 판 이름 = <접두>c<배수> (예 mK2c1). 카드 AN_GPUS 에 돌려 가며, 카드당 동시 AN_PER 개.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
df -h /tmp | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
PF=$1; CL=$2; shift 2; EXTRA="$*"
R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M
SC=L000,$(seq -f %04g 0 19 | paste -sd,)
VL=""
for c in $CL; do
  VK=${PF}c$(echo $c | tr '.' 'p'); VL="$VL,$VK"
  bash exe/fem_an_run.sh $VK $SC --cell $c $EXTRA > $R/s2_$VK.out 2>&1
done
python exe/fem_an_table.py $R ${VL#,} $SC > $M/s2_${PF}_table.md 2>&1
cp $R/s2_${PF}*.out $M/ 2>/dev/null; echo "[s2 $PF 끝] $(date +%T)" > $M/s2_${PF}_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
