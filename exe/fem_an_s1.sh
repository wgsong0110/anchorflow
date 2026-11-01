#!/bin/bash
# 절 AN 1 단계 (비측정): 적분기·걸음 수 비교 -- 블록 아핀 (신축·전단), 떨어지는 블록, ficus 잎 끝 주기, 셀 = 정답 dx 1 배.
# 판: mK2 (암시적 중점, 2 걸음), bK2 (후퇴 오일러, 2 걸음), mK1 (중점, 1 걸음). 사용: bash exe/fem_an_s1.sh [판 목록]
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
df -h /tmp | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
B=/tmp/dkta/ak_block; D=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_an; M=$R/marks; mkdir -p $M
declare -A VS=( [mK2]="--integ mid --K 2" [bK2]="--integ be --K 2" [mK1]="--integ mid --K 1" )
VL=${1:-"mK2 bK2 mK1"}
GL=(${AN_GPUS:-1 2 3 4 5 6}); j=0
for vk in $VL; do
  G2="${GL[$((j % ${#GL[@]}))]} ${GL[$(((j + 1) % ${#GL[@]}))]}"; j=$((j + 2))
  AN_GPUS="$G2" bash exe/fem_an_run.sh $vk stretch,shear,drop1,ficus ${VS[$vk]} > $R/s1_$vk.out 2>&1 &
  sleep 20
done
wait
{
for k in stretch shear; do python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $(for vk in $VL; do echo "$vk:$R/${k}_$vk/pred.pt"; done) 2>&1 | grep -v "^ \|Traceback\|Warning"; done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $(for vk in $VL; do echo "$R/ficus_$vk/pred.pt"; done) 2>&1 | grep -v Warning
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 $(for vk in $VL; do echo "$vk:$R/drop1_$vk/pred.pt:0.0166666667"; done) 2>&1 | grep -v Warning
for t in stretch shear drop1 ficus; do for vk in $VL; do echo "${t}_$vk $(grep -a '롤아웃' $R/${t}_$vk/log.txt | tail -1 | cut -c1-200)"; done; done
} > $R/s1_ana.txt 2>&1
cp $R/s1_*.out $R/s1_ana.txt $M/; echo "[s1 끝] $(date +%T)" > $M/s1_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
