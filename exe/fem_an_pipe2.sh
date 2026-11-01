#!/bin/bash
# 절 AN 파이프 2 (비측정): 고친 판 (고착 시작점 √k, 멀티그리드 ω = 1/λ_max) 8 프레임 시험 -> 떨어지는 블록 80 프레임 x 적분기 3 -> lego 21 (mK2, 셀 1, 멀티그리드).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an; M=$R/marks; D=/tmp/dkta/ak_drop2; mkdir -p $M
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; df -h /tmp | tail -1
AN_GPUS="${AN_GPUS:-3 5}" bash exe/fem_an_run.sh smk2 0001,drop1,ficus --frames 8 --pc mg
for t in 0001 drop1 ficus; do cp $R/${t}_smk2/log.txt $M/${t}_smk2.log; done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
for vk in mK2 bK2 mK1; do
  case $vk in mK2) X="--integ mid --K 2";; bK2) X="--integ be --K 2";; mK1) X="--integ mid --K 1";; esac
  AN_GPUS="${AN_GPUS:-3 5}" bash exe/fem_an_run.sh ${vk}mg drop1 $X --pc mg &
  sleep 10
done; wait
python exe/fem_ak_drop_ana.py $D/init.pt mpm:$D/mpm/x240.pt:0.0041666667 $(for vk in mK2 bK2 mK1; do echo "$vk:$R/drop1_${vk}mg/pred.pt:0.0166666667"; done) > $M/drop_ana.txt 2>&1
for vk in mK2 bK2 mK1; do cp $R/drop1_${vk}mg/log.txt $M/drop1_${vk}mg.log; done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
AN_GPUS="${AN_GPUS:-3 5}" bash exe/fem_an_s2.sh mK2mg "1" --integ mid --K 2 --pc mg
echo "[pipe2 끝] $(date +%T)" > $M/pipe2_done; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
