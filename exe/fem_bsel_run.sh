#!/bin/bash
# 절 AK.20: B-스플라인 요소 판 (exe/fem_bsel_sim.py, 비측정) -- 미달 6 + 대조 2 를 카드 1~6 에 나눠. 출력 /tmp/dkta/lbours/fem_ak_bsel/<궤적>_<판> + R2.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
VK=${BS_V:-b05}; EXTRA=${BS_X:-}; TS=${BS_TS:-"0001 0003 0015 0016 0017 0018 0005 0013"}
i=0
for t in $TS; do
  G=$((1 + i % 6)); i=$((i + 1)); O=/tmp/dkta/lbours/fem_ak_bsel/${t}_$VK
  if [ $t = ficus ]; then DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --N 640 --dump"
  else DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --node_h 0.028284 --floor vel --N 267"; fi
  ( [ -f $O/summary.json ] && exit 0; mkdir -p $O
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_bsel_sim.py --out $O --tag $t $DA $EXTRA < /dev/null > $O/log.txt 2>&1
    echo "$(date +%T) ${t}_$VK $(tail -1 $O/log.txt | cut -c1-150)"; rclone copy $O r2:storage/result/anchorflow/lbours/fem_ak_bsel/${t}_$VK -q ) &
  [ $((i % 6)) -eq 0 ] && wait
done
wait
echo "[끝] $(date +%T)"
