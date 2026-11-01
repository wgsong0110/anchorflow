#!/bin/bash
# 절 AN: 암시적 B-스플라인 하이브리드 검증판 (비측정). 사용: AN_GPUS="1 2 3" bash exe/fem_an_run.sh <판 이름> <궤적,...> [추가 인자]
# 궤적: stretch·shear (블록 아핀), drop1 (떨어지는 블록), ficus, L000, 0000 에서 0019 (lego). 출력 /tmp/dkta/lbours/fem_an/<궤적>_<판> -> R2 즉시.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
VK=$1; TS=$(echo $2 | tr ',' ' '); shift 2; EXTRA="$*"
GS=(${AN_GPUS:-1 2 3 4 5 6}); i=0
for t in $TS; do
  O=/tmp/dkta/lbours/fem_an/${t}_$VK; mkdir -p $O
  if [ $t = ficus ]; then DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump"
  elif [ $t = stretch ] || [ $t = shear ]; then DA="--data /tmp/dkta/ak_block/$t.pt --frames 80 --floor none --dump"
  elif [ $t = drop1 ]; then DA="--data /tmp/dkta/ak_drop2/drop_f1.pt --frames 80 --floor vel --dump"
  elif [ $t = L000 ]; then DA="--data /home/dkta/work/one/L000/L000.pt --frames 80 --floor vel --dump"
  else DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --floor vel --dump"; fi
  G=${GS[$((i % ${#GS[@]}))]}; i=$((i + 1))
  ( echo "gpu $G $(git log -1 --oneline) $(date +%T)" > $O/run_info.txt
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_sim.py --out $O --tag $t $DA $EXTRA < /dev/null > $O/log.txt 2>&1
    rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/${t}_$VK -q ) &
  # 같은 카드에 여러 개를 얹을 때 순서대로 시작 (컴파일 경합 방지)
  sleep 5
done
wait
for t in $TS; do O=/tmp/dkta/lbours/fem_an/${t}_$VK; echo "${t}_$VK $(grep -a '롤아웃\|Error\|error\|대조' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"; done
echo "[끝 $VK] $(date +%T)"
