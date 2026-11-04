#!/bin/bash
# 절 AN.15 지속형 커널 판 1 차: 결과 불변 대조 (그래프 판 vs 지속형 판, 같은 입력, 10 프레임; 결정적·비결정, 허용치 판·고정 판) + 짧은 시간 비교 (pk_loc 별).
# 카드 0 (gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인). 출력 /dev/shm/dkta/lbours/fem_an/<묶음> -> R2 같은 경로, 표 marks/<묶음>.md, 끝 표식 marks/<묶음>.END
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pk XDG_CACHE_HOME=/dev/shm/dkta/cache
NM=${1:-pk1}; R=/dev/shm/dkta/lbours/fem_an/$NM; M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $R $M $CUPY_CACHE_DIR
SD=/dev/shm/dkta/anerr_data
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done) $($PY -c 'import cupy;print("cupy",cupy.__version__)')"
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --integ be"
T="--tolN 1e-2 --tolP 1e-1"
LG="--data $SD/L000.pt --floor vel --tag L000 --frames 10 --dump"
FI="--data $SD/F_pg_v0.pt --floor none --cfg_bc --region_density --pg_vdamp --tag ficus --frames ${PK_FIFR:-10} --dump"
run() { local v=$1; shift; local O=$R/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/fem_an_rt_sim.py --out $O $B "$@" 9>&- < /dev/null > $O/log.txt 2>&1
  echo "$v rc=$? $(date +%T) $(grep -a '롤아웃\|rror\|지속형\|발산' $O/log.txt | tail -4 | cut -c1-300 | tr '\n' ' ')"
  [ -n "$(grep -a 'Error' $O/log.txt)" ] && tail -40 $O/log.txt
  rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/$NM/$v -q; }
PAIRS=""
for e in ${PK_LIST:-lgd lgn fid fin lgfd}; do
  case $e in
    lgd) A="$LG $T --newton 1 --cg 5 --det";; lgn) A="$LG $T --newton 1 --cg 5";;
    fid) A="$FI $T --newton 2 --cg 10 --det";; fin) A="$FI $T --newton 2 --cg 10";;
    lgfd) A="$LG --newton 1 --cg 5 --det";; fifd) A="$FI --newton 2 --cg 10 --det";;
  esac
  run ${e}_pk $A --pk ${PK_X:-}
  run ${e}_g $A
  PAIRS="$PAIRS ${e}_g:${e}_pk"
done
# 짧은 시간 비교 (비결정, 허용치 판): 지속형 pk_loc 1·2·99
for loc in ${PK_LOCS:-1 2 99}; do
  run lgt_pk$loc $LG $T --newton 1 --cg 5 --pk --pk_loc $loc
  run fit_pk$loc $FI $T --newton 2 --cg 10 --pk --pk_loc $loc
  PAIRS="$PAIRS lgn_g:lgt_pk$loc fin_g:fit_pk$loc"
done
flock -u 9; exec 9>&-
$PY exe/fem_an_pk_cmp.py $R $PAIRS > $M/$NM.md 2>&1; cat $M/$NM.md
echo "[끝] $(date '+%F %T')" > $M/$NM.END
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q --include "$NM.*"
