#!/bin/bash
# 절 AN.14 단독 측정 (카드 0, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인, 2 회): 목록 AN_LIST 의 항목 "<이름>:<궤적>:<프레임>:<인자 (+ 는 빈칸)>".
# 궤적 L000 / ficus / 0016 등. 출력 /dev/shm/dkta/lbours/fem_an/<묶음>/<이름>_r<회> -> R2. 표식 marks/<묶음>.out
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_imp XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
NM=${1:?묶음}; R=/dev/shm/dkta/lbours/fem_an/$NM; M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $R $M $CUPY_CACHE_DIR
SD=/dev/shm/dkta/anerr_data; LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=$SD
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done)" > $R/info.txt
for rep in 1 2; do for e in $AN_LIST; do IFS=: read v t fr A <<< "$e"; A=$(echo "$A" | tr '+' ' ')
  case $t in L000) DA="--data $SD/L000.pt --floor vel";; ficus) DA="--data $SD/F_pg_v0.pt --floor none --cfg_bc --region_density --pg_vdamp";; *) DA="--data $LD/$t.pt --floor vel";; esac
  O=$R/${v}_r$rep; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/fem_an_rt_sim.py --out $O --tag $t $DA --frames $fr $A 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  echo "$v r$rep $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-200) $($PY -c "import json;d=json.load(open('$O/summary.json'));print('max',round(d.get('ms_max') or -1,2),'p90',round(d.get('ms_p90') or -1,2),'nP',d.get('it_nP'),'nN',d.get('it_nN'))" 2>/dev/null)"
done; done > $R/run.out 2>&1
flock -u 9; exec 9>&-
cp $R/run.out $M/${NM}.out
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
echo "[끝] $(date '+%F %T')" > $M/${NM}.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
