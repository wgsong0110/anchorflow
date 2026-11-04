#!/bin/bash
# 절 AN.20 목록 실행 (카드 0 단독, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인): 목록 파일 $2 의 줄 "<이름> <인자...>" (공통 앞 인자 B 뒤에 붙음; 궤적 줄임 @LG @FI).
# 출력 /dev/shm/dkta/lbours/fem_an/<묶음>/<이름> -> R2, 줄 요약 run.out (ficus 에 pred.pt 가 있으면 주기 period.txt; --kbench 면 kbench.json)
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mg XDG_CACHE_HOME=/dev/shm/dkta/cache
NM=$1; LST=$2; R=/dev/shm/dkta/lbours/fem_an/$NM; mkdir -p $R $CUPY_CACHE_DIR; RR=r2:storage/result/anchorflow/lbours/fem_an/$NM
SD=/dev/shm/dkta/anerr_data; FD=$SD/F_pg_v0.pt
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done)" | tee -a $R/run.out
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --integ be"
while read -r v A; do [ -z "$v" ] && continue; [ "${v:0:1}" = "#" ] && continue
  A=${A//@LG/--data $SD/L000.pt --floor vel --tag L000}; A=${A//@FI/--data $FD --floor none --cfg_bc --region_density --pg_vdamp --tag ficus}
  O=$R/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout ${MG_TMO:-1800} $PY -u exe/fem_an_rt_sim.py --out $O $B $A 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  [ -f $O/pred.pt ] && grep -q ficus <<< "$A" && $PY exe/fem_p_period.py --data $FD --pred $O/pred.pt > $O/period.txt 2>&1
  echo "$v $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ') $($PY -c "import json;d=json.load(open('$O/summary.json'));print('sha',d['x_sha1'],'mean',round(d.get('ms_mean') or -1,2),'max',round(d.get('ms_max') or -1,2),'p90',round(d.get('ms_p90') or -1,2),'nP',(d.get('it_nP') or {}).get('mean'),'nN',(d.get('it_nN') or {}).get('mean'))" 2>/dev/null) $(grep -a '대비' $O/period.txt 2>/dev/null | cut -c1-140)" | tee -a $R/run.out
  grep -q Error $O/log.txt && tail -30 $O/log.txt
  rclone copy $O $RR/$v -q --exclude pred.pt
  [ -f $O/pred.pt ] && rclone copy $O/pred.pt $RR/$v -q
done < $LST
rclone copy $R/run.out $RR -q
echo "[끝 $NM] $(date '+%F %T')"
