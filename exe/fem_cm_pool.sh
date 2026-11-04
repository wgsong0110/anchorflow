#!/bin/bash
# 절 CM 판정 풀 (잡 안, 비측정): 목록 줄 "<판> <씬> <공통 인자...>" 를 빈 카드 (계산 프로세스 0 이고 아무도 잡지 않은 카드; 카드 0 은 단독 측정 몫) 하나에 하나씩.
# 카드 잡기 규약은 exe/fem_an_pool.sh 와 같음 (/dev/shm/dkta/anerr_claim/<카드> = 일꾼 pid). 출력 /dev/shm/dkta/lbours/fem_cm/<묶음>/<판>_<씬> -> 끝나면 R2 같은 경로.
# 사용: CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh <묶음> <목록>
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
NM=$1; LST=$2
R=/dev/shm/dkta/lbours/fem_cm/$NM; RR=r2:storage/result/anchorflow/lbours/fem_cm/$NM; CL=/dev/shm/dkta/anerr_claim
mkdir -p $R $CL $CUPY_CACHE_DIR
source exe/fem_cm_scn.sh
echo "[묶음 $NM] $(cat .done) $(date '+%F %T')"
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    case " ${CM_CARDS:-1 2 3 4 5 6} " in *" $i "*) ;; *) continue;; esac
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
run_one() {
  local G=$1 V=$2 t=$3; shift 3
  local O=$R/${V}_$t; mkdir -p $O
  cm_scene $t "$@"
  echo "카드 $G $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/apps_at_start.txt
  CUDA_VISIBLE_DEVICES=$G timeout ${CM_TMO:-2400} $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  if [ $t = ficus ] && [ -f $O/pred.pt ]; then $PY exe/fem_p_period.py --data $FI --pred $O/pred.pt > $O/period.txt 2>&1; fi
  echo "$V $t (카드 $G) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')" >> $R/done.txt
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt; rclone copy $R/done.txt $RR -q
}
while read -r V t A; do
  [ -z "$V" ] && continue; case $V in \#*) continue;; esac
  [ -f $R/${V}_$t/run_info.txt ] && grep -q rc= $R/${V}_$t/run_info.txt && continue
  while true; do
    while [ $(jobs -rp | wc -l) -ge ${CM_PAR:-6} ]; do sleep 15; done
    G=$(free_card); [ -n "$G" ] && break; sleep 20
  done
  run_one $G $V $t $A &
  echo $! > $CL/$G
  echo "띄움 $V $t 카드 $G $(date +%T)"
  sleep ${CM_GAP:-8}
done < $LST
wait
echo "[묶음 끝 $NM] $(date '+%F %T')" | tee $R/END; rclone copy $R/END $RR -q
