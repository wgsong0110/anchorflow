#!/bin/bash
# 절 AN.err 실행 풀 (잡 안, 비측정): 할 일 목록 파일의 줄 "<판> <궤적> [인자...]" 을 차례로, 빈 카드 (카드 1 에서 6 중 계산 프로세스 0 이고 내가 잡지 않은 카드) 하나에 하나씩 띄운다. 카드 0 은 쓰지 않는다.
# 궤적: L000 (NFS -> 노드 /tmp 복사), 0000 에서 0019 (/tmp/dkta/af_data), drop1 (/tmp/dkta/ak_drop2/drop_f1.pt). 출력 /dev/shm/dkta/lbours/fem_anerr/<묶음> (노드 /tmp 가득 참)/<판>_<궤적> -> 끝나면 바로 R2 같은 경로.
# 사용: bash exe/fem_anerr_pool.sh <묶음> <목록 파일 (작업 사본 상대)> ; 동시 실행 상한 ERR_PAR (기본 6), 후보 카드 ERR_CARDS (기본 "1 2 3 4 5 6")
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
NM=$1; LST=$2
R=${ERR_ROOT:-/dev/shm/dkta/lbours/fem_anerr}/$NM; RR=r2:storage/result/anchorflow/lbours/fem_anerr/$NM; CL=/dev/shm/dkta/anerr_claim
mkdir -p $R $CL $CUPY_CACHE_DIR /tmp/dkta/anerr_data
[ -s /tmp/dkta/anerr_data/L000.pt ] || cp /home/dkta/work/one/L000/L000.pt /tmp/dkta/anerr_data/L000.pt
echo "[묶음 $NM] $(cat .done) $(date '+%F %T')"; df -h /tmp | tail -1
free_card() {   # 빈 카드 하나 (없으면 빈 문자열)
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    case " ${ERR_CARDS:-1 2 3 4 5 6} " in *" $i "*) ;; *) continue;; esac
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
run_one() {   # 카드 판 궤적 인자...
  local G=$1 V=$2 t=$3; shift 3
  local O=$R/${V}_$t; mkdir -p $O
  case $t in
    L000) DA="--data /tmp/dkta/anerr_data/L000.pt --floor vel";;
    drop1) DA="--data /tmp/dkta/ak_drop2/drop_f1.pt --floor vel --dump";;
    *) DA="--data /tmp/dkta/af_data/$t.pt --floor vel";;
  esac
  echo "카드 $G $(cat .done) $(date '+%F %T') 인자 $DA $*" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/apps_at_start.txt
  FA=--fast; case " $* " in *" --slow "*) FA="";; esac
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_anerr_sim.py --out $O --tag $t $DA --frames ${ERR_FR:-80} $FA --gfin $(echo " $* " | sed "s/ --slow / /") < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then
    $PY exe/fem_ak_drop_ana.py /tmp/dkta/ak_drop2/init.pt mpm:/tmp/dkta/ak_drop2/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1
  fi
  echo "$V $t (카드 $G) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')" >> $R/done.txt
  rclone copy $O $RR/${V}_$t -q; rclone copy $R/done.txt $RR -q
}
while read -r V t A; do
  [ -z "$V" ] && continue; case $V in \#*) continue;; esac
  [ -f $R/${V}_$t/run_info.txt ] && grep -q rc= $R/${V}_$t/run_info.txt && continue        # 이미 끝난 것은 건너뜀 (다시 띄운 경우)
  while true; do
    while [ $(jobs -rp | wc -l) -ge ${ERR_PAR:-6} ]; do sleep 20; done
    G=$(free_card); [ -n "$G" ] && break; sleep 30
  done
  run_one $G $V $t $A &
  echo $! > $CL/$G
  echo "띄움 $V $t 카드 $G $(date +%T)"
  sleep 40      # 프로세스가 카드에 잡힐 때까지
done < $LST
wait
echo "[묶음 끝 $NM] $(date '+%F %T')" | tee $R/END; rclone copy $R/END $RR -q
