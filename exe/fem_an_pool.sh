#!/bin/bash
# 절 AN.14 이후 실행 풀 (잡 안, 비측정): 목록 파일의 줄 "<판> <궤적> [인자...]" 을 빈 카드 (계산 프로세스 0 이고 아무도 잡지 않은 카드; 카드 0 은 단독 측정 몫이라 안 씀) 하나에 하나씩.
# 카드 잡기는 오차 원인 에이전트 (exe/fem_anerr_pool.sh, 가지 implicit-err) 와 같은 규약: /dev/shm/dkta/anerr_claim/<카드> = 일꾼 pid.
# 궤적: L000, 0000 에서 0019, ficus (125 프레임 덤프 + 주기), stretch·shear (블록 80 프레임 덤프 + 주기), drop1 (낙하 덤프 + 반발). 출력 /dev/shm/dkta/lbours/fem_an/<묶음>/<판>_<궤적> -> 끝나면 바로 R2 같은 경로.
# 사용: bash exe/fem_an_pool.sh <묶음> <목록 파일>; 후보 카드 AN_CARDS (기본 "1 2 4 6"), 동시 상한 AN_PAR (기본 4)
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_imp XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
NM=$1; LST=$2
R=/dev/shm/dkta/lbours/fem_an/$NM; RR=r2:storage/result/anchorflow/lbours/fem_an/$NM; CL=/dev/shm/dkta/anerr_claim
mkdir -p $R $CL $CUPY_CACHE_DIR
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=$SD
L0=$SD/L000.pt; [ -s $L0 ] || cp /home/dkta/work/one/L000/L000.pt $L0
FI=$SD/F_pg_v0.pt; [ -s $FI ] || cp /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $FI
DD=/tmp/dkta/ak_drop2; [ -s $DD/drop_f1.pt ] && [ -s $DD/mpm/x240.pt ] || { DD=$SD/ak_drop2; [ -s $DD/mpm/x240.pt ] || rclone copy r2:storage/result/anchorflow/lbours/ak_drop2 $DD -q --include "drop_f1.pt" --include "init.pt" --include "mpm/x240.pt"; }
BK=/tmp/dkta/ak_block
echo "[묶음 $NM] $(cat .done) $(date '+%F %T')"; df -h /tmp /dev/shm | tail -2
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    case " ${AN_CARDS:-1 2 4 6} " in *" $i "*) ;; *) continue;; esac
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
run_one() {
  local G=$1 V=$2 t=$3; shift 3
  local O=$R/${V}_$t; mkdir -p $O
  case $t in
    L000) DA="--data $L0 --frames 80 --floor vel";;
    ficus) DA="--data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump";;
    stretch|shear) DA="--data $BK/$t.pt --frames 80 --floor none --dump";;
    drop1) DA="--data $DD/drop_f1.pt --frames 80 --floor vel --dump";;
    *) [ -s $LD/$t.pt ] || { cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt.$BASHPID && mv $LD/$t.pt.$BASHPID $LD/$t.pt; }; DA="--data $LD/$t.pt --frames 80 --floor vel";;
  esac
  echo "카드 $G $(cat .done) $(date '+%F %T') 인자 $DA $*" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/apps_at_start.txt
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_an_rt_sim.py --out $O --tag $t $DA "$@" < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  if [ $t = ficus ] && [ -f $O/pred.pt ]; then $PY exe/fem_p_period.py --data $FI --pred $O/pred.pt > $O/period.txt 2>&1; fi
  if { [ $t = stretch ] || [ $t = shear ]; } && [ -f $O/pred.pt ]; then $PY exe/fem_ak_block_ana.py $BK/$t.pt mpm:$BK/mpm_$t/x.pt $V:$O/pred.pt > $O/block_ana.txt 2>&1; fi
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1; fi
  echo "$V $t (카드 $G) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')" >> $R/done.txt
  rclone copy $O $RR/${V}_$t -q; rclone copy $R/done.txt $RR -q
}
while read -r V t A; do
  [ -z "$V" ] && continue; case $V in \#*) continue;; esac
  [ -f $R/${V}_$t/run_info.txt ] && grep -q rc= $R/${V}_$t/run_info.txt && continue
  while true; do
    while [ $(jobs -rp | wc -l) -ge ${AN_PAR:-4} ]; do sleep 20; done
    G=$(free_card); [ -n "$G" ] && break; sleep 30
  done
  run_one $G $V $t $A &
  echo $! > $CL/$G
  echo "띄움 $V $t 카드 $G $(date +%T)"
  sleep ${AN_GAP:-10}
done < $LST
wait
echo "[묶음 끝 $NM] $(date '+%F %T')" | tee $R/END; rclone copy $R/END $RR -q
