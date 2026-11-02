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
mkdir -p $R $CL $CUPY_CACHE_DIR
# 자료: 노드에 없으면 /dev/shm/dkta/anerr_data 로 (다음 잡이 다른 노드일 때) -- lego NFS /home/dkta/work/lbpt/test, L000 NFS, 낙하 블록 R2 lbours/ak_drop2
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=$SD
L0=/tmp/dkta/anerr_data/L000.pt; [ -s $L0 ] || { L0=$SD/L000.pt; [ -s $L0 ] || cp /home/dkta/work/one/L000/L000.pt $L0; }
DD=/tmp/dkta/ak_drop2; [ -s $DD/drop_f1.pt ] && [ -s $DD/mpm/x240.pt ] || { DD=$SD/ak_drop2; [ -s $DD/mpm/x240.pt ] || rclone copy r2:storage/result/anchorflow/lbours/ak_drop2 $DD -q --include "drop_f1.pt" --include "init.pt" --include "mpm/x240.pt"; }
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
    L000) DA="--data $L0 --floor vel";;
    drop1) DA="--data $DD/drop_f1.pt --floor vel --dump";;
    stretch|shear) BK=/tmp/dkta/ak_block; [ -s $BK/$t.pt ] || { BK=$SD/ak_block; [ -s $BK/$t.pt ] || { mkdir -p $BK; $PY exe/fem_ak_block.py $BK.tmp$BASHPID 1.0 0.03 0.0125 > /dev/null 2>&1 && mv -n $BK.tmp$BASHPID/*.pt $BK/; }; }
           DA="--data $BK/$t.pt --frames 80 --floor none --dump";;
    ficus) [ -s /dev/shm/dkta/anerr_data/F_pg_v0.pt ] || { mkdir -p /dev/shm/dkta/anerr_data; cp /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt /dev/shm/dkta/anerr_data/; }
           DA="--data /dev/shm/dkta/anerr_data/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump";;
    *) [ -s $LD/$t.pt ] || { cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt.$BASHPID && mv $LD/$t.pt.$BASHPID $LD/$t.pt; }; DA="--data $LD/$t.pt --floor vel";;
  esac
  echo "카드 $G $(cat .done) $(date '+%F %T') 인자 $DA $*" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/apps_at_start.txt
  FA=--fast; case " $* " in *" --slow "*) FA="";; esac
  FR=${ERR_FR:-80}; [ $t = ficus ] && FR=${ERR_FR:-125}
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_anerr_sim.py --out $O --tag $t $DA --frames $FR $FA --gfin $(echo " $* " | sed "s/ --slow / /") < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then
    $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1
  fi
  if { [ $t = stretch ] || [ $t = shear ]; } && [ -f $O/pred.pt ]; then
    $PY exe/fem_ak_block_ana.py $BK/$t.pt $V:$O/pred.pt > $O/block_ana.txt 2>&1
  fi
  if [ $t = ficus ] && [ -f $O/pred.pt ]; then
    $PY exe/fem_p_period.py --data /dev/shm/dkta/anerr_data/F_pg_v0.pt --pred $O/pred.pt > $O/period.txt 2>&1
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
