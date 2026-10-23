#!/bin/bash
# 국소 어텐션 범위(--attn_ring K) 비용 측정 (사용자 10-23). 잡 안에서 job_run.sh 로:
#   AF_DETACH=1 bash exe/job_run.sh <이 스크립트를 부르는 래퍼> <로그>
#   bash exe/attnring_bench.sh <커밋>
# 1) 워크트리 /home/dkta/work/wt/attnring_<커밋> (git fetch; git worktree add)
# 2) **단독 카드** 찾기: memory.used < 500 MiB · utilization 0 · compute app 없음 (3 초 간격 두 번). 없으면 20 초마다 다시 (최대 AB_WAIT 초)
# 3) 그 카드에서 단위 확인 (test_attn_ring.py: K=1 이 이전 커밋과 비트 같음, K=2,3 이웃 = 조밀 BFS)
# 4) 구성 8 개 (sgnn 4·8 층, lattn 4·8 층 × K=1,2,3) 마다: pfip_noin L000 명령 그대로에서
#    --ov_roll 2000 -> AB_ROLL (기본 40), --eval_len 80 -> 2, AF_OVNET_ESTAR 500 -> 0 (E* 안 잼), 출력 경로·카드만 바꿈.
#    AF_ITER_TIME 로 Adam 반복마다 (순전파+역전파+자르기+걸음) 시간·최대 메모리. 구성마다 카드가 여전히 비었는지 다시 확인하고,
#    도는 동안 1 초마다 그 카드의 compute app 목록을 적어 (gpu_watch_<구성>.txt) 우리 PID 말고 다른 것이 있었는지 판정.
# 출력 /home/dkta/work/lbours/attnring_bench/<커밋>/  -> R2 r2:storage/result/anchorflow/lbours/attnring_bench/<커밋> (통째)
set -u
C=${1:?커밋}
W=/home/dkta/work; WT=$W/wt/attnring_$C
O=$W/lbours/attnring_bench/$C; R=r2:storage/result/anchorflow/lbours/attnring_bench/$C
ROLL=${AB_ROLL:-40}; WAIT=${AB_WAIT:-21600}
mkdir -p $O
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; set -u
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=$HOME/bin:$PATH
up(){ rclone copy $O $R -q; }
echo "[시작] $(date '+%F %T') 노드 $(hostname) 커밋 $C"
if [ ! -d $WT ]; then
  ( cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C ) || { echo "[오류] 워크트리"; exit 1; }
fi
[ "$(cd $WT && git rev-parse --short HEAD)" = "$C" ] || { echo "[오류] 워크트리 커밋 다름"; exit 1; }
BASE=$(sed -n 's/^\[실행 명령[^]]*\] //p' $W/lbours/pfip_noin/L000/cmd.txt | head -1)
[ -n "$BASE" ] || { echo "[오류] 기준 명령 없음"; exit 1; }
echo "[기준 명령 pfip_noin L000] $BASE" > $O/cmd_base.txt

# 카드 g 가 비었나: 메모리 < 500, 사용률 0, compute app 0 -- 3 초 간격 두 번
free_now(){ nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3<500 && $4==0 {print $1" "$2}'; }
apps_on(){ nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$1"; }
pick(){   # -> "index uuid" 또는 빈 문자열
  local a b
  a=$(free_now); [ -z "$a" ] && return
  sleep 3
  b=$(free_now)
  while read -r i u; do
    [ -z "$i" ] && continue
    grep -q "^$i $u$" <<<"$a" || continue
    [ "$(apps_on $u)" = 0 ] && { echo "$i $u"; return; }
  done <<<"$b"
}
wait_card(){
  local t0=$(date +%s) r
  while :; do
    r=$(pick); [ -n "$r" ] && { echo "$r"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt $WAIT ] && return 1
    sleep 20
  done
}
snap(){ { echo "== $1 $(date '+%F %T')"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
          nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } >> $O/gpu_snap.txt; }

read -r G U <<<"$(wait_card)" || true
[ -n "${G:-}" ] || { echo "[오류] $WAIT 초 동안 빈 카드 없음"; snap no_card; up; exit 1; }
echo "[카드] GPU$G $U ($(date '+%T'))"; snap "pick GPU$G"

# 단위 확인 (이 카드에서, 수 초)
git -C $WT show 75f64c6:lib/anchorflow/simplex_gnn.py > $O/old_simplex_gnn_75f64c6.py
( cd $WT && CUDA_VISIBLE_DEVICES=$G python -u exe/test_attn_ring.py --old $O/old_simplex_gnn_75f64c6.py > $O/test_attn_ring.log 2>&1 )
tail -1 $O/test_attn_ring.log
# 실패해도 측정은 이어 간다 (빈 카드를 잡은 김에; 시간·메모리 측정은 비트 동일성과 무관) -- 결과는 로그로 보고
grep -q "TEST_ATTN_RING OK" $O/test_attn_ring.log || { echo "[경고] 단위 확인 실패 항목:"; grep "실패" $O/test_attn_ring.log; }

CFGS="sgnn8 sgnn4 lattn4_k1 lattn4_k2 lattn4_k3 lattn8_k1 lattn8_k2 lattn8_k3"
for K in $CFGS; do
  D=$O/$K; RD=$R/$K
  [ -f $D/.done ] && { echo "[이미 끝] $K"; continue; }
  case $K in
    sgnn8) LY=8; XF="" ;;
    sgnn4) LY=4; XF="" ;;
    lattn4_k*) LY=4; XF="--gnn_layer lattn --attn_heads 8 --attn_ring ${K##*_k}" ;;
    lattn8_k*) LY=8; XF="--gnn_layer lattn --attn_heads 8 --attn_ring ${K##*_k}" ;;
  esac
  # 카드가 여전히 단독인지 (사이에 대기열 러너가 가져갈 수 있다) -- 아니면 다른 빈 카드
  ok=0
  while read -r i u; do [ "$i" = "$G" ] && ok=1; done <<<"$(pick)"
  if [ $ok = 0 ]; then
    echo "[카드] GPU$G 가 더 이상 단독이 아님 -- 다시 찾는다"; snap "lost GPU$G"
    read -r G U <<<"$(wait_card)" || true
    [ -n "${G:-}" ] || { echo "[오류] 빈 카드 없음"; up; exit 1; }
    echo "[카드] GPU$G $U"
  fi
  rm -rf $D.partial; mkdir -p $D.partial
  CMD=$(sed -e "s#/home/dkta/work/lbours/pfip_noin/L000#$D.partial#g" -e "s#r2:storage/result/anchorflow/lbours/pfip_noin/L000#$RD#g" \
            -e "s#CUDA_VISIBLE_DEVICES=[0-9]*#AF_ITER_TIME=$D.partial/iter_time.jsonl CUDA_VISIBLE_DEVICES=$G#" \
            -e "s/AF_OVNET_ESTAR=500/AF_OVNET_ESTAR=0/" -e 's/ --net_inertial / /' \
            -e "s/ --gnn_layers 8 / --gnn_layers $LY /" -e "s# --bc_stick_xn # --bc_stick_xn $XF #" \
            -e "s/ --ov_roll 2000 / --ov_roll $ROLL /" -e "s/ --eval_len 80 / --eval_len 2 /" <<<"$BASE" | tr -s ' ')
  for pat in "--gnn_layers $LY " "--ov_roll $ROLL " "--eval_len 2 " "AF_OVNET_ESTAR=0 " "AF_ITER_TIME=" "CUDA_VISIBLE_DEVICES=$G "; do
    grep -q -- "$pat" <<<"$CMD" || { echo "[오류] $K 명령 바꾸기 실패 ($pat)"; exit 1; }
  done
  grep -q -- "--net_inertial" <<<"$CMD" && { echo "[오류] --net_inertial"; exit 1; }
  [ -z "$XF" ] || grep -q -- "$XF" <<<"$CMD" || { echo "[오류] $K 추가 인자 ($XF)"; exit 1; }
  echo "[실행 명령 $(date '+%F_%T') 커밋 $C GPU$G $U] $CMD" > $D.partial/cmd.txt
  snap "start $K GPU$G"
  ( cd $WT && exec $CMD > $D.partial/run.log 2>&1 < /dev/null ) &
  P=$!
  ( while kill -0 $P 2>/dev/null; do echo "$(date +%s.%N | cut -c1-14) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $G) | $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep $U | tr '\n' ';')"; sleep 1; done ) > $D.partial/gpu_watch.txt 2>&1 &
  wait $P; rc=$?
  sleep 1
  snap "end $K GPU$G rc=$rc"
  # 단독 판정: 감시 기록에 나온 PID 가 하나뿐인가
  npid=$(grep -o "$U, [0-9]*" $D.partial/gpu_watch.txt | awk '{print $2}' | sort -u | wc -l)
  echo "$K GPU$G pids_on_card=$npid ($(grep -o "$U, [0-9]*" $D.partial/gpu_watch.txt | awk '{print $2}' | sort -u | tr '\n' ' '))" > $D.partial/exclusive.txt
  grep -q OVNET_FIT_DONE $D.partial/run.log || { echo "[오류] $K 안 끝남 rc=$rc"; tail -5 $D.partial/run.log; mv $D.partial $D.fail_$(date +%H%M%S); up; continue; }
  mv $D.partial $D; touch $D/.done
  echo "[끝] $K $(cat $D/exclusive.txt) $(date '+%T')"
  up
done
( cd $WT && python -u exe/attnring_report.py $O > $O/report.txt 2>&1 ); cat $O/report.txt
up
rclone check $O $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo "ATTNRING_BENCH_DONE $(date '+%F %T')"
