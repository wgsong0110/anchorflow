#!/bin/bash
# 절 CM191 (final_fix) 일꾼 (exe/fpa_worker.sh 복사: 결과 lbours/final_fix, 명령 exe/ffx_cmd.py): 이 인스턴스의 작업 대기열 (/root/ffx/queue.txt, 한 줄 = 장면:판:반복) 에서 하나씩 꺼내 카드 FPA_GPU 에서 단독 실행.
#   카드마다 일꾼 하나 (그 카드에는 이 실행 하나뿐 = 단독 시간). 결과 R2 lbours/final_fix/runs/<장면>_<판>_r<반복>/ (metrics.json·summary.json·cmd.txt·log.txt ...).
export PATH=/home/dkta/bin:$PATH
H=$1
RA=r2:storage/result/anchorflow/lbours/final_fix; GPU=${FPA_GPU:-0}; O=/dev/shm/dkta/ffx$GPU; Q=/root/ffx/queue.txt; mkdir -p $O /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
say() { echo "[w$GPU $(date '+%F %T')] $*"; }
# 대기열이 비면 R2 marks/<FFX_TAG>.queue 를 다시 읽어 아직 시작 안 한 줄을 붙임 (실행 중 작업 추가용); FFX_IDLE 초 (기본 1200) 동안 새 일이 없으면 끝
pop() { ( flock 9; touch /root/ffx/started.txt; j=$(head -1 $Q); if [ -z "$j" ] && [ -n "${FFX_TAG:-}" ]; then
    rclone cat $RA/marks/$FFX_TAG.queue 2>/dev/null | grep . | grep -vxF -f /root/ffx/started.txt > $Q.new; mv $Q.new $Q; j=$(head -1 $Q); fi
  [ -n "$j" ] && { sed -i 1d $Q; echo "$j" >> /root/ffx/started.txt; }; echo "$j" ) 9>/root/ffx/queue.lock; }
post() { local D=$1 n=$2 s=$3 v=$4 r=$5 rc=$6 G=$7
  CUDA_VISIBLE_DEVICES=$G ABL_EIG_VARS=def,q1,q27 timeout 1800 $PY exe/abl_an.py $D $s $v $r > $D/an.log 2>&1; say "$n 분석 rc=$? $(tail -1 $D/an.log | cut -c1-200)"
  [ -f $D/metrics.json ] || echo "{\"rc\": $rc, \"no_summary\": true}" > $D/metrics.json
  rclone copy $D $RA/${FFX_RUNS:-runs}/$n -q --transfers 8 --exclude "q27src/**" --exclude "pred.pt"
  [ -f $D/pred.pt ] && rclone copyto $D/pred.pt $RA/${FFX_RUNS:-runs}/$n/pred.pt -q --s3-chunk-size 64M --s3-upload-concurrency 16
  rm -rf $D; }
idle=0
while :; do
  j=$(pop); if [ -z "$j" ]; then [ $idle -ge ${FFX_IDLE:-1200} ] && break; sleep 60; idle=$((idle + 60)); continue; fi; idle=0
  IFS=: read s v r <<< "$j"; n=${s}_${v}_r$r; D=$O/$n
  if rclone cat $RA/${FFX_RUNS:-runs}/$n/metrics.json 2>/dev/null | grep -aq '"frames_done": [0-9]'; then say "$n 이미 있음"; continue; fi
  rm -rf $D; mkdir -p $D
  $PY exe/ffx_cmd.py $s $v $r $D > $D/cmd.txt || { say "$n 명령 생성 실패"; continue; }
  echo "$H" > $D/commit.txt
  say "$n 시작 (카드 $GPU, 다른 계산 프로세스 $(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU timeout ${FFX_TMO:-5400} bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/log.txt 2>&1; rc=$?
  nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader > $D/card_after.txt 2>&1
  say "$n rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $D/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $D/log.txt | tail -1 | cut -c1-200)"
  if [ "$r" = 1 ] && case " def q1 q27 " in *" $v "*) true;; *) false;; esac; then
    post "$D" "$n" "$s" "$v" "$r" "$rc" $GPU                                        # 영 모드 분석 (GPU) 은 카드에서 차례로
  else
    post "$D" "$n" "$s" "$v" "$r" "$rc" "" &                                        # 에너지·뒤집힘 분석은 CPU 로 뒤에서 (카드는 다음 판으로)
  fi
done
wait
say "일꾼 끝"
