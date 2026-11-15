#!/bin/bash
# 절 CM190 일꾼: 이 인스턴스의 작업 대기열 (/root/fpa/queue.txt, 한 줄 = 장면:판:반복) 에서 하나씩 꺼내 카드 FPA_GPU 에서 단독 실행.
#   카드마다 일꾼 하나 (그 카드에는 이 실행 하나뿐 = 단독 시간). 결과 R2 lbours/final_pa/runs/<장면>_<판>_r<반복>/ (metrics.json·summary.json·cmd.txt·log.txt ...).
export PATH=/home/dkta/bin:$PATH
H=$1
RA=r2:storage/result/anchorflow/lbours/final_pa; GPU=${FPA_GPU:-0}; O=/dev/shm/dkta/fpa$GPU; Q=/root/fpa/queue.txt; mkdir -p $O /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
say() { echo "[w$GPU $(date '+%F %T')] $*"; }
pop() { ( flock 9; j=$(head -1 $Q); [ -n "$j" ] && sed -i 1d $Q; echo "$j" ) 9>/root/fpa/queue.lock; }
post() { local D=$1 n=$2 s=$3 v=$4 r=$5 rc=$6 G=$7
  CUDA_VISIBLE_DEVICES=$G ABL_EIG_VARS=def,q1,q27 timeout 1800 $PY exe/abl_an.py $D $s $v $r > $D/an.log 2>&1; say "$n 분석 rc=$? $(tail -1 $D/an.log | cut -c1-200)"
  [ -f $D/metrics.json ] || echo "{\"rc\": $rc, \"no_summary\": true}" > $D/metrics.json
  rclone copy $D $RA/runs/$n -q --transfers 8 --exclude "q27src/**" --exclude "pred.pt"
  [ -f $D/pred.pt ] && rclone copyto $D/pred.pt $RA/runs/$n/pred.pt -q --s3-chunk-size 64M --s3-upload-concurrency 16
  rm -rf $D; }
while :; do
  j=$(pop); [ -z "$j" ] && break
  IFS=: read s v r <<< "$j"; n=${s}_${v}_r$r; D=$O/$n
  if rclone cat $RA/runs/$n/metrics.json 2>/dev/null | grep -aq '"frames_done": [0-9]'; then say "$n 이미 있음"; continue; fi
  rm -rf $D; mkdir -p $D
  $PY exe/fpa_cmd.py $s $v $r $D > $D/cmd.txt || { say "$n 명령 생성 실패"; continue; }
  echo "$H" > $D/commit.txt
  say "$n 시작 (카드 $GPU, 다른 계산 프로세스 $(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU timeout ${FPA_TMO:-5400} bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/log.txt 2>&1; rc=$?
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
