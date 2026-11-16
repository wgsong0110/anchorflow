#!/bin/bash
# [stabfinal CM192] 작업 대기열 일꾼 (인스턴스 안, 부팅 stabfinal_boot.sh 의 queue 단계가 띄움; 환경은 부팅이 만든 것 그대로).
#   목록 R2 lbours/stabfinal/queue/<목록>.txt, 한 줄 = "<id>|<장면>|<단독 0/1>|<stab1115_drv.py 인자 (따옴표 가능)>"  (# 로 시작하면 무시)
#   표식: queue/lock/<id> (잠금 = 일꾼 이름, 쓰고 5 초 뒤 다시 읽어 내 이름이면 내 것), queue/done/<id> (끝, rc), queue/fail/<id>
#   작업: R2 결과 받기 (그 장면, --update) -> 드라이버 -> 장면 결과 올리기 (PNG 포함) -> done 표시. 단독 작업은 /tmp/sfq_solo 를 세우고 다른 일꾼 작업이 끝날 때까지 기다린 뒤 돔
#   id 뒤 @선행1,선행2 = 선행 작업이 모두 done 일 때만 꺼냄.
#   인자가 ! 로 시작하면 셸 명령 그대로 (예 "!python -u exe/stab_extra.py --scene mic_metal_lego --hextag _f6").
#   목록에 남은 (잠기지 않은) 작업이 없으면 끝.
#   bash exe/sfq_worker.sh <목록> <일꾼 이름>
Q=$1; ME=$2
RX=r2:storage/result/anchorflow/lbours/stabfinal; QD=$RX/queue; OUT=/home/dkta/work/stab
LOG=/root/sf/sfq_$ME.log; mkdir -p /tmp/sfq_run
say() { echo "[sfq $ME $(date '+%F %T')] $*" | tee -a $LOG; }
EXC=(--exclude "**/vis/dump_*/**" --exclude "**/vis/hexdump_*/**" --exclude "hex_in.pt" --exclude "**/pred.pt" --exclude "**/*_tmp*" --exclude "**/*.ckpt.pt*" --exclude "*/gt/**" --exclude "*/traj0/**")
while true; do
  rclone cat $QD/$Q.txt 2>/dev/null | grep -v '^#' | grep '|' > /tmp/sfq_$ME.list
  DONE=$(rclone lsf $QD/done/ 2>/dev/null); LOCK=$(rclone lsf $QD/lock/ 2>/dev/null)
  pick=""
  while IFS= read -r line; do
    id=${line%%|*}; deps=""
    [[ "$id" == *@* ]] && { deps=${id#*@}; id=${id%%@*}; }                # id@선행1,선행2 = 선행 작업이 done 일 때만
    echo "$DONE" | grep -qx "$id" && continue
    ok=1; for dp in ${deps//,/ }; do echo "$DONE" | grep -qx "$dp" || ok=0; done; [ $ok = 1 ] || continue
    echo "$LOCK" | grep -qx "$id" && continue
    echo "$ME $(date +%s)" | rclone rcat $QD/lock/$id 2>/dev/null; sleep 5
    [ "$(rclone cat $QD/lock/$id 2>/dev/null | cut -d' ' -f1)" = "$ME" ] && { pick=$line; break; }
  done < /tmp/sfq_$ME.list
  if [ -z "$pick" ]; then
    IDLE=$(( ${IDLE:-0} + 1 )); [ $IDLE -ge 3 ] && { say "목록 $Q 에 남은 작업 없음 (5 분 간격 3 번) -> 올리기 끝 기다린 뒤 끝"; wait; break; }
    sleep 300; continue
  fi
  IDLE=0
  IFS='|' read -r id s solo args <<< "$pick"; id=${id%%@*}
  while [ -e /tmp/sfq_solo ]; do sleep 30; done                     # 다른 일꾼의 단독 작업 중이면 기다림
  if [ "$solo" = 1 ]; then
    touch /tmp/sfq_solo
    while [ -n "$(ls /tmp/sfq_run 2>/dev/null)" ]; do sleep 30; done  # 다른 일꾼 작업이 끝날 때까지
  fi
  touch /tmp/sfq_run/$ME
  say "[작업] $id $s 단독 $solo :: $args"
  rclone copy $RX/res/$s $OUT/$s -q --update --transfers 16 --exclude "drv*.log" --exclude "DRV_DONE_*" --exclude "config.json" --exclude "init/**" --exclude "gt/**" --exclude "hex_in.pt"
  t0=$(date +%s)
  if [ "${args:0:1}" = "!" ]; then eval "${args:1}" >> /root/sf/drv_${s}_$ME.log 2>&1; rc=$?          # ! = 셸 명령 그대로 (코드 폴더에서)
  else eval python -u exe/stab1115_drv.py --scene $s --par ${SF_PAR:-4} $args >> /root/sf/drv_${s}_$ME.log 2>&1; rc=$?; fi
  el=$(( $(date +%s) - t0 ))
  rm -f /tmp/sfq_run/$ME; [ "$solo" = 1 ] && rm -f /tmp/sfq_solo
  say "[끝] $id rc $rc ${el}s -- 올리기는 뒤에서 (다 올린 뒤 done 표시)"
  ( rclone copy $OUT/$s $RX/res/$s -q --update --transfers 8 "${EXC[@]}"                 # [CM192] 업로드는 백그라운드, GPU 는 바로 다음 작업
    rclone copy /root/sf $RX/marks/${SF_TAG:-q}/logs -q --include "*.log"
    [ $rc != 0 ] && echo "rc=$rc $ME" | rclone rcat $QD/fail/$id
    echo "rc=$rc ${el}s $ME" | rclone rcat $QD/done/$id; echo "[sfq $ME $(date '+%F %T')] [올림 끝] $id" >> $LOG ) &
done
