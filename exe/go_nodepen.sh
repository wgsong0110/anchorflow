#!/bin/bash
# 절 CM111 묶음 nodepen 진입점 (exe/go_floorpen.sh 복사). 코드 디렉토리에서. 사용: bash exe/go_nodepen.sh <커밋> <단계>
#   단계 chk = 명령줄만 만들어 R2 에 (실행 안 함), run = aC_1 같은 명령 2 판 (nA1·nA2) + H8 같은 명령 1 판 (nH1) 에 노드 덤프 (카드 3 장 표식 잡고 동시에), an = 노드 덤프 분석 (CPU)
# 노드 / ·/tmp 는 안 씀: 코드·TMPDIR·캐시 /dev/shm/dkta, 덤프·결과 /home/dkta/work/nodepen (노드 덤프는 R2 에도).
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolfcv/nodepen
SRC=r2:storage/result/anchorflow/lbours/wolfh12
O=/home/dkta/work/nodepen; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_nodepen
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
CFG=/dev/shm/dkta/ag_data/gt_wolf/config.json; DATA=/dev/shm/dkta/ag_data/wolf.pt
mkcmd() {  # 이름 원본판 -> cmd 파일: 원본 cmd.txt 에서 --dump_F 뺌 (pred.pt 의 F 는 안 씀, 풀이 결과 불변) + exe/nodepen_wrap.py --npd 1 을 맨 앞 감싸개로 + --out 바꿈
  local n=$1 s=$2; mkdir -p $O/$n
  rclone cat $SRC/$s/cmd.txt | tr -d '\n' | sed  -e 's/ --dump_F / /' -e 's# -u exe/wolfh12_wrap.py # -u exe/nodepen_wrap.py --npd 1 exe/wolfh12_wrap.py #' -e "s# --out [^ ]*\$# --out $O/$n#" -e 's/  */ /g' > $O/$n/cmd.txt
  echo >> $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
}
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
runone() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $O/$n/nodes; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 노드 덤프 $(ls $O/$n/nodes 2>/dev/null | wc -l) 개 $(du -sh $O/$n/nodes 2>/dev/null | cut -f1) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --exclude "nodes/**" --transfers 8
  rclone copy $O/$n/nodes $R/$n/nodes -q --transfers 8
  say "$n R2 올림"; }
case $STG in
chk) mkcmd nA1 aC_1; mkcmd nA2 aC_1; mkcmd nH1 h8_1; for n in nA1 nA2 nH1; do say "$n: $(cat $O/$n/cmd.txt | cut -c1-6000)"; done;;
run) for n in nA1 nA2 nH1; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
  rm -f $O/go_run.END
  ( while [ ! -f $O/go_run.END ]; do sleep 120; for n in nA1 nA2 nH1; do [ -f $O/$n/log.txt ] && { { tail -c 3000 $O/$n/log.txt; echo; echo "노드 덤프 $(ls $O/$n/nodes 2>/dev/null | wc -l) 개 $(cat $O/$n/nodepen_err.txt 2>/dev/null | tail -2)"; } | rclone rcat $R/$n/log_tail.txt; }; done; done ) &
  for n in nA1 nA2 nH1; do runone $n & sleep 20; done; wait;;
an)  for n in nA1 nA2 nH1; do
       [ -s $O/$n/pred.pt ] || { say "$n pred.pt 없음"; continue; }
       $PY -u exe/nodepen.py $n $O/$n $CFG $DATA $O/nodepen_$n.json > $O/nodepen_$n.log 2>&1
       say "$n an rc=$? $(tail -1 $O/nodepen_$n.log | cut -c1-200)"
     done
     rclone copy $O $R --include "nodepen_*" -q;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
