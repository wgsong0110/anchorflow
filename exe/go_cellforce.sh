#!/bin/bash
# 절 CM111 묶음 cellforce 진입점 (exe/go_cellh.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_cellforce.sh <커밋> <단계>
#   chk = 명령줄만 만들어 R2 에 (실행 안 함), s0 = 연기 (H9 8 프레임 k9), run = H9 wolf 1 판 (g9_1, 원본 wolfh12/h8_1 명령) + H5 wolf 1 판 (g5_1, 원본 wolfbar/b5_1 명령), an = 칸 구간 집계 (CPU)
#   기록 감싸개 exe/cellforce_wrap.py --cf 1 를 가장 바깥에, --dump·--dump_part·--dump_F 만 뺌 (pred.pt 덤프 없음; 계산 그대로).
# 노드 / ·/tmp 는 안 씀: 코드·TMPDIR·캐시·결과 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_cellforce.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/cellforce
RB=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/cellforce; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cellforce
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
mkcmd() {  # 이름 원본 앞 감싸개 문자열 [프레임]
  local n=$1 s=$2 pre=$3 fr=$4; mkdir -p $O/$n
  rclone cat $RB/$s/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e 's/$/ /' -e "s# -u $pre # -u exe/cellforce_wrap.py --cf 1 $pre #" -e "s# --out [^ ]* \$# --out $O/$n#" \
    -e 's# --dump_part # #g' -e 's# --dump_F # #g' -e 's# --dump # #g' > $O/$n/cmd.txt
  [ -n "$fr" ] && sed -i "s# --frames [0-9]* # --frames $fr #" $O/$n/cmd.txt
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
  rm -rf $O/$n/cf; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $O/$n/cf 2>/dev/null | wc -l) 개 $(du -sh $O/$n/cf 2>/dev/null | cut -f1) 기록 오류 $(python3 -c "import json;print(json.load(open('$O/$n/cfinfo.json')).get('err'))" 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --exclude "cf/**" --transfers 8
  say "$n R2 올림 (칸 파일 cf/ 은 an 뒤 고른 프레임만)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 120; for n in "$@"; do [ -f $O/$n/log.txt ] && { { tail -c 3000 $O/$n/log.txt; echo; echo "칸 파일 $(ls $O/$n/cf 2>/dev/null | wc -l) 개"; tail -c 1500 $O/$n/cfinfo.json 2>/dev/null; } | rclone rcat $R/$n/log_tail.txt; }; done; done ) & }
case $STG in
chk) mkcmd g9_1 wolfh12/h8_1 exe/wolfh12_wrap.py; mkcmd g5_1 wolfbar/b5_1 'exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py'; mkcmd k9 wolfh12/h8_1 exe/wolfh12_wrap.py 8
  sed -i 's# -u exe/cellforce_wrap.py --cf 1 exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py # -u exe/cellforce_wrap.py --cf 1 exe/speed13_wrap.py #' $O/g5_1/cmd.txt
  rclone copyto $O/g5_1/cmd.txt $R/g5_1/cmd.txt -q
  for n in k9 g9_1 g5_1; do say "$n: $(cat $O/$n/cmd.txt | cut -c1-6000)"; done;;
s0)  [ -s $O/k9/cmd.txt ] || { say "k9 cmd.txt 없음 -- chk 먼저"; exit 1; }
     rm -f $O/go_s0.RUNEND; watch $O/go_s0.RUNEND k9
     runone k9; touch $O/go_s0.RUNEND
     $PY -u exe/cellforce_an.py $O k9 > $O/an_k9.log 2>&1; say "k9 an rc=$? $(tail -3 $O/an_k9.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an_k9.log" --include "cellforce_k9.json" -q; rclone copy $O/k9/cf $R/k9/cf -q;;
run) for n in g9_1 g5_1; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     rm -f $O/go_run.RUNEND; watch $O/go_run.RUNEND g9_1 g5_1
     PIDS=""; for n in g9_1 g5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run.RUNEND;;
an)  $PY -u exe/cellforce_an.py $O g9_1 g5_1 > $O/an.log 2>&1; say "an rc=$? $(tail -3 $O/an.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an.log" --include "cellforce_*.json" -q
     for n in g9_1 g5_1; do for f in 5 8 9 10 15 20 25 30; do ff=$(printf "f%03d.npz" $f); [ -f $O/$n/cf/$ff ] && rclone copyto $O/$n/cf/$ff $R/$n/cf/$ff -q; done; done
     say "고른 프레임 칸 파일 올림 (f5 8 9 10 15 20 25 30)";;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
