#!/bin/bash
# 절 CM113 묶음 startpt 진입점 (exe/go_cellforce.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_startpt.sh <커밋> <단계[,단계]>
#   chk = 명령줄만, s0 = 연기 (H9 5 프레임 k9, 기록 프레임 3~5), run = H9 wolf (p9_1, wolfh12/h8_1 명령) + H5 wolf (p5_1, wolfbar/b5_1 명령) 각 12 프레임 (기록 f3~f12), an = 집계 (CPU)
#   기록 감싸개 exe/startpt_wrap.py --sp 1 를 가장 바깥에, --dump·--dump_part·--dump_F 만 뺌, --frames 12 (50 프레임 판의 앞 12 프레임과 같은 계산).
# 노드 / ·/tmp 는 안 씀: 코드·TMPDIR·캐시·결과 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_startpt.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/startpt
RB=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/startpt; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_startpt
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
mkcmd() {  # 이름 원본 앞 감싸개 문자열 [프레임]
  local n=$1 s=$2 pre=$3 fr=$4; mkdir -p $O/$n
  rclone cat $RB/$s/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e 's/$/ /' -e "s# -u $pre # -u exe/startpt_wrap.py --sp 1 $pre #" -e "s# --out [^ ]* \$# --out $O/$n#" \
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
  rm -rf $O/$n/sp; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 개 $(du -sh $O/$n/sp 2>/dev/null | cut -f1) 기록 오류 $(python3 -c "import json;d=json.load(open('$O/$n/spinfo.json'));print(d.get('err'), d.get('n_snapdrop'))" 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8
  say "$n R2 올림 (칸 파일 sp/ 포함, f3~f12 만이라 작음)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 120; for n in "$@"; do [ -f $O/$n/log.txt ] && { { tail -c 3000 $O/$n/log.txt; echo; echo "칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 개"; tail -c 1500 $O/$n/spinfo.json 2>/dev/null; } | rclone rcat $R/$n/log_tail.txt; }; done; done ) & }
case $STG in
chk) mkcmd p9_1 wolfh12/h8_1 exe/wolfh12_wrap.py 12; mkcmd p5_1 wolfbar/b5_1 'exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py' 12; mkcmd k9 wolfh12/h8_1 exe/wolfh12_wrap.py 5
  sed -i 's# -u exe/startpt_wrap.py --sp 1 exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py # -u exe/startpt_wrap.py --sp 1 exe/speed13_wrap.py #' $O/p5_1/cmd.txt
  rclone copyto $O/p5_1/cmd.txt $R/p5_1/cmd.txt -q
  for n in k9 p9_1 p5_1; do say "$n: $(cat $O/$n/cmd.txt | cut -c1-6000)"; done;;
s0)  [ -s $O/k9/cmd.txt ] || { say "k9 cmd.txt 없음 -- chk 먼저"; exit 1; }
     rm -f $O/go_s0.RUNEND; watch $O/go_s0.RUNEND k9
     runone k9; touch $O/go_s0.RUNEND
     $PY -u exe/startpt_an.py $O k9 > $O/an_k9.log 2>&1; say "k9 an rc=$? $(tail -4 $O/an_k9.log | tr '\n' ' ' | cut -c1-900)"
     rclone copy $O $R --include "an_k9.log" --include "startpt_k9.json" -q;;
run) for n in p9_1 p5_1; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     rm -f $O/go_run.RUNEND; watch $O/go_run.RUNEND p9_1 p5_1
     PIDS=""; for n in p9_1 p5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run.RUNEND;;
an)  $PY -u exe/startpt_an.py $O p9_1 p5_1 > $O/an.log 2>&1; say "an rc=$? $(tail -3 $O/an.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an.log" --include "startpt_*.json" -q;;
run2) for n in q9_1 q5_1; do s_=p${n:1}; mkdir -p $O/$n; sed "s# --out [^ ]*\$# --out $O/$n#" $O/$s_/cmd.txt | sed "s#--out $O/$s_ #--out $O/$n #" > $O/$n/cmd.txt; rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q; say "$n: $(cut -c1-300 $O/$n/cmd.txt) ... $(grep -o -- '--out [^ ]*' $O/$n/cmd.txt)"; done
     rm -f $O/go_run2.RUNEND; watch $O/go_run2.RUNEND q9_1 q5_1
     PIDS=""; for n in q9_1 q5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run2.RUNEND;;
an2) $PY -u exe/startpt_an.py $O q9_1 q5_1 > $O/an2a.log 2>&1; say "an2a rc=$?"
     $PY -u exe/startpt2_an.py $O q9_1 q5_1 > $O/an2.log 2>&1; say "an2 rc=$? $(tail -3 $O/an2.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an2*.log" --include "startpt_q*.json" --include "startpt2_*.json" -q;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
