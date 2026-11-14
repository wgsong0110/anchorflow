#!/bin/bash
# 절 CM140 묶음 (wolf 높이 과다 -- 뉴턴 Anderson 가속). 코드 디렉토리에서. 사용: bash exe/go_cm140.sh <커밋> <카드0 판들> <카드1 판들>   (판 이름 = 아래 cm140_cmd 의 변형 글자, 쉼표 구분; 반복은 이름 뒤 _2 _3)
#   위로 쌓기 = K (k5) 에 요소 더함 · 아래로 빼기 = KacdhnzEF (H9) 에서 요소 뺌. 원본 = R2 cm131/k5 명령 (기록 감싸개만 뺌, 50 프레임, --out 만 바꿈).
#   판 디렉토리 /home/dkta/work/cm140/<판> -> R2 lbours/wolfcv/cm140/<판> (덤프 포함, 안 지움). 끝에 exe/convabl_an.py 집계 + 정답 높이.
export PATH=/home/dkta/bin:$PATH
H=$1; G0=$2; G1=$3
R=r2:storage/result/anchorflow/lbours/wolfcv/cm140
O=/home/dkta/work/cm140
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
source exe/cm139_gen.sh
source exe/cm140_gen.sh
rclone cat r2:storage/result/anchorflow/lbours/wolfcv/cm131/k5/cmd.txt > $O/k5src.txt
[ $(wc -c < $O/k5src.txt) -gt 100 ] || { say "k5 명령 못 읽음"; exit 1; }
mk() { local n=$1; mkdir -p $O/$n; cm140_cmd $n $O/k5src.txt $O/$n > $O/$n/cmd.txt
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt && grep -q -- "--frames 50 " $O/$n/cmd.txt || { say "$n 명령 확인 실패"; return 1; }
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 명령: $(grep -o 'anderson_wrap.py --anx [0-9]\|--newton [0-9]*\|AF_[EG]1_DG=1\|nlwhy_wrap.py\|speed13_wrap.py\|wolfh1[12]_wrap.py[^e]*\|wolfh9_wrap.py\|finterp_wrap.py --fi 0 --fidg 1\|speed12_wrap.py --s12z 1\|ext2_wrap.py --w8[a-z] [0-9]*\( --w8[a-z] [0-9]*\)*\|--w5[sc] 1' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 60; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 3600 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함)"; return 0; }
line() { local g=$1; shift; for n in "$@"; do runone $n $g; done; }
for n in ${G0//,/ } ${G1//,/ }; do mk $n || exit 1; done
( line 0 ${G0//,/ } ) & P1=$!; sleep 15; ( line 1 ${G1//,/ } ) & P2=$!; wait $P1 $P2
RUNS=""; for n in ${G0//,/ } ${G1//,/ }; do [ -s $O/$n/summary.json ] && RUNS="$RUNS $n"; done
$PY -u exe/convabl_an.py $O $RUNS > $O/an_$(date +%H%M).log 2>&1; say "an rc=$?"
$PY -u exe/cm140_an.py /dev/shm/dkta/ag_data/wolf.pt $O $RUNS > $O/an140_$(date +%H%M).log 2>&1; say "an140 rc=$?"
rclone copy $O $R --include "an_*.log" --include "an140_*.log" --include "cm140_an.json" --include "convabl_an.json" -q
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
