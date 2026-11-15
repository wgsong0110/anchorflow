#!/bin/bash
# 절 CM184 검증: hexff 수정 (exe/hexff_wrap.py --hff 1, nlwhy 바로 앞) 이 PG 장면 (ficus · bread · wolf) 결과를 바꾸는가. 코드 디렉토리에서.
# 사용: bash exe/go_cm184pg.sh <커밋> <카드> <판 목록 쉼표>   한 카드에서 차례로 (단독). 판 디렉토리 /home/dkta/work/cm184pg/<판> -> R2 lbours/wolfcv/cm184pg/<판>.
# 판 이름:
#   FA_n · BA_n = 절 CM157 A 판 그대로 (exe/cm157_gen.py, 같은 기계 대조) · FH_n · BH_n = 같은 명령 + hexff (CM157_HFF=1)
#   W<접미>_n = wolf 기본 판 (R2 lbours/wolfcv/cm139/Kacdhnz/cmd.txt = exe/pr_ours_tmpl.json 의 wolf 원본) 에서 --out 만 바꿈, 접미 글자:
#     T = 시간 예산 끔 (--tb 1000, 절 CM140 T 와 같음) · H = hexff · M = --mlow 1e-3 -> 0 (절 CM184 장벽 바닥 장면 규칙)   예 WB_1 (그대로) WH_1 WHM_1 WTB_1 WTH_1 WTHM_1
export PATH=/home/dkta/bin:$PATH
H=$1; G=$2; RUNS=$3
R=r2:storage/result/anchorflow/lbours/wolfcv/cm184pg
O=/home/dkta/work/cm184pg
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
for s in F B; do rclone cat r2:storage/result/anchorflow/lbours/ext2/oA${s}_1/cmd.txt > $O/src$s.txt; [ $(wc -c < $O/src$s.txt) -gt 100 ] || { say "$s 원본 명령 못 읽음"; exit 1; }; done
rclone cat r2:storage/result/anchorflow/lbours/wolfcv/cm139/Kacdhnz/cmd.txt > $O/srcW.txt; [ $(wc -c < $O/srcW.txt) -gt 100 ] || { say "wolf 원본 명령 못 읽음"; exit 1; }
wcmd() { local n=${1%%_*} out=$2 C; C=$(cat $O/srcW.txt | tr -s ' \n' ' ' | sed -e 's/ *$//')
  C=$(echo "$C" | sed -e "s#--out [^ ]*\$#--out $out#")
  [[ ${n:1} == *T* ]] && C=$(echo "$C" | sed -e "s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#")
  [[ ${n:1} == *H* ]] && C=$(echo "$C" | sed -e "s#exe/nlwhy_wrap.py#exe/hexff_wrap.py --hff 1 exe/nlwhy_wrap.py#")
  [[ ${n:1} == *M* ]] && C=$(echo "$C" | sed -e "s#--mlow 1e-3 #--mlow 0 #")
  echo "$C"; }
mk() { local n=$1 s=${1:0:1} k=${1:1:1}
  mkdir -p $O/$n
  case $s in
    F|B) CM157_HFF=$([ "$k" = H ] && echo 1 || echo 0) python3 exe/cm157_gen.py $O/src$s.txt $s A $O/$n > $O/$n/cmd.txt || return 1;;
    W) wcmd $n $O/$n > $O/$n/cmd.txt;;
    *) return 1;; esac
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt || return 1
  local nh=$(grep -c 'exe/hexff_wrap.py --hff 1 exe/nlwhy_wrap.py' $O/$n/cmd.txt) want=0
  [[ ${n%%_*} == *H* ]] && want=1; [ "$nh" = "$want" ] || return 1
  [[ $s == W && ${n%%_*} == *M* ]] && { grep -q -- "--mlow 0 " $O/$n/cmd.txt || return 1; }
  [[ $s == W && ${n%%_*} == *T* ]] && { grep -q -- "--tb 1000 " $O/$n/cmd.txt || return 1; }
  return 0; }
runone() { local n=$1
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  say "$n 시작 카드 $G (계산 프로세스 $(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)) hexff $(grep -c 'hexff_wrap' $O/$n/cmd.txt) $(grep -o -- '--mlow [^ ]*' $O/$n/cmd.txt | head -1) $(grep -o -- '--tb [0-9.]* ' $O/$n/cmd.txt | tr '\n' ' ')"
  env CUDA_VISIBLE_DEVICES=$G timeout 3600 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) hff $(cat $O/$n/hffinfo.json 2>/dev/null | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림"; return 0; }
for n in ${RUNS//,/ }; do mk $n || { say "$n 명령 생성·확인 실패"; exit 1; }; done
for n in ${RUNS//,/ }; do runone $n; done
FB=""; WW=""; for n in ${RUNS//,/ }; do case $n in [FB]*) FB="$FB $n";; W*) WW="$WW $n";; esac; done
{ [ -n "$FB" ] && $PY -u exe/cm157_an.py $O $FB
  [ -n "$WW" ] && $PY -u exe/convabl_an.py $O $WW
  for n in ${RUNS//,/ }; do echo "$n hffinfo $(cat $O/$n/hffinfo.json 2>/dev/null || echo 없음) x_sha1 $(python3 -c "import json;print(json.load(open('$O/$n/summary.json')).get('x_sha1'))" 2>&1)"; done
} > $O/an_$(date +%H%M).md 2>&1; say "an rc=$?"
rclone copy $O $R --include "an_*.md" --include "convabl_an.json" -q
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
