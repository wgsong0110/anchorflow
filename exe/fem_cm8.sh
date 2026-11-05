#!/bin/bash
# 절 CM8 묶음 (커밋 $1, 잡 130567 일꾼 w2; 사용자 승인 11-05 정의 변경 둘, 씬 파일 exe/fem_cm8_scn.sh --v=a|b):
#   (A) 모든 바닥 씬 바닥 = 가우스 점 고착 (장벽 대신), (B) 장벽 유지 + 시작점 섞기 β 장벽 에너지 기준 (--y0eb). 나머지는 절 CM4 채택 공통 설정 그대로
#   (HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, BA = exe/fem_cm3_scn.sh, 시작점 관성 예측 --ga_y0ut, 점유 가중, 예산 규칙).
#  0. 연기 시험 (결정적, 빈 카드): a·b x L000·0015·wolf 12 프레임
#  T. 단독 시간표 (카드 0, gpu0.lock, 허용치 0, 예산 규칙 = 절 CM2.0 (1); 잡 만료 때문에 c 3·6·12 -> 맞춤 c*±1 한 번 (원래 규칙은 c 3·6·12·24 -> 두 번)), 씬 L000·0016·0019·wolf; lego 상한 = 셋 중 작은 값, 낙하 = lego 값
#  J. 판정 (결정적, 빈 카드 1-6) -- 시간표와 동시에 먼저 cm4 상한 (lego·낙하 2x6, wolf 2x5) 으로 (a6·b6), 시간표가 다른 c 를 고르면 그 c 로 (a<c>·b<c>); ficus 2x24·bread 2x2 는 두 변형 같은 명령이라 한 번 (o)
# 결과 R2 lbours/fem_cm/cm8s·cm8z_a·cm8z_b·cm8j, 표 marks/cm8_table.md (판정 + 시간표), 끝 표식 marks/cm8.END. 12:21 KST 뒤 새 판 안 띄움, 12:27 KST 에 표·끝 표식 (돌던 판이 있어도), 12:31 에 표 한 번 더.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm8 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
B=/dev/shm/dkta/lbours/fem_cm; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; L=/dev/shm/dkta/cm_lists/cm8; mkdir -p $M $L $CUPY_CACHE_DIR
LOG=$M/cm8.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
NOSTART=$(date -d '2026-11-05 12:21 KST' +%s); TBL=$(date -d '2026-11-05 12:27 KST' +%s); TBL2=$(date -d '2026-11-05 12:31 KST' +%s)
say "[cm8 시작] $C $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
export CM_SCN=exe/fem_cm8_scn.sh
source exe/fem_cm8_scn.sh
LG="L000 $(seq -f %04g 0 19 | tr '\n' ' ')"
for t in $LG drop1 ficus wolf bread; do cm_data $t; done
G="--ga hht --rho 0.8 --K 2 --newton 2"; Z="--tolN 0 --tolP 0"; T="--tolN 1e-2 --tolP 1e-1"
table() {
  { echo "# cm8 ($C) -- 절 CM8: (A) 바닥 = 가우스 점 고착, (B) 장벽 + 시작점 섞기 β 장벽 에너지 기준 (--y0eb); 나머지 절 CM4 공통 설정, 결정적 판정 ($(date '+%F %T') 기준)"; echo
    echo "## 단독 시간표 (예산 규칙, 허용치 0, 카드 0 단독; 씬 c 최대 (평균) ms, x = 발산)"
    echo "- (A)"; $PY exe/fem_cm2_sel.py pick $B/cm8z_a L000 0016 0019 wolf 2>&1 | sed 's/^/  /'
    echo "- (B)"; $PY exe/fem_cm2_sel.py pick $B/cm8z_b L000 0016 0019 wolf 2>&1 | sed 's/^/  /'
    echo "- 고른 상한: $(cat $M/cm8_caps.txt 2>/dev/null)"; echo
    echo "## 판정 (결정적; 판 = 변형 + PCG 상한 c (K 2 x 뉴턴 2), 칸 = 평균 오차 % 또는 x발산 프레임 (발산 전 평균); 낙하 정답 반발 0.750 / 83.4 mm; ficus·bread 는 o = cm4 와 같은 명령)"
    $PY exe/fem_cm8_table.py $B/cm8j $(ls $B/cm8j 2>/dev/null | grep -E '^[ab][0-9]+_' | sed 's/_.*//' | sort -u)
    echo; echo "## 연기 시험"; $PY exe/fem_cm_table.py judge $B/cm8s 2>&1
  } > $M/cm8_table.md 2>&1
  rclone copy $M $MR -q --include "cm8*"; }
( while [ $(date +%s) -lt $TBL ]; do sleep 20; done; table
  rclone copy $B/cm8j r2:storage/result/anchorflow/lbours/fem_cm/cm8j -q --exclude pred.pt
  echo "[cm8 끝] $(date '+%F %T') (마감 표; 돌던 판은 표에 '도는 중')" > $M/cm8.END; rclone copy $M/cm8.END $MR -q; say "[표·끝 표식] $(date '+%T')"
  while [ $(date +%s) -lt $TBL2 ]; do sleep 20; done; table; say "[표 다시] $(date '+%T')" ) &
# 0 연기
: > $L/s.txt; for v in a b; do for t in L000 0015 wolf; do echo "$v $t $G --cg 6 $T --det --frames 12 --v=$v" >> $L/s.txt; done; done
CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=3 bash exe/fem_cm_pool.sh cm8s $L/s.txt > $M/cm8s_pool.log 2>&1
for v in a b; do for t in L000 0015 wolf; do say "  연기 $v $t: $(grep -a '롤아웃\|rror\|발산' $B/cm8s/${v}_$t/log.txt 2>/dev/null | tail -2 | cut -c1-200 | tr '\n' ' ')"; done; done
OKV=""; for v in a b; do bad=0; for t in L000 0015 wolf; do grep -q '"err_mean"' $B/cm8s/${v}_$t/summary.json 2>/dev/null || bad=1; done; [ $bad = 0 ] && OKV="$OKV $v"; done
say "[연기 통과 변형]$OKV $(date '+%T')"
[ -z "$OKV" ] && { table; echo "[cm8 멈춤 -- 연기 실패] $(date '+%F %T')" > $M/cm8.END; rclone copy $M/cm8.END $MR -q; exit 1; }
# J 먼저 (cm4 상한) -- 발산 6 궤적 먼저
DV="0001 0006 0014 0015 0017 0018"
jl() { local v=$1 c=$2 cw=$3 x; for x in $DV L000 0016 0019 drop1; do echo "$v$c $x $G --cg $c $T --det --v=$v"; done
  echo "$v$cw wolf $G --cg $cw $T --det --hgt --v=$v"
  for x in $LG; do case " $DV L000 0016 0019 " in *" $x "*) continue;; esac; echo "$v$c $x $G --cg $c $T --det --v=$v"; done; }
: > $L/j1.txt; for v in $OKV; do jl $v 6 5 | head -11 >> $L/j1.txt; done
for v in $OKV; do jl $v 6 5 | tail -n +12 >> $L/j1.txt; done
echo "o ficus $G --cg 24 $T --det" >> $L/j1.txt; echo "o bread $G --cg 2 $T --det" >> $L/j1.txt
cat $L/j1.txt >> $LOG
( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=3 bash exe/fem_cm_pool.sh cm8j $L/j1.txt > $M/cm8j1_pool.log 2>&1; say "[판정 1 끝] $(date '+%T')" ) &
PJ1=$!
# T 시간표 (카드 0)
SC="L000 0016 0019 wolf"
for v in $OKV; do : > $L/z1$v.txt; for c in 3 6 12; do for t in $SC; do echo "z${c}_$t $t $G --cg $c $Z --v=$v" >> $L/z1$v.txt; done; done
  CM_TMO=300 bash exe/fem_cm_q.sh cm8z_$v $L/z1$v.txt > $M/cm8z_${v}_q1.log 2>&1; say "[시간표 1 $v] $(date '+%T')"; done
for v in $OKV; do [ $(date +%s) -ge $NOSTART ] && break
  $PY exe/fem_cm2_sel.py next $B/cm8z_$v $SC | awk -v g="$G" -v z="$Z" -v v=$v '{print "z"$1"_"$2, $2, g, "--cg", $1, z, "--v="v}' > $L/z2$v.txt
  [ -s $L/z2$v.txt ] && { CM_TMO=300 bash exe/fem_cm_q.sh cm8z_$v $L/z2$v.txt > $M/cm8z_${v}_q2.log 2>&1; say "[시간표 2 $v] $(wc -l < $L/z2$v.txt) 판 $(date '+%T')"; }; done
: > $M/cm8_caps.txt
for v in $OKV; do $PY exe/fem_cm2_sel.py pick $B/cm8z_$v $SC > $M/cm8_sel_$v.txt 2>&1; cat $M/cm8_sel_$v.txt >> $LOG
  cl=$(awk '$1=="L000"||$1=="0016"||$1=="0019"{print $2}' $M/cm8_sel_$v.txt | sort -n | head -1); cw=$(awk '$1=="wolf"{print $2}' $M/cm8_sel_$v.txt)
  echo -n "$v: lego·낙하 $cl wolf $cw; " >> $M/cm8_caps.txt; done
say "[상한] $(cat $M/cm8_caps.txt)"; rclone copy $M $MR -q --include "cm8*"
# J 둘째: 고른 상한이 cm4 상한과 다르면 그 상한으로 (12:21 KST 전까지만 띄움)
: > $L/j2.txt
for v in $OKV; do cl=$(sed -n "s/.*$v: lego·낙하 \([0-9]*\) wolf \([0-9]*\).*/\1/p" $M/cm8_caps.txt); cw=$(sed -n "s/.*$v: lego·낙하 \([0-9]*\) wolf \([0-9]*\).*/\2/p" $M/cm8_caps.txt)
  [ -n "$cl" ] && [ "$cl" != 0 ] && [ "$cl" != 6 ] && jl $v $cl ${cw:-5} | grep -v " wolf " >> $L/j2.txt
  [ -n "$cw" ] && [ "$cw" != 0 ] && [ "$cw" != 5 ] && echo "$v$cw wolf $G --cg $cw $T --det --hgt --v=$v" >> $L/j2.txt; done
if [ -s $L/j2.txt ] && [ $(date +%s) -lt $NOSTART ]; then
  cat $L/j2.txt >> $LOG
  CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=3 bash exe/fem_cm_pool.sh cm8j $L/j2.txt > $M/cm8j2_pool.log 2>&1
  say "[판정 2 끝] $(date '+%T')"
fi
wait $PJ1
table; rclone copy $B/cm8j r2:storage/result/anchorflow/lbours/fem_cm/cm8j -q --exclude pred.pt
[ $(date +%s) -lt $TBL ] && { echo "[cm8 끝] $(date '+%F %T') (모두 완주)" > $M/cm8.END; rclone copy $M/cm8.END $MR -q; say "[cm8 끝] $(date '+%F %T')"; }
wait
