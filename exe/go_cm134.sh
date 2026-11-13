#!/bin/bash
# 절 CM134 묶음 (wolf k5 높이 과다 몫 가르기 + 고침). 코드 디렉토리에서. 사용: bash exe/go_cm134.sh <커밋> [카드0 판들] [카드1 판들] [렌더 판들]
#   원본 = R2 cm131/k5 명령 (H5 − 고스트 강성) 에서 기록 감싸개 (resmode·escwhy, ESC_*·RM_* env) 만 빼고 --out 만 바꿈.
#   판 (한 가지씩 바꿈, 몫 가르기 = 진단):
#     b5 원본 · r1 HHT ρ∞ 1.0 · ns 바닥 고착 끔 (--fl_nost) · sc S_구속 + C (--w5s 1 --w5c 1, 절 CM70 정답 격자 구속 받침) · s1 S_구속만 · c1 C 만
#     cv 수렴 (뉴턴 60 · PCG 100 · tolN 1e-4 · tolP 1e-3 · 예산 끔 --tb 1000, --xit_from·--xg 뺌, 절 CM125 cvb 와 같은 조건)
#     k4 프레임당 걸음 4 (--K 4, --ksplit 3 --ks_big 뺌 -- K 1 전용)
#     h1 셀 1 (해상도 몫; --cell 2 -> 1) · gk ghost penalty 강성만 0 (--w5gk0 1, 절 CM70 진단; 수렴 위에서 매달림 몫 가르기)
#     y<판> = 같은 판 14 프레임 (낙하 가속 측정용 짧은 판). 빼기 시험 (cv_sc 위): hcx --hc 5 --tc 8 · mfx --mf2 · barx --fl_bar --y0eb · pbx --pbc --bar_cnt · l3x --l3* · mlx --mlow 1e-6 · fpx --fp_interp · ibe --integ be
#     조합: 판 이름을 _ 로 이어 붙임 (예 cv_sc = 수렴 + S_구속 + C), 고침 판 x* 는 맨 끝 토막
#     x* 고침 판 (exe/cm134_wrap.py <플래그>, ext2_wrap.py 앞): flags() 참고. z* = 같은 판 6 프레임 연기.
#   판 디렉토리 /home/dkta/work/cm134/<판> -> R2 lbours/wolfcv/cm134/<판> (덤프 포함, 안 지움). 끝에 an (convabl_an · cm134_an · sinv_esc) · 렌더.
export PATH=/home/dkta/bin:$PATH
H=$1; G0=${2:-b5,r1,ns}; G1=${3:-sc,cv,k4}; VR=${4:-}
R=r2:storage/result/anchorflow/lbours/wolfcv/cm134
O=/home/dkta/work/cm134
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
SRC=r2:storage/result/anchorflow/lbours/wolfcv/cm131/k5/cmd.txt
# 판 -> sed 식 (원본 글자 고침) ; 고침 판의 감싸개 플래그는 wflags
edits1() { case $1 in
  b5) echo "s#^##";;
  r1) echo "s#--rho 0.8 #--rho 1.0 #";;
  ns) echo "s#--bar_cnt #--bar_cnt --fl_nost #";;
  sc) echo "s#--l3fr 1 --s9c 1#--l3fr 1 --w5s 1 --w5c 1 --s9c 1#";;
  s1) echo "s#--l3fr 1 --s9c 1#--l3fr 1 --w5s 1 --s9c 1#";;
  c1) echo "s#--l3fr 1 --s9c 1#--l3fr 1 --w5c 1 --s9c 1#";;
  cv) echo "s#--newton 6 --cg 5 --tolN 1e-2 --tolP 1e-1#--newton 60 --cg 100 --tolN 1e-4 --tolP 1e-3#;s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#;s# --xit_from 2##;s# --xg 1##";;
  k4) echo "s#--K 1 #--K 4 #;s# --ksplit 3 --ks_big##";;
  h1) echo "s#--cell 2 #--cell 1 #";;
  gk) echo "s#--s9c 1 --s9y 1#--w5gk0 1 --s9c 1 --s9y 1#";;
  hcx) echo "s# --hc 5 --tc 8##";;
  mfx) echo "s# --mf2 1##";;
  barx) echo "s# --fl_bar --fl_bst 1 --y0eb##";;
  pbx) echo "s# --pbc##;s# --bar_cnt##";;
  l3x) echo "s#--l3 10 --l3sc 1 --l3fr 1 ##";;
  mlx) echo "s#--mlow 1e-3#--mlow 1e-6#";;
  fpx) echo "s# --fp_interp##";;
  ibe) echo "s# --ga hht # #;s#--rho 0.8 #--integ be #;s# --fp_interp##";;
  x*) echo "s#^##";;
  *) echo "?";; esac; }
edits() { local n=${1#[zy]} e o=""; for t in ${n//_/ }; do e=$(edits1 $t); [ "$e" = "?" ] && { echo "?"; return; }; o="$o;$e"; done; echo "${o#;}"; }
wflags() { local n=${1#[zy]}; case ${n##*_} in
  x1) echo "--xpc 1";;
  x2) echo "--xpc 1 --xfl 1";;
  x3) echo "--xfl 1";;
  *) echo "";; esac; }
chk1() { case $1 in   # 고친 글자 확인
  b5) grep -q -- "--rho 0.8 " $2;; r1) grep -q -- "--rho 1.0 " $2;; ns) grep -q -- "--fl_nost" $2;; sc) grep -q -- "--w5s 1 --w5c 1" $2;;
  s1) grep -q -- "--w5s 1 --s9c" $2 && ! grep -q -- "--w5c" $2;; c1) grep -q -- "--w5c 1 --s9c" $2 && ! grep -q -- "--w5s" $2;;
  cv) grep -q -- "--newton 60 --cg 100 --tolN 1e-4 --tolP 1e-3" $2 && grep -q -- "--tb 1000 exe/" $2 && ! grep -q -- "--xit_from" $2;;
  k4) grep -q -- "--K 4 " $2 && ! grep -q -- "--ksplit" $2;; h1) ! grep -q -- "--cell 2 " $2;; gk) grep -q -- "--w5gk0 1 --s9c" $2;; hcx) ! grep -q -- "--hc 5" $2;; mfx) ! grep -q -- "--mf2" $2;; barx) ! grep -q -- "--fl_bar" $2;; pbx) ! grep -q -- "--pbc" $2;; l3x) ! grep -q -- "--l3 10" $2;; mlx) grep -q -- "--mlow 1e-6" $2;; fpx) ! grep -q -- "--fp_interp" $2;; ibe) grep -q -- "--integ be" $2;; x*) grep -q -- "exe/cm134_wrap.py $(wflags $1) exe/ext2_wrap.py" $2;; *) false;; esac; }
chk() { local n=${1#[zy]} t; for t in ${n//_/ }; do chk1 $t $2 || return 1; done; }
mk() { local n=$1 E C W; E=$(edits $n); [ "$E" = "?" ] && { say "$n 모르는 판"; return 1; }
  C=$(rclone cat $SRC | tr -d "\n")
  [ ${#C} -gt 100 ] || { say "k5 명령 못 읽음"; return 1; }
  mkdir -p $O/$n
  C=$(echo "$C" | sed -e 's#^ESC_F0=5 ESC_F1=30 RM_F0=5 RM_F1=30 ##' -e 's#exe/resmode_wrap.py exe/escwhy_wrap.py ##' -e "$E")
  W=$(wflags $n); if [ -n "$W" ]; then C=$(echo "$C" | sed -e "s#exe/speed12_wrap.py exe/ext2_wrap.py#exe/speed12_wrap.py exe/cm134_wrap.py $W exe/ext2_wrap.py#"); fi
  case $n in z*) C=$(echo "$C" | sed -e 's#AF_SP_FR=1,50#AF_SP_FR=1,6#' -e 's# --frames 50 # --frames 6 #');; y*) C=$(echo "$C" | sed -e 's#AF_SP_FR=1,50#AF_SP_FR=1,14#' -e 's# --frames 50 # --frames 14 #');; esac
  echo "$C" | sed -e "s#--out [^ ]*\$#--out $O/$n#" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  grep -q "resmode_wrap\|escwhy_wrap\|ESC_F0" $O/$n/cmd.txt && { say "$n 기록 감싸개가 남음"; return 1; }
  chk $n $O/$n/cmd.txt || { say "$n 명령 고침 확인 실패"; return 1; }
  case $n in y*) grep -q -- " --frames 14 " $O/$n/cmd.txt || { say "$n 프레임 고침 확인 실패"; return 1; };; z*) grep -q -- " --frames 6 " $O/$n/cmd.txt || { say "$n 프레임 고침 확인 실패"; return 1; };; *) grep -q -- " --frames 50 " $O/$n/cmd.txt || { say "$n 프레임 50 아님"; return 1; };; esac
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt || { say "$n --out 확인 실패"; return 1; }
  say "$n 명령: $(grep -o 'cm134_wrap.py[^e]*exe/ext2_wrap.py\|--rho [0-9.]*\|--fl_nost\|--w5[sc] 1\|--K [0-9]*\|--frames [0-9]*\|--tb [0-9.]*\|--newton [0-9]*\|--cg [0-9]*\|--out [^ ]*' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 90; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(cat $O/$n/xcinfo.json 2>/dev/null | cut -c1-300) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함)"; return $rc; }
line() { local g=$1; shift; for n in "$@"; do runone $n $g || { case $n in z*) say "연기 $n 실패 -- 카드 $g 줄 멈춤"; return 1;; esac; }; done; }
for n in ${G0//,/ } ${G1//,/ }; do mk $n || exit 1; done
( line 0 ${G0//,/ } ) & P1=$!; sleep 20; ( line 1 ${G1//,/ } ) & P2=$!; wait $P1 $P2
RUNS=""; for n in ${G0//,/ } ${G1//,/ }; do case $n in z*) ;; *) [ -s $O/$n/summary.json ] && RUNS="$RUNS $n";; esac; done
$PY -u exe/convabl_an.py $O $RUNS > $O/an.log 2>&1; say "an rc=$? $(cat $O/an.log | tr '\n' ' ' | cut -c1-3000)"
DIRS=""; for n in $RUNS; do DIRS="$DIRS $O/$n"; $PY -u exe/sinv_esc.py $O/$n >> $O/an.log 2>&1; done
$PY -u exe/cm134_an.py /dev/shm/dkta/ag_data/wolf.pt $O/cm134_an.json $DIRS > $O/an134.log 2>&1; say "an134 rc=$? $(cat $O/an134.log | tr '\n' ' ' | cut -c1-4000)"
rclone copy $O $R --include "an.log" --include "an134.log" --include "convabl_an.json" --include "cm134_an.json" -q
for n in $RUNS; do rclone copy $O/$n $R/$n -q --include "esc.json"; done
GTW=/dev/shm/dkta/ag_data/gt_wolf; g=0
for n in ${VR//,/ }; do
  [ -s $O/$n/pred.pt ] || { say "$n pred.pt 없음 -- 렌더 안 함"; continue; }
  gg=$g; ( V=$O/$n/vj; HD=/dev/shm/dkta/cm134_vid/$n; mkdir -p $V $HD
    $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$gg $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
    rm -rf $HD
    say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160)"
    rclone copy $V $R/$n/vj -q ) &
  g=$(( (g + 1) % 2 )); [ $g = 0 ] && wait
done; wait
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
