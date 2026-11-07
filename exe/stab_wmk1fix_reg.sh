#!/bin/bash
# 절 CM24 묶음 wmk1fix 2 단계: 수박에서 효과 있는 판을 레고·ficus·wolf·bread 기본 판 (K 1, 절 CM23 k1new 명령 그대로) 에 켜서 회귀 (오차·시간).
#   씬마다: off = k1new 판정 명령 그대로 (새 코드, 새 플래그 없음) -> 끝 해시가 k1new <씬>_K1 과 같아야 (끄면 비트 같음, 결정적 --det),
#           <판> = 같은 판정 명령 + 판 플래그 (결정적 --det + 결과 불변 진단), 단독 시간 = 비결정·진단 없음, 카드 하나에 차례로 (씬마다 off a · 판 a · off b · 판 b).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2, 판 이름 $3, 판 플래그 = 나머지 인자.
O=$1; R=$2; VN=$3; VF="${*:4}"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
[ -n "$VF" ] || { echo "판 플래그 없음 ($VN)" > $O/wmk1fix.FAIL; exit 1; }
# 앞 실행 (go 가 판 플래그를 안 넘김) 의 같은 이름 판은 플래그가 명령에 없으면 noflag/ 로 옮김 (지우지 않음, R2 도 같이)
for s_ in L F W B; do for n in r_${s_}_$VN z_${s_}_${VN}_a z_${s_}_${VN}_b; do
  [ -f $O/$n/cmd.txt ] && ! grep -qF -- "$VF" $O/$n/cmd.txt && { mkdir -p $O/noflag; mv $O/$n $O/noflag/$n; rclone move $R/$n $R/noflag/$n -q; echo "[$(date '+%F %T')] $n 플래그 없는 앞 실행 -> noflag/" >> $O/reg_$VN.log; }
done; done
[ -f $O/smoke_$VN/L/cmd.txt ] && ! grep -qF -- "$VF" $O/smoke_$VN/L/cmd.txt && { mkdir -p $O/noflag; mv $O/smoke_$VN $O/noflag/smoke_$VN; rclone move $R/smoke_$VN $R/noflag/smoke_$VN -q; }
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
for t in L000 ficus wolf bread; do cm_data $t; done
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
base_of() { case $1 in                                                                         # 절 CM23 stab_k1new.sh base_of 글자 그대로
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread";;
esac; }
dg_of() { case $1 in                                                                           # 절 CM23 판정 판 진단 그대로
  L) echo "--det --pdiag --qjd --fhash";;
  F) echo "--det --pdiag --qjd --fhash --dump";;
  W) echo "--det --hgt --bar_cnt --pdiag --qjd --fhash";;
  B) echo "--det --pdiag --qjd --fhash";;
esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/reg_$VN.log; rclone copyto $O/reg_$VN.log $R/reg_$VN.log -q; }
up() { rclone copy $O/$1 $R/$1 -q --transfers 8 --s3-upload-concurrency 16 --s3-chunk-size 64M; }
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
run() {   # run 이름 씬 추가플래그 카드
  local n=$1 s=$2 x=$3 g=$4 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) --K 1 $(dg_of $s) $x"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
# ---- 연기: 판 플래그로 씬마다 3 프레임
SM=$O/smoke_$VN; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  for s in L F W B; do ( g=$(claim_wait $BASHPID); c="$(base_of $s | sed 's/--frames [0-9]*/--frames 3/') --K 1 $(dg_of $s | sed 's/--dump//') $VF"
    mkdir -p $SM/$s; echo "$PY -u $c --out $SM/$s" > $SM/$s/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$s < /dev/null > $SM/$s/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$s/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for s in L F W B; do
    v=$($PY -c "import json; s=json.load(open('$SM/$s/summary.json')); print(int(s['frames_done'] == 3 and not s.get('diverged')))" 2>/dev/null || echo 0)
    [ "$v" = 1 ] || OK=0; MSG="$MSG $s:$v $(cat $SM/$s/rc.txt) $(grep -a 'rror\|Trace' $SM/$s/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"; done
  log "[연기 $VN] OK=$OK$MSG"; rclone copy $SM $R/smoke_$VN -q
  [ $OK = 1 ] || { echo "연기 실패 $VN $(date '+%F %T')" > $O/wmk1fix.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판정 판 (씬마다 off·판, 카드마다 하나)
for s in B F W L; do
  for v in off $VN; do x=""; [ $v = $VN ] && x="$VF"
    ( me=$BASHPID; g=$(claim_wait $me); run r_${s}_$v $s "$x" $g; rm -f $CL/$g ) & sleep 3
  done; done
wait
# 끄면 비트 같음: off 끝 해시 = k1new <씬>_K1 끝 해시
for s in L F W B; do
  rclone copyto r2:storage/result/anchorflow/lbours/k1new/${s}_K1/summary.json $O/r_${s}_off/k1new_summary.json -q
  log "[끔 대조] $s off $($PY -c "import json; print(json.load(open('$O/r_${s}_off/summary.json')).get('x_sha1'))" 2>&1) k1new $($PY -c "import json; print(json.load(open('$O/r_${s}_off/k1new_summary.json')).get('x_sha1'))" 2>&1)"
  up r_${s}_off; done
log "판정 판 끝 $VN"
# ---- 단독 시간 (카드 하나, 차례로; 비결정·진단 없음)
zrun() { local n=$1 s=$2 x=$3 g=$4 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) --K 1 $x"; rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  log "$n 카드 $g $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"; up $n; }
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for s in L F W B; do for r in a b; do zrun z_${s}_off${VN}_$r $s "" $ZG; zrun z_${s}_${VN}_$r $s "$VF" $ZG; done; done          # off 시간도 판마다 (같은 카드에서 판과 번갈아; 판 묶음끼리 이름 안 겹침)
rm -f $CL/$ZG
log "단계 끝 reg $VN"
