#!/bin/bash
# 절 CM26 묶음 wolfprop: wolf 기본 판 (B, 장벽 바닥) 은 f15~22 에 무엇으로 몸을 받치는가 -- 결정적, 50 프레임, --nrec --prec 10:30 (결과 불변 진단) (잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2).
#   B = 절 CM25 base 명령 + --prec, A = 절 CM25 A_stick 명령 + --prec. 결과 불변 = 절 CM25 base·A_stick 프레임 해시와 대조 (표 0 절).
# 카드: 공용 잡기 규약 (/dev/shm/dkta/anerr_claim/<카드>, 그 카드 계산 프로세스 0 일 때만), 카드 하나에 판 하나. 판이 끝나면 바로 R2.
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WD=/tmp/dkta/ag_data; [ -s $WD/wolf.pt ] || { WD=/dev/shm/dkta/ag_data; mkdir -p $WD; [ -s $WD/wolf.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt $WD/wolf.pt.$$ -q && mv $WD/wolf.pt.$$ $WD/wolf.pt; }; }
FB="--floor vel --fl_bar --fl_bst 1 --y0eb"
REST="--pbc --mlow 1e-3 --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8 --cell 2"
IT="--newton 2 --cg 5"; TL="--tolN 1e-2 --tolP 1e-1"; TAIL="--fp_interp --tag wolf --K 1 --det --hgt --fhash"
cmdof() {   # cmdof 판 프레임 진단구간
  case $1 in
    B) echo "--data $WD/wolf.pt --frames $2 $FB $REST $IT $TL $TAIL --bar_cnt --nrec --prec $3";;
    A) echo "--data $WD/wolf.pt --frames $2 --floor vel $REST $IT $TL $TAIL --nrec --prec $3";;
  esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfprop.log; rclone copyto $O/wolfprop.log $RO/wolfprop.log -q; }
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
run() {    # run 판 카드 프레임 진단구간 출력
  local n=$1 g=$2 fr=$3 pr=$4 od=$5
  [ -f $od/summary.json ] && return
  rm -rf $od; mkdir -p $od
  local c="exe/fem_an_rt_sim.py $(cmdof $n $fr $pr) --out $od"
  echo "$PY -u $c" > $od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $od/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작 (프레임 $fr, 진단 $pr)"
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $od/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $od/log.txt | tail -1 | cut -c1-200) 진단 줄 $(wc -l < $od/prec.jsonl 2>/dev/null) 실패 $(grep -c '"err"' $od/prec.jsonl 2>/dev/null) $(grep -a 'rror' $od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
}
# ---- 연기 (12 프레임, 진단 10:12)
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  rm -rf $SM/B $SM/A
  for n in B A; do ( g=$(claim_wait $BASHPID); run $n $g 12 10:12 $SM/$n; rm -f $CL/$g ) & sleep 3; done; wait
  OK=1
  for n in B A; do [ -f $SM/$n/summary.json ] && [ "$(wc -l < $SM/$n/prec.jsonl 2>/dev/null)" = 3 ] && ! grep -q '"err"' $SM/$n/prec.jsonl || OK=0; done
  log "[연기] OK=$OK $(for q in B A; do echo "$q: $(grep -a 'rror\|Trace' $SM/$q/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ') $(grep -o '"err": "[^"]*' $SM/$q/prec.jsonl 2>/dev/null | head -1 | tail -c 600)"; done)"
  rclone copy $SM $RO/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wolfprop.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판 (결정적, 카드마다 하나씩 동시에)
for n in B A; do ( g=$(claim_wait $BASHPID); run $n $g 50 10:30 $O/$n; rm -f $CL/$g; rclone copy $O/$n $RO/$n -q --transfers 4 ) & sleep 3; done
wait
log "판 끝"
