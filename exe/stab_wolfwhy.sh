#!/bin/bash
# 절 CM25 묶음 wolfwhy: wolf 새 기본 판 (장벽 바닥 + 보간 교체, K 1 = 절 CM23 W_K1 명령) 이 왜 안 무너지는가 -- 결정적, 50 프레임 (잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2).
#   base0 = W_K1 그대로 (+ --hgt --bar_cnt --fhash), base = + 진단 --nrec (결과 불변: 프레임 해시 base0 와 같아야) + 덤프
#   대조: a_noy0eb (--y0eb 끔; 시작점 = U 에서 ŷ 쪽 실현 가능 최대 2^-k = --ga_y0ut 표준), b_nostep (--bar_nostep: 뉴턴 보폭의 장벽 실현 가능 자름 끔),
#         e_n8p20 (뉴턴 8 x PCG 20 만), A_stick ((A) 가우스 점 고착 바닥 = --fl_bar --fl_bst 1 --y0eb 뺌). (c) 뒤집힘 필터는 재질 판 경로에 없음 -> 해당 없음, (d) = (a).
#   렌더: base 와 무너진 판 (f50 질량중심 z < 0.70 m) 덤프 -> h5 -> 3DGS (exe/lb_render.py --shape wolf).
# 카드: 공용 잡기 규약 (/dev/shm/dkta/anerr_claim/<카드>, 그 카드 계산 프로세스 0 일 때만), 카드 하나에 판 하나. 판이 끝나면 바로 R2.
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work
WD=/tmp/dkta/ag_data; [ -s $WD/wolf.pt ] || { WD=/dev/shm/dkta/ag_data; mkdir -p $WD; [ -s $WD/wolf.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt $WD/wolf.pt.$$ -q && mv $WD/wolf.pt.$$ $WD/wolf.pt; }; }
FB="--floor vel --fl_bar --fl_bst 1 --y0eb"
REST="--pbc --mlow 1e-3 --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8 --cell 2"
IT="--newton 2 --cg 5"; TL="--tolN 1e-2 --tolP 1e-1"; TAIL="--fp_interp --tag wolf --K 1 --det --hgt --fhash"
cmdof() {   # cmdof 판 -> 인자
  case $1 in
    base0)    echo "--data $WD/wolf.pt --frames 50 $FB $REST $IT $TL $TAIL --bar_cnt";;
    base)     echo "--data $WD/wolf.pt --frames 50 $FB $REST $IT $TL $TAIL --bar_cnt --nrec --dump --dump_F";;
    a_noy0eb) echo "--data $WD/wolf.pt --frames 50 --floor vel --fl_bar --fl_bst 1 $REST $IT $TL $TAIL --bar_cnt --nrec --dump --dump_F";;
    b_nostep) echo "--data $WD/wolf.pt --frames 50 $FB $REST $IT $TL $TAIL --bar_cnt --nrec --dump --bar_nostep";;
    e_n8p20)  echo "--data $WD/wolf.pt --frames 50 $FB $REST --newton 8 --cg 20 $TL $TAIL --bar_cnt --nrec --dump --dump_F";;
    A_stick)  echo "--data $WD/wolf.pt --frames 50 --floor vel $REST $IT $TL $TAIL --nrec --dump --dump_F";;
  esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfwhy.log; rclone copyto $O/wolfwhy.log $RO/wolfwhy.log -q; }
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
run() {    # run 판 카드 [프레임]
  local n=$1 g=$2 fr=${3:-50} od=$O/$1; [ -n "${3:-}" ] && od=$O/smoke/$1
  [ -f $od/summary.json ] && return
  rm -rf $od; mkdir -p $od
  local c="exe/fem_an_rt_sim.py $(cmdof $n | sed "s/--frames 50/--frames $fr/") --out $od"
  echo "$PY -u $c" > $od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $od/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작 (프레임 $fr)"
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $od/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $od/log.txt | tail -1 | cut -c1-200) 해시 $(grep -o '"x_sha1": "[0-9a-f]*"' $od/summary.json 2>/dev/null) $(grep -a 'rror' $od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  [ -z "${3:-}" ] && rclone copy $od $RO/$n -q --transfers 4
}
# ---- 연기 (6 프레임: 1·2 는 순차 방출 구속으로 뉴턴 0): base0·base 프레임 해시 같고 nr 기록 있음, 나머지 판 실행만
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  for n in base0 base a_noy0eb b_nostep e_n8p20 A_stick; do ( g=$(claim_wait $BASHPID); run $n $g 6; rm -f $CL/$g ) & sleep 3; done; wait
  H0=$(grep -o '"xh": "[0-9a-f]*"' $SM/base0/fem.jsonl 2>/dev/null | cut -d'"' -f4 | tr '\n' ' '); H1=$(grep -o '"xh": "[0-9a-f]*"' $SM/base/fem.jsonl 2>/dev/null | cut -d'"' -f4 | tr '\n' ' ')
  OK=1; [ -n "$H0" ] && [ "$H0" = "$H1" ] || OK=0
  grep -q '"nr": {"steps"' $SM/base/fem.jsonl 2>/dev/null || OK=0
  for n in base0 base a_noy0eb b_nostep e_n8p20 A_stick; do [ -f $SM/$n/summary.json ] || OK=0; done
  log "[연기] OK=$OK base0 [$H0] base [$H1] $(for q in base0 base a_noy0eb b_nostep e_n8p20 A_stick; do echo "$q: $(grep -a 'rror\|Trace' $SM/$q/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"; done)"
  rclone copy $SM $RO/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wolfwhy.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판 (결정적, 카드마다 하나씩 동시에)
for n in base base0 a_noy0eb b_nostep e_n8p20 A_stick; do ( g=$(claim_wait $BASHPID); run $n $g; rm -f $CL/$g ) & sleep 3; done
wait
log "판 끝"
$PY exe/wolfwhy_tab.py $O $WD/wolf.pt > $O/wolfwhy_table.md 2> $O/wolfwhy_table.err; rclone copy $O $RO -q --include "wolfwhy_*"
# ---- 렌더 (base + 무너진 판)
RG=$(claim_wait $$); log "렌더 카드 $RG"
AGD=/tmp/dkta/ag_data; GTD=$AGD/gt_wolf; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wolf
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GTD > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wolf_rend_tmp.pt; fi
for n in base a_noy0eb b_nostep e_n8p20 A_stick; do
  Od=$O/$n
  if [ $n != base ]; then
    cz=$($PY -c "import json;J=[json.loads(l) for l in open('$Od/fem.jsonl')];print(J[49]['hp'][0] if len(J)>=50 and J[49].get('hp') else 9)" 2>/dev/null)
    $PY -c "import sys; sys.exit(0 if float('${cz:-9}') < 0.70 else 1)" || { log "[렌더] $n 건너뜀 (f50 질량중심 z ${cz:-?})"; continue; }
  fi
  if [ -f $Od/pred.pt ] && [ ! -f $Od/vj/fem_wolf.mp4 ]; then
    $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
    mkdir -p $Od/vj
    CUDA_VISIBLE_DEVICES=$RG $PY -u exe/lb_render.py --sim $Od/h5 --config $GTD/config.json --shape wolf --out $Od/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
    log "[렌더] $n $(ls -la $Od/vj/fem_wolf.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r.log | cut -c1-200) $(tail -1 $Od/h5.log | cut -c1-200)"
    rclone copy $Od/vj $RO/$n/vj -q; rclone copy $Od/h5.log $RO/$n -q
  fi
done
rm -f $CL/$RG
$PY exe/wolfwhy_tab.py $O $WD/wolf.pt > $O/wolfwhy_table.md 2> $O/wolfwhy_table.err
log "WOLFWHY_END"
