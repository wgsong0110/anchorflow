#!/bin/bash
# 절 CM22 묶음 wolfnew: wolf 를 새 기본 판 = (B) 장벽 바닥 (절 CM8 b 판 명령 그대로) + --fp_interp (절 CM20 정의를 재질 판에: 마지막 걸음도 가우스 점 커밋 RM 뒤 F·재질 상태 교체) 로 (잡 안, 커밋 디렉토리에서; 출력 $1).
#   b_off = (B) 그대로 + 결과 불변 진단 (--bar_cnt --pdiag --qjd --fhash, 덤프) -- 끝 해시 = 절 CM8 cm8j b5_wolf (fa318a68b4) 이면 (B) 기존 결과 재사용 확인
#   b_on  = b_off + --fp_interp (결정적)
#   단독 시간: 판정 명령에서 --det·--hgt·진단·덤프 뺀 비결정 판, 카드 하나에 끔·켬 번갈아 두 번 (그 카드·노드 계산 프로세스 수 10 초마다)
#   렌더: b_on 덤프 -> h5 -> 3DGS (exe/lb_render.py --shape wolf); (A)·(B) 영상은 기존 (R2 barstick/a_wolf·barx/x_wolf vj) 재사용
# 카드: 공용 잡기 규약 (/dev/shm/dkta/anerr_claim/<카드>, 그 카드 계산 프로세스 0 일 때만), 카드 하나에 판 하나. 판이 끝나면 바로 R2 ($RO/<판>).
O=$1; mkdir -p $O
RO=r2:storage/result/anchorflow/lbours/wolfnew
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work
WD=/tmp/dkta/ag_data; [ -s $WD/wolf.pt ] || { WD=/dev/shm/dkta/ag_data; mkdir -p $WD; [ -s $WD/wolf.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt $WD/wolf.pt.$$ -q && mv $WD/wolf.pt.$$ $WD/wolf.pt; }; }
B="--data $WD/wolf.pt --frames 50 --floor vel --fl_bar --fl_bst 1 --y0eb --pbc --mlow 1e-3 --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --cell 2 --ga hht --rho 0.8 --K 2 --newton 2 --cg 5 --tolN 1e-2 --tolP 1e-1"
DG="--det --hgt --bar_cnt --pdiag --qjd --fhash --dump --dump_F"
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfnew.log; rclone copyto $O/wolfnew.log $RO/wolfnew.log -q; }
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
run() {    # run 이름 카드 추가인자...
  local n=$1 g=$2; shift 2
  [ -f $O/$n/summary.json ] && return
  rm -rf $O/$n; mkdir -p $O/$n
  echo "$PY -u exe/fem_an_rt_sim.py $B --tag wolf --out $O/$n $*" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u exe/fem_an_rt_sim.py $B --tag wolf --out $O/$n "$@" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 해시 $(grep -o '"x_sha1": "[0-9a-f]*"' $O/$n/summary.json 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $RO/$n -q
}
# ---- 연기 (3 프레임): 켬 컴파일·실행, 끔·켬 프레임 1 해시 같고 (교체는 다음 프레임 P2G 부터) 3 은 다름 (1·2 는 구속으로 풀이 없음)
SM=$O/smoke; mkdir -p $SM
smk() { local n=$1 g me=$BASHPID; shift
  g=$(claim_wait $me); mkdir -p $SM/$n
  CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u exe/fem_an_rt_sim.py $(echo $B | sed 's/--frames 50/--frames 3/') --tag wolf --out $SM/$n --det --fhash --bar_cnt --pdiag --qjd "$@" < /dev/null > $SM/$n/log.txt 2>&1
  echo "rc=$? 카드 $g" > $SM/$n/rc.txt; rm -f $CL/$g; }
if [ ! -f $SM/ok ]; then
  smk off & sleep 5; smk on --fp_interp & wait
  HO=$(grep -o '"xh": "[0-9a-f]*"' $SM/off/fem.jsonl 2>/dev/null | cut -d'"' -f4 | tr '\n' ' '); HN=$(grep -o '"xh": "[0-9a-f]*"' $SM/on/fem.jsonl 2>/dev/null | cut -d'"' -f4 | tr '\n' ' ')
  OK=1; set -- $HO; A1=$1; A3=$3; set -- $HN; N1=$1; N3=$3
  [ -f $SM/off/summary.json ] && [ -f $SM/on/summary.json ] || OK=0
  [ -n "$A1" ] && [ -n "$A3" ] && [ "$A1" = "$N1" ] && [ "$A3" != "$N3" ] || OK=0           # wolf 는 프레임 1·2 가 순차 방출 구속으로 풀이 0 (뉴턴 0) -> 프레임 3 에서 달라야 함
  log "[연기] OK=$OK 끔 [$HO] 켬 [$HN] $(for q in off on; do grep -a 'rror\|Trace' $SM/$q/log.txt | tail -2 | cut -c1-300 | tr '\n' ' '; done)"
  rclone copy $SM $RO/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wolfnew.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판정 (결정적): 두 판 각자 카드
( g=$(claim_wait $BASHPID); run b_on $g $DG --fp_interp; rm -f $CL/$g ) & sleep 3
( g=$(claim_wait $BASHPID); run b_off $g $DG; rm -f $CL/$g ) &
wait
log "판정 판 끝"
$PY exe/wolfnew_tab.py $O $WD/wolf.pt > $O/wolfnew_table.md 2> $O/wolfnew_table.err; rclone copy $O $RO -q --include "wolfnew_table.*"
# ---- 단독 시간 (비결정; 카드 하나 잡고 차례로)
zrun() { local n=$1 g=$2; shift 2
  [ -f $O/$n/summary.json ] && return
  mkdir -p $O/$n; echo "$PY -u exe/fem_an_rt_sim.py $B --tag wolf --out $O/$n $*" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u exe/fem_an_rt_sim.py $B --tag wolf --out $O/$n "$@" < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  log "$n 카드 $g $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"
  rclone copy $O/$n $RO/$n -q; }
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
zrun z_off_a $ZG; zrun z_on_a $ZG --fp_interp; zrun z_off_b $ZG; zrun z_on_b $ZG --fp_interp
rm -f $CL/$ZG
$PY exe/wolfnew_tab.py $O $WD/wolf.pt > $O/wolfnew_table.md 2> $O/wolfnew_table.err; rclone copy $O $RO -q --include "wolfnew_table.*"
# ---- 렌더 (b_on; (A)·(B) 는 기존 영상)
RG=$(claim_wait $$); log "렌더 카드 $RG"
AGD=/tmp/dkta/ag_data; GTD=$AGD/gt_wolf; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wolf
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GTD > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wolf_rend_tmp.pt; fi
Od=$O/b_on
if [ -f $Od/pred.pt ] && [ ! -f $Od/vj/fem_wolf.mp4 ]; then
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
  mkdir -p $Od/vj
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/lb_render.py --sim $Od/h5 --config $GTD/config.json --shape wolf --out $Od/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $Od/vj/r.log 2>&1
  log "[렌더] b_on $(ls -la $Od/vj/fem_wolf.mp4 2>&1 | cut -c1-160) $(tail -1 $Od/vj/r.log | cut -c1-200) $(tail -1 $Od/h5.log | cut -c1-200)"
  rclone copy $Od $RO/b_on -q
fi
rm -f $CL/$RG
$PY exe/wolfnew_tab.py $O $WD/wolf.pt > $O/wolfnew_table.md 2> $O/wolfnew_table.err
log "WOLFNEW_END"
