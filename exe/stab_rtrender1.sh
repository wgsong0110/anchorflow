#!/bin/bash
# 절 CM54 묶음 rtrender1: 시뮬 + 가우시안 갱신 + 3DGS 래스터화 프레임 시간 (사용자 11-08 '실시간 렌더까지 포함해서 실시간'). 잡 안, 커밋 디렉토리에서.
# 판 명령 = exe/stab_speed4.sh 의 run() 과 같은 글자 (speed4_wrap -> escape4_wrap -> ...) 앞에 감싸개 exe/rtrender1_wrap.py ('--rr...').
# 사용: bash exe/stab_rtrender1.sh <출력> <R2> <단계>   (판 목록 exe/rtrender1_spec.txt: '<단계> <이름> <씬> <방식> <갱신 0|1> <플래그...>')
#   씬: L 레고 (L000) · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식: z 단독 시간 (계산 프로세스 0 인 카드 하나에 차례로 a · b) · v 확인 (동시, 덤프 + 확인 프레임 PNG -> 영상 렌더 경로 (lb_roll_to_h5 + lb_render / wm_render) 와 화면 대조)
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work; CD=$(pwd -P)
source exe/fem_cm2_scn.sh
for t in L000 ficus bread wolf; do cm_data $t; done
eval "$(sed -n '/^BA0=/p;/^FBB=/p;/^TL=/p;/^DFL=/p;/^base_of()/,/^esac; }/p;/^nw()/p;/^dfl()/,/echo "\$x"; }/p' exe/stab_speed4.sh)"
RD=/dev/shm/dkta/rtrender1; mkdir -p $RD/data
WMD=/tmp/dkta/ag_data; [ -s $WMD/wm.pt ] || WMD=/dev/shm/dkta/ag_data; [ -s $WMD/wm.pt ] || WMD=$RD/data       # /tmp 에 쓰지 않음 (노드 / 98 %)
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
eval "$(sed -n '/^WMB=/p' exe/stab_speed4.sh)"                                                  # 수박 바탕 (절 CM50 과 같은 글자, 자료 경로만 위 WMD)
gtd() { local n=$1 d; for d in /tmp/dkta/ag_data/gt_$n /dev/shm/dkta/ag_data/gt_$n $RD/gt_$n; do [ -f $d/meta.json ] && [ -f $d/sim_0000000000.h5 ] && { echo $d; return; }; done; echo $RD/gt_$n; }
GTW=$(gtd wolf); GTB=$(gtd bread); GTM=$(gtd wm)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; WCF=$W/wm2/gf/config.json
FGT=$W/lbours/ficus_v0/gt
rr_of() { case $1 in
  L) echo "--rr pg --rr_shape lego --rr_traj $W/lbdata/test_low/L000 --rr_cfg $W/lbdata/config.json";;
  F) echo "--rr pg --rr_shape ficus --rr_traj $FGT --rr_cfg $FGT/config.json";;
  W) echo "--rr pg --rr_shape wolf --rr_traj $GTW --rr_cfg $GTW/config.json";;
  B) echo "--rr pg --rr_shape bread --rr_traj $GTB --rr_cfg $GTB/config.json";;
  M3) echo "--rr gf --rr_shape watermelon --rr_traj $GTM --rr_cfg $WCF --rr_model $WMM --rr_gf $W/GaussianFluent";;
esac; }
chk_of() { case $1 in M*) echo "10,20,30,40,50,60,70,80,90,100";; L) echo "1,10,20,30,40,50,60,70,80";; W) echo "1,10,20,30,40,50";; F) echo "1,10,30,60,90,125";; B) echo "1,10,50,100,200,300";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/rtrender1_$STG.log; rclone copyto $O/rtrender1_$STG.log $RO/rtrender1_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt --exclude "h5/**" --exclude "offpng/**"; }
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
# 정답 궤적 디렉토리 (meta.json R·0 프레임 h5 만; 남의 것은 읽기만, 없으면 $RD 에 만듦 -- CPU 변환, 앞 1 프레임만)
mkgt() { local n=$1 d=$2; shift 2; [ -f $d/meta.json ] && return
  CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py "$@" --out_pt $RD/tmp_$n.pt --out_dir $d --out_dir_n 1 > $RD/mkgt_$n.log 2>&1; rm -f $RD/tmp_$n.pt
  log "정답 궤적 $n -> $d $(tail -1 $RD/mkgt_$n.log | cut -c1-160)"; }
mkgt wolf $GTW --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json
mkgt bread $GTB --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json
mkgt wm $GTM --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --allow_nonfinite
log "자료 L000 $SD ficus $FI ficus 궤적 $FGT ($(ls $FGT/meta.json 2>&1)) wolf $GTW bread $GTB 수박 $WMD/wm.pt 궤적 $GTM 모델 $WMM"
run() {   # run 이름 씬 방식 갱신 카드 플래그
  local n=$1 s=$2 md=$3 fs=$4 g=$5 x="$(dfl $2 "$6")" c b wo="" w2="" w3="" w4="" w5="" dg=""
  [ -f $O/$n/summary.json ] && return
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+-]*' | grep -v '^@\(e[235]\|s4\)' | sed 's/^@/--/' | tr '\n' ' ')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  case $s in M*) dg="--erec";; esac                                                     # 수박 단독 판은 절 CM50 과 같게 --erec
  local rk=""
  if [ $md = v ]; then rk="--rr_chk $(chk_of $s)"; case $s in M*) dg="$dg --dump";; *) dg="$dg --dump --dump_F --dump_part";; esac; fi
  c="exe/escape4_wrap.py $w5 $w3 $w2 $wo $b $dg $x"
  [ -n "$w4" ] && c="exe/speed4_wrap.py $w4 $c"
  c="exe/rtrender1_wrap.py $(rr_of $s) --rr_fast $fs $rk $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; echo "카드 $g 계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_start.txt; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_end.txt; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-160) $(grep -ah 'CM54 rtrender1\] [pg]' $O/$n/log.txt | tail -1 | cut -c1-160) 프레임 기록 $(wc -l < $O/$n/rr.jsonl 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  if [ $md = v ] && [ -f $O/$n/pred.pt ]; then                                          # 영상 렌더 경로로 같은 덤프를 그려 PNG 대조
    case $s in
    M*) $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTM --out $O/$n/h5 > $O/$n/h5.log 2>&1
        (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $WCF --sim $O/$n/h5 --out $O/$n/offpng/off.mp4 < /dev/null > $O/$n/off.log 2>&1)
        PD=$O/$n/offpng/off_frames; PF="%03d";;
    *)  local SH TR CF; read -r _ _ _ SH _ TR _ CF <<< "$(rr_of $s)"
        $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $TR --out $O/$n/h5 > $O/$n/h5.log 2>&1
        mkdir -p $O/$n/offpng; CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $O/$n/h5 --config $CF --shape $SH --out $O/$n/offpng/off.mp4 --png $O/$n/offpng < /dev/null > $O/$n/off.log 2>&1
        PD=$O/$n/offpng; PF="%04d";;
    esac
    $PY exe/rtrender1_cmp.py --a $O/$n/rrimg --b $PD --bfmt "$PF" > $O/$n/cmp.json 2> $O/$n/cmp.err
    log "$n 대조 $(head -c 400 $O/$n/cmp.json) $(tail -1 $O/$n/off.log | cut -c1-120)"
    rm -rf $O/$n/h5 $O/$n/offpng/*.png $O/$n/pred.pt
  fi
  up $n; }
[ -f exe/rtrender1_pre_$STG.sh ] && { log "앞 단계 스크립트 rtrender1_pre_$STG.sh"; bash exe/rtrender1_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
SPEC=$(grep -v '^#' exe/rtrender1_spec.txt | awk -v s=$STG '$1==s')
# ---- 확인 판 (v): 카드마다 하나 동시
while read -r st n s md fs x; do
  [ -z "$n" ] && continue; [ $md = v ] || continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $fs $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
# ---- 단독 (z): 계산 프로세스 0 인 카드 하나에 차례로 (a 다 돌고 b)
if echo "$SPEC" | awk '$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG (노드 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  for r in a b; do
    while read -r st n s md fs x; do
      [ "$md" = z ] || continue
      [ -f $O/${n}_$r/summary.json ] && continue
      while [ "$(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; do log "카드 $ZG 에 다른 계산 프로세스 -- 기다림"; sleep 30; done
      log "${n}_$r 단독 카드 $ZG 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s z $fs $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
log "판 끝 $STG"
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 + rr.jsonl 프레임 기록, 아니면 실패 표식
  OK=1; while read -r st n s md fs x; do [ -z "$n" ] && continue
    case $md in z) nn=${n}_a;; *) nn=$n;; esac
    $PY -c "import json,sys; s=json.load(open('$O/$nn/summary.json')); sys.exit(0 if s['frames_done'] >= 5 else 1)" 2>/dev/null || { OK=0; log "[연기] $nn 5 프레임 못 감"; }
    [ "$(wc -l < $O/$nn/rr.jsonl 2>/dev/null)" -ge 5 ] 2>/dev/null || { OK=0; log "[연기] $nn rr.jsonl 프레임 기록 없음"; }
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/rtrender1.FAIL; rclone copy $O $RO -q --include rtrender1.FAIL; };;
esac
$PY exe/rtrender1_tab.py $O > $O/rtrender1_table.md 2> $O/rtrender1_table.err; rclone copy $O $RO -q --include "rtrender1_table.*"
log "단계 끝 $STG"
