#!/bin/bash
# 절 CM56 묶음 t2chain: 시간 예산 T2 덜 풀린 걸음 연쇄 끊기 (exe/t2chain_wrap.py --tc M) 를 공통 후보 F + A3 위에서. 잡 안, 코드 디렉토리에서.
# exe/stab_speed4.sh (절 CM50) 틀 + 맨 앞 감싸개 exe/t2chain_wrap.py ('@tc M') 가 speed4_wrap 자리 ('@s4...') -> escape4_wrap ('@e5...') -> escape3 ('@e3...') -> escape2 -> escape1/speed3 (나머지 '@').
# 노드 / 와 /tmp 는 안 씀: 자료·덤프·임시 = /dev/shm/dkta (사용률 70 % 넘으면 판 시작을 기다림), 덤프 (wolf·ficus pred.pt) 는 R2 크기 확인 뒤 로컬 사본만 지움.
# 사용: bash exe/stab_t2chain.sh <출력> <R2> <단계> (판 목록 exe/t2chain_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식 z: 빈 카드 (계산 프로세스 0) 하나에 판 하나 -- 카드 여러 장 (최대 NPAR) 에 나눠 동시, 판마다 시작·끝 nvidia-smi·부하 기록. 진단 E1 (AF_E1_DG=1, 프레임 사이 읽기만) 늘 켬.
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
NPAR=${NPAR:-3}
source exe/fem_cm2_scn.sh
AG=/dev/shm/dkta/ag_data; mkdir -p $AG                                                      # /tmp 쪽 자료를 쓰지 않음
for t in L000 ficus bread wolf; do cm_data $t; done
WMD=$AG
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --gq_oct 0 --ws_bar 4 --ws_fi"
DFL="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1"   # 공통 후보 D (절 CM37 C + 절 CM38 X3c)
base_of() { case $1 in
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M3) echo "${WMB/--cell 1.5/--cell 3}";;
esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
dfl() { local s=$1 x=" $2 "                                                                 # exe/stab_speed4.sh 와 같은 낱말 풀기
  x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { case $1 in M*) echo "--erec";; W) echo "--hgt --dump --dump_part";; F) echo "--dump";; *) echo "";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/t2chain_$STG.log; rclone copyto $O/t2chain_$STG.log $RO/t2chain_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt; }
shm_wait() { while [ "$(df --output=pcent /dev/shm | tail -1 | tr -dc 0-9)" -ge 70 ]; do log "/dev/shm 사용률 $(df --output=pcent /dev/shm | tail -1) -- 기다림"; sleep 60; done; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL                                                   # 다른 묶음과 같은 카드 표식 디렉토리 (서로 비켜감)
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
run() {   # run 이름 씬 방식 카드 플래그
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b wo="" w2="" w3="" w4="" w5="" tc=""
  [ -f $O/$n/summary.json ] && return
  tc="$(echo " $x " | grep -o '@tc [0-9]*' | awk '{print $2}')"
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+-]*' | grep -v '^@\(e[235]\|s4\|tc\)' | sed 's/^@/--/' | tr '\n' ' ')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s) $x"
  c="exe/t2chain_wrap.py --tc ${tc:-0} $w4 exe/escape4_wrap.py $w5 $w3 $w2 $wo $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "AF_E1_DG=1 $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; echo "카드 $g 계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_start.txt; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  log "$n 카드 $g 시작 (부하 $(cut -d' ' -f1 /proc/loadavg), /dev/shm $(df --output=pcent /dev/shm | tail -1 | tr -d ' '))"
  env AF_E1_DG=1 CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_end.txt; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -1 $O/$n/period.txt 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  if [ -f $O/$n/pred.pt ]; then                                                              # 덤프: R2 에 올리고 크기 확인 뒤 로컬 사본만 지움
    rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
    [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && rm -f $O/$n/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"
  fi; }
SPEC=$(grep -v '^#' exe/t2chain_spec.txt | awk -v s=$STG '$1==s')
log "단계 $STG 시작: 판 $(echo "$SPEC" | grep -c .), 동시 카드 최대 $NPAR, 노드 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NPAR ]; do sleep 5; done
  shm_wait
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판 끝 $STG"
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나고 기록·훅 확인, 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $n 5 프레임 못 감/발산"; }
    grep -q "절 CM56 t2chain\] 모드" $O/$n/log.txt || { OK=0; log "[연기] $n CM56 줄 없음"; }
    [ ! -s $O/$n/t2c_err.txt ] && [ "$(wc -l < $O/$n/t2c.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n t2c.jsonl 없음/오류"; }
    [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n e1 없음/오류"; }
    case " $x " in *" @tc "[2367]*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/t2c.jsonl')]; m=max(q['ins'] for q in L); print('$n tc_all 넣은 수', m, '자른 걸음 몸 전체', sum(q['nall'] for q in L), '자름', sum(q['cut'] for q in L)); sys.exit(0 if m > 0 else 1)" >> $O/t2chain_$STG.log 2>&1 || { OK=0; log "[연기] $n tc_all 안 들어감"; };; esac
    $PY -c "import json; L=[json.loads(l) for l in open('$O/$n/t2c.jsonl')]; print('$n 잔고 ms', [q['bank'] for q in L][:12], '한도', [q['lim'] for q in L][:12], '자름', [q['cut'] for q in L][:12])" >> $O/t2chain_$STG.log 2>&1
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/t2chain.FAIL; rclone copy $O $RO -q --include t2chain.FAIL; };;
esac
$PY exe/t2chain_tab.py $O > $O/t2chain_table.md 2> $O/t2chain_table.err; rclone copy $O $RO -q --include "t2chain_table.*"
log "단계 끝 $STG"
