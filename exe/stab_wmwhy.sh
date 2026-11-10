#!/bin/bash
# 절 CM97 묶음 wmwhy (진단, 기본 판 안 바꿈): 수박 첫 충돌 f3~f8 다져짐·위 끝 인장 깨짐이 정답과 다른 원인 가르기 -- 셀 (1·2·3) · 바닥 (장벽 / 노드 정답식 / 노드 고착) · 탄성 (SNH / NHB) · 반환 사상 (가우스 점 / 입자) · 에너지 기록.
# 판 목록 exe/wmwhy_spec.txt. 판마다 카드 하나 (공용 표식 디렉토리 /dev/shm/dkta/anerr_claim, 계산 프로세스 0 인 카드만), 동시에. 끝나면 exe/wmwhy_prof.py (CPU, 읽기만) 로 층별·압력·손상·에너지 표.
# 노드 / ·/tmp 안 씀 (코드·자료·TMPDIR·결과 /dev/shm/dkta). /dev/shm 사용률 65 % 이상이면 판 시작을 기다림.
# 사용: bash exe/stab_wmwhy.sh <출력> <R2> <단계>
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
AG=/dev/shm/dkta/ag_data; mkdir -p $AG
[ -s $AG/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $AG/wm.pt.$$ -q && mv $AG/wm.pt.$$ $AG/wm.pt; }
JP=$AG/gt_jp; mkdir -p $JP
[ $(ls $JP | wc -l) -ge 101 ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $JP -q --transfers 16
GH5=/home/dkta/work/wm2/gf/simulation_ply
log() { echo "[$(date '+%F %T')] $*" >> $O/wmwhy_$STG.log; rclone copyto $O/wmwhy_$STG.log $RO/wmwhy_$STG.log -q; }
shm_use() { df --output=pcent /dev/shm | tail -1 | tr -dc '0-9'; }
log "시작 $(hostname) 코드 $(pwd -P) 단계 $STG; wm.pt $(stat -c %s $AG/wm.pt) gt_jp $(ls $JP | wc -l) GF h5 $(ls $GH5 | wc -l); /dev/shm $(shm_use) %"
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
cmd_of() { local c=$1 K=$2 fl=$3 el=$4 rm=$5 cg=$6 Od=$7 x
  x="exe/wmwhy_wrap.py --wyv 1 exe/s6/fem_ao_rt_sim.py --data $AG/wm.pt --frames 8 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell $c --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut"
  x="$x --tolN 1e-3 --tolP 1e-2 --newton 40 --cg $cg --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --tag wm --K $K"
  case $fl in bar) x="$x --fl_bar --fl_bst 1 --y0eb";; slip) x="$x --floor_slip";; stick) ;; esac
  case $el in snh) x="$x --elx snh";; nhb) ;; esac
  case $rm in gq) x="$x --lj_interp --fp_interp";; pp) ;; esac
  echo "$PY -u $x --erec --nrec --ufrec --wsh_rec 8 --wsh_sv 3,4,5,6,8 --out $Od"; }
one() { local n=$1 g=$2 Od=$O/$1 c
  shift 2
  [ -f $Od/summary.json ] && [ -s $Od/wsh/fs_008.npz ] && { log "$n 이미 있음"; return; }
  rm -rf $Od; mkdir -p $Od
  c=$(cmd_of "$@" $Od); echo "$c" > $Od/cmd.txt
  { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; echo "카드 $g"; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader; cat /proc/loadavg; } > $Od/apps_at_start.txt 2>&1
  while [ "$(shm_use)" -ge 65 ]; do log "$n /dev/shm $(shm_use) % -- 기다림"; sleep 120; done
  log "$n 카드 $g 시작: $*"
  local T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g timeout 14400 $c < /dev/null > $Od/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(( $(date +%s) - T0 )) s $(grep -a '프레임 8/8' $Od/log.txt | tail -1 | cut -c1-160) $(grep -a 'rror\|Killed\|memory' $Od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $Od $RO/$n -q --transfers 8; }
while read -r s n a1 a2 a3 a4 a5 a6; do [ -z "$s" ] && continue; case $s in \#*) continue;; esac
  [ "$s" = "$STG" ] || continue
  ( me=$BASHPID; g=$(claim_wait $me); one $n $g $a1 $a2 $a3 $a4 $a5 $a6; rm -f $CL/$g ) & sleep 20
done < exe/wmwhy_spec.txt
wait
NS=$(for d in $O/*/; do n=$(basename $d); [ -s $d/wsh/fs_004.npz ] && echo $n; done | tr '\n' ',')
log "판 끝; 표 판 $NS"
$PY exe/wmwhy_prof.py $O $JP $AG/wm.pt $GH5 "$NS" > $O/wmwhy_table_$STG.md 2> $O/wmwhy_prof_$STG.err
log "표 rc=$? $(wc -l < $O/wmwhy_table_$STG.md) 줄 $(tail -2 $O/wmwhy_prof_$STG.err | tr '\n' ' ' | cut -c1-300)"
rclone copy $O $RO -q
log "끝"
