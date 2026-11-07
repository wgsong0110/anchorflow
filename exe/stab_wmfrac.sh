#!/bin/bash
# 절 CM28 묶음 wmfrac (수박 깨짐): 기준 판 = 수박 기본 판 (장벽 바닥 + 보간 교체, K 1) + --ksplit 3. 원본 GF NACC 와 이식 차이를 하나씩 없애는 판 + 첫 접촉 분석.
#   판 이름 = 원자들을 + 로 이음 (조합). 원자: base (기준 판) · seed ((a) 씨 영역 β 3e9) · ljp0 / ljmax ((b) 가우스 점 logJp 모으기) · sub16 / sub64 ((c) 걸음 안 RM n 등분)
#   · rmsnh ((d) RM 식 SNH 와 맞춤) · ksr1 ((e) 쪼갠 시도 HHT ρ∞ 1) · ksit ((e) 쪼갠 시도 뉴턴 8 x PCG 20) -- 모두 기준 판 위에.
#   분석 판 (12 프레임, 1 회): ref (K 64 고정, 뉴턴 8 x PCG 20 -- 같은 공간 이산화를 작은 걸음으로 잘 푼 기준) · ref_r1 (ref + ρ∞ 1) · ref_norm (ref + RM 끔) · ks3_norm (기준 판 + RM 끔)
#   판정 판 = 판마다 비결정 두 번 (a·b), 진단 (결과 불변) --nrec --qjd --jdiag --bar_cnt --erec --ufrec (en.fs = 프레임 1~12 J·에너지·logJp 분포).
# 사용 (잡 안, 커밋 디렉토리에서): bash exe/stab_wmfrac.sh <출력> <R2> <단계 이름> <판 ...>   (판에 ana 를 주면 분석 판 넷 + 정답 통계)
O=$1; R=$2; STG=$3; shift 3; VARS="$*"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
SEED=$O/wm_seed.npz; [ -s $SEED ] || rclone copyto $R/wm_seed.npz $SEED -q
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
DG="--nrec --qjd --jdiag --bar_cnt --erec --ufrec"
atom() { case $1 in
    base) echo "";; seed) echo "--seedb $SEED";; ljp0) echo "--ljq p0";; ljmax) echo "--ljq max";;
    sub16) echo "--rm_sub 16";; sub64) echo "--rm_sub 64";; rmsnh) echo "--rm_snh";; ksr1) echo "--ks_rho 1.0";; ksit) echo "--ks_it 8,20";;
    *) echo "??$1";; esac; }
cmd_of() {   # 판 이름 -> 명령 (진단 없이)
  case $1 in
    ref) echo "$BASE" | sed 's/--frames 100/--frames 12/; s/--K 1$/--K 64/; s/--newton 2 --cg 5/--newton 8 --cg 20/';;
    ref_r1) cmd_of ref | sed 's/--rho 0.7/--rho 1.0/';;
    ref_norm) echo "$(cmd_of ref) --no_rm";;
    ks3_norm) echo "$BASE --ksplit 3 --no_rm" | sed 's/--frames 100/--frames 12/';;
    *) local c="$BASE --ksplit 3" x; for x in ${1//+/ }; do c="$c $(atom $x)"; done; echo "$c";;
  esac; }
dg_of() { case $1 in *ksit*) echo "$DG" | sed 's/--nrec //';; *) echo "$DG";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmfrac.log; rclone copyto $O/wmfrac.log $R/wmfrac.log -q; }
up() { rclone copy $O/$1 $R/$1 -q; }
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
for v in $VARS; do [ "$v" = ana ] && continue; case "$(cmd_of $v)" in *"??"*) echo "판 이름 틀림 $v" > $O/wmfrac.FAIL; exit 1;; esac; done
# ---- 0. 끄면 같은 소스 (앞 커밋 97eee1e 와 생성 CUDA 소스 sha1 대조, 플래그 끔 64 조합)
if [ ! -f $O/srcchk.ok ]; then
  rm -rf $O/srcold; mkdir -p $O/srcold; git archive 97eee1e lib | tar -x -C $O/srcold
  $PY - $O/srcold/lib $(pwd -P)/lib > $O/srcchk.txt 2>&1 <<'PYX'
import hashlib, importlib, itertools, sys
res = {}
BAR = dict(NR=[0.0, 0.0, 1.0], PN=0.02, DH=1 / 300, KS=1.0, y0eb=True)
for tag, p in (("old", sys.argv[1]), ("new", sys.argv[2])):
    for m in [k for k in sys.modules if k.startswith("anchorflow")]:
        del sys.modules[m]
    sys.path.insert(0, p)
    A = importlib.import_module("anchorflow.fem_ao_rt")
    assert A.__file__.startswith(p), A.__file__
    for tol, fq, gq, b, lj, uf in itertools.product((False, True), repeat=6):
        res.setdefault((tol, fq, gq, b, lj, uf), {})[tag] = hashlib.sha1(A._src(tol=tol, fqrl=fq, gqm=gq, elx="snh", fpi=True, bar=BAR if b else None, lji=lj, uf=uf).encode()).hexdigest()
    sys.path.pop(0)
bad = [k for k, v in res.items() if v["old"] != v["new"]]
print("끔 조합", len(res), "다름", len(bad), "OK" if not bad else "FAIL")
PYX
  tail -1 $O/srcchk.txt | grep -q " OK$" && touch $O/srcchk.ok
  log "[소스 대조] $(tail -1 $O/srcchk.txt)"; rclone copy $O $R -q --include "srcchk.txt"
  [ -f $O/srcchk.ok ] || { echo "소스 대조 실패 $(date '+%F %T')" > $O/wmfrac.FAIL; exit 1; }
fi
# ---- 1. 줄 만들기 (분석 판은 1 회, 판정 판은 a·b)
Q=$O/queue_$STG.txt
if [ ! -f $Q ]; then
  for v in $VARS; do
    if [ "$v" = ana ]; then for w in ref ref_r1 ref_norm ks3_norm; do echo "$w $w"; done; else echo "${v}_a $v"; echo "${v}_b $v"; fi
  done > $Q
fi
# ---- 2. 연기: 판마다 5 프레임 (진단 포함) -- 원자 하나씩 (조합은 원자 연기가 통과했으면 생략)
SM=$O/smoke_$STG; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  SV=$(awk '{print $2}' $Q | sort -u | grep -v "^ref" | grep -v "+" | tr '\n' ' ')
  for v in $SV; do ( g=$(claim_wait $BASHPID); c="$(cmd_of $v | sed 's/--frames [0-9]*/--frames 5/') $(dg_of $v)"
    mkdir -p $SM/$v; echo "$PY -u $c --out $SM/$v" > $SM/$v/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$v < /dev/null > $SM/$v/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$v/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for v in $SV; do
    x=$($PY -c "import json; s=json.load(open('$SM/$v/summary.json')); r=[json.loads(l) for l in open('$SM/$v/fem.jsonl')]; print(int(s['frames_done'] >= 4 and all('fs' in q.get('en', {}) for q in r)), s['frames_done'], s.get('diverged'))" 2>/dev/null || echo "0 - -")
    set -- $x; [ "$1" = 1 ] || OK=0; MSG="$MSG | $v:$x $(cat $SM/$v/rc.txt) $(grep -a 'rror\|Trace\|절 CM28' $SM/$v/log.txt | tail -3 | cut -c1-260 | tr '\n' ' ')"; done
  log "[연기 $STG] OK=$OK$MSG"; rclone copy $SM $R/smoke_$STG -q
  [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/wmfrac.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 3. 정답 통계 (분석 단계, CPU -- 카드를 쥐지 않음)
if echo " $VARS " | grep -q " ana " && [ ! -s $O/gt_stats.json ]; then
  GH=/tmp/dkta/wmfrac_gt_h5; mkdir -p $GH
  for f in $(seq 0 12); do n=$(printf "sim_%010d.h5" $f); [ -s $GH/$n ] || rclone copyto r2:storage/result/anchorflow/_vastmirror/wm2/gf/simulation_ply/$n $GH/$n -q; done
  GJ=/tmp/dkta/wmfrac_gt_jp; mkdir -p $GJ
  for f in $(seq 0 12); do n=$(printf "sc_%04d.npy" $f); [ -s $GJ/$n ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp/$n $GJ/$n -q; done
  ( CUDA_VISIBLE_DEVICES="" $PY -u exe/wmfrac_gt.py $WMD/wm.pt $GH $GJ $O/gt_stats.json 12 > $O/gt_stats.log 2>&1; log "정답 통계 rc=$? $(tail -1 $O/gt_stats.log | cut -c1-200)"; rclone copy $O $R -q --include "gt_stats.*" ) &
fi
# ---- 4. 판: 공용 대기열, 줄마다 카드 하나, 끝나면 바로 R2
echo 0 > $O/qpos_$STG
run() {   # run 이름 판 카드
  local n=$1 v=$2 g=$3 c
  [ -f $O/$n/summary.json ] && return
  c="$(cmd_of $v) $(dg_of $v)"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"; }
lane() { local me=$BASHPID i ln g
  while :; do
    i=$(flock $O/qpos_$STG.lock bash -c "i=\$(cat $O/qpos_$STG); echo \$((i + 1)) > $O/qpos_$STG; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/$1/summary.json ] && continue
    g=$(claim_wait $me); run $1 $2 $g; rm -f $CL/$g
    up $1 &
  done; wait; }
for l in $(seq 1 $NG); do lane & sleep 2; done
wait
log "판 끝 $STG"
$PY exe/wmfrac_tab.py $O > $O/wmfrac_table.md 2> $O/wmfrac_table.err; rclone copy $O $R -q --include "wmfrac_table.*"
log "단계 끝 $STG"
