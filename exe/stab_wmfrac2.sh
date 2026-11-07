#!/bin/bash
# 절 CM32 묶음 wmfrac2 (수박 첫 접촉: 절 CM30 (a)+(b) 위 뒤집힘 필터 노드별 국소화 + CM27 (d) + 쪼갠 걸음 뉴턴 8 x PCG 20). 판 = 원자를 + 로 이음.
#   원자: base (기준 판 = 수박 기본 판 장벽 바닥 + 보간 교체 K 1 + --ksplit 3) · oct (--gq_oct 0) · bl (--ws_bar 4) · fi (--ws_fi: 필터도 노드별, 절 CM32)
#   · d (절 CM27 (d) --y0loc 3 --ue be --uf_res) · kit (--ks_it 8,20: 쪼갠 시도만 뉴턴 8 x PCG 20) · nks (--ksplit 끔) · c25·c3 (셀 2.5·3).
#   판마다 비결정 두 번: a = 진단 (결과 불변) --qjd --jdiag --bar_cnt --erec --ufrec (+ --w2rec bl 판, + --nrec kit 없는 판), b = --erec 만 -- 시간은 b.
#   단계 a: a 판 전부 (카드 하나에 하나, 동시에), 단계 b: b 판 (카드 하나에 하나, 동시 4 개까지 -- CPU 나눔 줄이기).
# 사용 (잡 안, 커밋 디렉토리에서): bash exe/stab_wmfrac2.sh <출력> <R2> <단계 이름> <판 ...>
O=$1; R=$2; STG=$3; shift 3; VARS="$*"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
atom() { case $1 in
    base|nks|c25|c3) echo "";; oct) echo "--gq_oct 0";; bl) echo "--ws_bar 4";; fi) echo "--ws_fi";; d) echo "--y0loc 3 --ue be --uf_res";; kit) echo "--ks_it 8,20";;
    *) echo "??$1";; esac; }
cmd_of() {   # 판 이름 -> 명령 (진단 없이)
  local c="$BASE" x
  case "+$1+" in *+nks+*) ;; *) c="$c --ksplit 3";; esac
  case "+$1+" in *+c25+*) c="${c/--cell 1.5/--cell 2.5}";; *+c3+*) c="${c/--cell 1.5/--cell 3}";; esac
  for x in ${1//+/ }; do c="$c $(atom $x)"; done; echo "$c"; }
dg_of() { local n=$1 v=${1%_*} x
  case $n in *_b) echo "--erec"; return;; esac
  x="--qjd --jdiag --bar_cnt --erec --ufrec"
  case "+$v+" in *+bl+*) x="$x --w2rec";; esac
  case "+$v+" in *+kit+*) ;; *) x="$x --nrec";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmfrac2.log; rclone copyto $O/wmfrac2.log $R/wmfrac2.log -q; }
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
for v in $VARS; do case "$(cmd_of $v)" in *"??"*) echo "판 이름 틀림 $v" > $O/wmfrac2.FAIL; exit 1;; esac; done
# ---- 0. 끄면 같은 소스 (새 CUDA 커널 없음 -- 앞 커밋과 생성 소스 sha1 대조, 플래그 끔·절 CM30 판 조합)
if [ ! -f $O/srcchk.ok ]; then
  rm -rf $O/srcold; mkdir -p $O/srcold; git archive b499d89 lib | tar -x -C $O/srcold
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
    for tol, fq, gq, b, lj, uf, ws in itertools.product((False, True), repeat=7):
        wf = dict(ws=dict(oct=0.0, red=False, bar=4)) if (ws and b and gq) else {}
        res.setdefault((tol, fq, gq, b, lj, uf, ws), {})[tag] = hashlib.sha1(A._src(tol=tol, fqrl=fq, gqm=gq, elx="snh", fpi=True, bar=BAR if b else None, lji=lj, uf=uf, wf=wf).encode()).hexdigest()
    sys.path.pop(0)
bad = [k for k, v in res.items() if v["old"] != v["new"]]
print("조합", len(res), "다름", len(bad), "OK" if not bad else "FAIL")
PYX
  tail -1 $O/srcchk.txt | grep -q " OK$" && touch $O/srcchk.ok
  log "[소스 대조] $(tail -1 $O/srcchk.txt)"; rclone copy $O $R -q --include "srcchk.txt"
  [ -f $O/srcchk.ok ] || { echo "소스 대조 실패 $(date '+%F %T')" > $O/wmfrac2.FAIL; exit 1; }
fi
# ---- 1. 연기: 셀 3 의 새 판 5 프레임 (진단 포함, f4 첫 접촉 쪼갬 지남)
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  SV="c3+oct+bl+fi c3+oct+bl+fi+nks c3+oct+bl+fi+d c3+oct+bl+fi+d+kit"
  for v in $SV; do ( g=$(claim_wait $BASHPID); c="$(cmd_of $v | sed 's/--frames [0-9]*/--frames 5/') $(dg_of ${v}_a)"
    mkdir -p $SM/$v; echo "$PY -u $c --out $SM/$v" > $SM/$v/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$v < /dev/null > $SM/$v/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$v/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for v in $SV; do
    x=$($PY -c "import json; s=json.load(open('$SM/$v/summary.json')); r=[json.loads(l) for l in open('$SM/$v/fem.jsonl')]; print(int(s['frames_done'] >= 4 and all('fs' in q.get('en', {}) for q in r) and all('w2' in q for q in r)), s['frames_done'], s.get('diverged'))" 2>/dev/null || echo "0 - -")
    set -- $x; [ "$1" = 1 ] || OK=0; MSG="$MSG | $v:$x $(cat $SM/$v/rc.txt) $(grep -a 'rror\|Trace' $SM/$v/log.txt | tail -3 | cut -c1-260 | tr '\n' ' ')"; done
  log "[연기] OK=$OK$MSG"; rclone copy $SM $R/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wmfrac2.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 2. 판: 단계 a (진단, 모든 카드) -> 단계 b (시간, 4 줄)
run() {   # run 이름 판 카드
  local n=$1 v=$2 g=$3 c
  [ -f $O/$n/summary.json ] && return
  c="$(cmd_of $v) $(dg_of $n)"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"; }
lane() { local Q=$1 me=$BASHPID i ln g
  while :; do
    i=$(flock $Q.lock bash -c "i=\$(cat $Q.pos); echo \$((i + 1)) > $Q.pos; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/$1/summary.json ] && continue
    g=$(claim_wait $me); run $1 $2 $g; rm -f $CL/$g
    up $1 &
  done; wait; }
for ph in a b; do
  Q=$O/queue_${STG}_$ph.txt
  [ -f $Q ] || for v in $VARS; do echo "${v}_$ph $v"; done > $Q
  echo 0 > $Q.pos
  NL=$NG; [ $ph = b ] && NL=4
  for l in $(seq 1 $NL); do lane $Q & sleep 2; done
  wait
  log "단계 $ph 끝"
done
rclone copyto r2:storage/result/anchorflow/lbours/wmfrac/gt_stats.json $O/gt_stats.json -q
$PY exe/wmfrac2_tab.py $O > $O/wmfrac2_table.md 2> $O/wmfrac2_table.err; rclone copy $O $R -q --include "wmfrac2_table.*"
log "판 끝 $STG"
