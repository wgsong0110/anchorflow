#!/bin/bash
# 절 CM33 묶음 wmen (수박 (2) 판 = 절 CM32 fi + CM27 (d) 쪼개기 끔 의 에너지 생성: 몫 분해 진단 --enrec + 고침 후보). 판 = 원자를 + 로 이음.
#   원자: p2 ((2) 판 = 기준 장벽 바닥 + 보간 교체 K 1 + --gq_oct 0 --ws_bar 4 --ws_fi --y0loc 3 --ue be --uf_res, 쪼개기 끔) · dn ((2) 에서 --y0loc 뺀 판)
#   · ro (--uf_ro 판정 잔차만) · nl (--ue_nl 자른 노드만 실제 이동 기반 갱신) · x6 (--newton 6 --xit_from 2: 덜 풀린 걸음만 반복 4 번 더) · c25·c3 (셀 2.5·3).
#   판마다 비결정 두 번: a = 진단 (결과 불변) --qjd --jdiag --bar_cnt --erec --ufrec --nrec --enrec 100 (+ --frec 30 p2·dn 단독 판), b = --erec 만 -- 시간은 b (한 줄, 카드 하나에 차례로).
# 사용 (잡 안, 커밋 디렉토리에서): bash exe/stab_wmen.sh <출력> <R2> <단계 이름> <판 ...>
O=$1; R=$2; STG=$3; shift 3; VARS="$*"; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
atom() { case $1 in
    p2) echo "--gq_oct 0 --ws_bar 4 --ws_fi --y0loc 3 --ue be --uf_res";; dn) echo "--gq_oct 0 --ws_bar 4 --ws_fi --ue be --uf_res";;
    ro) echo "--uf_ro";; nl) echo "--ue_nl";; x6) echo "--xit_from 2";; c25|c3) echo "";;
    *) echo "??$1";; esac; }
cmd_of() {   # 판 이름 -> 명령 (진단 없이; 모두 쪼개기 끔)
  local c="$BASE" x
  case "+$1+" in *+c25+*) c="${c/--cell 1.5/--cell 2.5}";; *+c3+*) c="${c/--cell 1.5/--cell 3}";; esac
  case "+$1+" in *+x6+*) c="${c/--newton 2/--newton 6}";; esac
  for x in ${1//+/ }; do c="$c $(atom $x)"; done; echo "$c"; }
dg_of() { local n=$1 v=${1%_*} x
  case $n in *_b) echo "--erec"; return;; esac
  x="--qjd --jdiag --bar_cnt --erec --ufrec --nrec --enrec 100"
  case "+$v+" in *+ro+*|*+nl+*|*+x6+*) ;; *) x="$x --frec 30";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmen.log; rclone copyto $O/wmen.log $R/wmen.log -q; }
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
for v in $VARS; do case "$(cmd_of $v)" in *"??"*) echo "판 이름 틀림 $v" > $O/wmen.FAIL; exit 1;; esac; done
# ---- 0. 끄면 같은 소스 (앞 커밋 7959f1d 와 생성 소스 sha1 대조, 플래그 끔·절 CM30 판 조합) + 켠 소스 nvrtc 컴파일 확인은 연기에서
if [ ! -f $O/srcchk.ok ]; then
  rm -rf $O/srcold; mkdir -p $O/srcold; git archive 7959f1d lib | tar -x -C $O/srcold
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
  [ -f $O/srcchk.ok ] || { echo "소스 대조 실패 $(date '+%F %T')" > $O/wmen.FAIL; exit 1; }
fi
# ---- 1. 연기: 셀 3 의 (2) 판·모두 켠 판 6 프레임 (진단 포함, f4 첫 접촉 지남), enb 몫 합 대조
SM=$O/smoke_$STG; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  SV="c3+p2 c3+p2+ro+nl+x6 c3+p2+x6"
  for v in $SV; do ( g=$(claim_wait $BASHPID); c="$(cmd_of $v | sed 's/--frames [0-9]*/--frames 6/') $(dg_of ${v}_a)"
    mkdir -p $SM/$v; echo "$PY -u $c --out $SM/$v" > $SM/$v/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$v < /dev/null > $SM/$v/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$v/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for v in $SV; do
    x=$($PY -c "
import json; s=json.load(open('$SM/$v/summary.json')); r=[json.loads(l) for l in open('$SM/$v/fem.jsonl')]
e=[q.get('enb') or {} for q in r]; ok=s['frames_done'] >= 5 and all('sum_chk' in q for q in e)
print(int(ok), s['frames_done'], s.get('diverged'), 'chk', max(abs(q.get('sum_chk', 9e9)) for q in e), 'step_chk', max(abs(q.get('step', {}).get('chk', 9e9)) for q in e))" 2>/dev/null || echo "0 - -")
    set -- $x; [ "$1" = 1 ] || OK=0; MSG="$MSG | $v:$x $(cat $SM/$v/rc.txt) $(grep -a 'rror\|Trace' $SM/$v/log.txt | tail -3 | cut -c1-260 | tr '\n' ' ')"; done
  log "[연기] OK=$OK$MSG"; rclone copy $SM $R/smoke_$STG -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wmen.FAIL; exit 1; }
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
  NL=$NG; [ $ph = b ] && NL=1
  for l in $(seq 1 $NL); do lane $Q & sleep 2; done
  wait
  log "단계 $ph 끝"
done
[ -s $O/gt_stats.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmfrac/gt_stats.json $O/gt_stats.json -q
$PY exe/wmen_tab.py $O > $O/wmen_table.md 2> $O/wmen_table.err; rclone copy $O $R -q --include "wmen_table.*"
log "판 끝 $STG"
