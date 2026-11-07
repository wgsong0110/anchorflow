#!/bin/bash
# 절 CM24 묶음 wmk1fix 1 단계 (수박): 기본 판 K 1 (장벽 바닥 + 보간 교체, 뉴턴 2 x PCG 5, HHT ρ 0.7) 첫 접촉 걸음 덜 풀림 고치기 (사용자 승인 11-07 정의 변경).
#   (1a) 보폭 막는 원인 끄기: --y0eb (시작점 장벽 에너지 규칙 -> 실현 가능 k0 만) / --y0u (시작점 뒤집힘 필터) / --ls_end / --ls_inv / 둘 다 -- 반복별 뒤집힘·장벽 보폭 기록 (--nrec 확장)
#   (1b) --ksplit 3: 덜 풀린 걸음이 있으면 프레임을 버리고 K 두 배 (최대 8)
#   (2)  --ue be (덜 풀린 걸음 후퇴 오일러 갱신), 비교 --ue beall (모든 걸음 후퇴 오일러 갱신) · --integ be (HHT 없는 후퇴 오일러 적분)
#   (3)  --ue a0 (덜 풀린 걸음 가속도 기억 0)
#   판정 판 = 판마다 비결정 두 번 (a·b), 진단 (결과 불변) --nrec --qjd --jdiag --bar_cnt --erec --ufrec (+ K 1 HHT 판 --frec 15) + --dump (pred.pt, 덤프는 R2 에 올리고 확인 뒤 노드에서만 지움).
#   단독 시간 판 z_<판> = 판 플래그만 (진단·덤프 없음), 카드 하나에 차례로. 연기 = 판마다 5 프레임 (진단 포함) 먼저.
#   단계 b (판정 2, --uf_res: 덜 풀린 걸음 = 허용치 미달 & (보폭 < 1 또는 마지막 뉴턴 반복 시작 ‖g‖ > F)): ks3r · ks5r (최대 K 32) · uber · ua0r.
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2, 단계 $3 (빈 값 = 1 단계, b).
O=$1; R=$2; STG=${3:-}; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
# 절 CM23 기본 판 K 1 수박 명령 (k1new M_K1 그대로) -- 판마다 여기서 플래그를 빼거나 더함
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
DG="--nrec --qjd --jdiag --bar_cnt --erec --ufrec"
cmd_of() {   # cmd_of 판 -> 판 명령 (진단 없이)
  case $1 in
    base) echo "$BASE";;
    noeb) echo "$BASE" | sed 's/ --y0eb//';;
    noy0u) echo "$BASE" | sed 's/ --y0u//';;
    noend) echo "$BASE" | sed 's/ --ls_end//';;
    noinv) echo "$BASE" | sed 's/ --ls_inv//';;
    noie) echo "$BASE" | sed 's/ --ls_inv//; s/ --ls_end//';;
    ks3) echo "$BASE --ksplit 3";;
    ube) echo "$BASE --ue be";;
    ubeall) echo "$BASE --ue beall";;
    be) echo "$BASE" | sed 's/--integ mid --ga hht --rho 0.7/--integ be/';;
    ua0) echo "$BASE --ue a0";;
    ks3r) echo "$BASE --ksplit 3 --uf_res";;
    ks5r) echo "$BASE --ksplit 5 --uf_res";;
    uber) echo "$BASE --ue be --uf_res";;
    ua0r) echo "$BASE --ue a0 --uf_res";;
  esac; }
dg_of() { case $1 in ks3|ks3r|ks5r|be) echo "$DG";; *) echo "$DG --frec 15";; esac; }
VARS="ks3 base noeb noy0u noend noinv noie ube ubeall be ua0"
[ "$STG" = b ] && VARS="ks5r ks3r uber ua0r"
log() { echo "[$(date '+%F %T')] $*" >> $O/wmk1fix.log; rclone copyto $O/wmk1fix.log $R/wmk1fix.log -q; }
up() { rclone copy $O/$1 $R/$1 -q --transfers 8; }
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
# ---- 0. 끄면 같은 소스: 이 커밋과 앞 커밋 (e598bf8) 의 생성 CUDA 소스를 플래그 끔 조합마다 sha1 대조 (켬 조합은 길이만)
if [ ! -f $O/srcchk.ok ]; then
  rm -rf $O/srcold; mkdir -p $O/srcold; git archive e598bf8 lib | tar -x -C $O/srcold
  $PY - $O/srcold/lib $(pwd -P)/lib > $O/srcchk.txt 2>&1 <<'PYX'
import hashlib, importlib, sys
res = {}
for tag, p in (("old", sys.argv[1]), ("new", sys.argv[2])):
    for m in [k for k in sys.modules if k.startswith("anchorflow")]:
        del sys.modules[m]
    sys.path.insert(0, p)
    A = importlib.import_module("anchorflow.fem_ao_rt")
    assert A.__file__.startswith(p), A.__file__
    bar = dict(NR=[0.0, 0.0, 1.0], PN=0.02, DH=1 / 300, KS=1.0, y0eb=True)
    for tol in (False, True):
        for fq in (False, True):
            for gq in (False, True):
                for b in (None, bar):
                    for lj in (False, True):
                        res.setdefault((tol, fq, gq, b is not None, lj), {})[tag] = hashlib.sha1(A._src(tol=tol, fqrl=fq, gqm=gq, elx="snh", fpi=True, bar=b, lji=lj).encode()).hexdigest()
                        if tag == "new":
                            res[(tol, fq, gq, b is not None, lj)]["on_len"] = len(A._src(tol=tol, fqrl=fq, gqm=gq, elx="snh", fpi=True, bar=b, lji=lj, uf=True))
    sys.path.pop(0)
bad = [k for k, v in res.items() if v["old"] != v["new"]]
for k, v in res.items():
    print(k, v["old"][:12], v["new"][:12], v["on_len"])
print("끔 조합", len(res), "다름", len(bad), "OK" if not bad else "FAIL")
PYX
  tail -1 $O/srcchk.txt | grep -q " OK$" && touch $O/srcchk.ok
  log "[소스 대조] $(tail -1 $O/srcchk.txt)"; rclone copy $O $R -q --include "srcchk.txt"
  [ -f $O/srcchk.ok ] || { echo "소스 대조 실패 $(date '+%F %T')" > $O/wmk1fix.FAIL; exit 1; }
fi
# ---- 1. 연기: 판마다 5 프레임 (진단 포함, 덤프 없음)
SM=$O/smoke$STG; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  for v in $VARS; do ( g=$(claim_wait $BASHPID); c="$(cmd_of $v | sed 's/--frames 100/--frames 5/') $(dg_of $v)"
    mkdir -p $SM/$v; echo "$PY -u $c --out $SM/$v" > $SM/$v/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$v < /dev/null > $SM/$v/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$v/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for v in $VARS; do
    x=$($PY -c "import json; s=json.load(open('$SM/$v/summary.json')); r=[json.loads(l) for l in open('$SM/$v/fem.jsonl')]; print(int(s['frames_done'] >= 4 and all('uf' in q and 'en' in q for q in r)), s['frames_done'], s.get('diverged'))" 2>/dev/null || echo "0 - -")
    set -- $x; [ "$1" = 1 ] || OK=0; MSG="$MSG | $v:$x $(cat $SM/$v/rc.txt) $(grep -a 'rror\|Trace' $SM/$v/log.txt | tail -2 | cut -c1-240 | tr '\n' ' ')"; done
  log "[연기$STG] OK=$OK$MSG"; rclone copy $SM $R/smoke$STG -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wmk1fix.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 2. 판정 판: 공용 대기열, 줄마다 카드 하나
Q=$O/queue$STG.txt; [ -f $Q ] || for v in $VARS; do echo "${v}_a $v"; echo "${v}_b $v"; done > $Q
echo 0 > $O/qpos$STG
run() {   # run 이름 판 카드
  local n=$1 v=$2 g=$3 c
  [ -f $O/$n/summary.json ] && return
  c="$(cmd_of $v) $(dg_of $v) --dump"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  local d=$(rclone check $O/$n $R/$n --one-way --size-only 2>&1 | grep -c ERROR)
  if [ "$d" = 0 ] && [ -f $O/$n/pred.pt ]; then echo "pred.pt $(stat -c %s $O/$n/pred.pt) R2 확인 차이 0 -> 노드 사본 지움 $(date '+%F %T')" > $O/$n/pred_on_r2.txt; rm -f $O/$n/pred.pt; up $n; fi
  log "$n R2 확인 차이 $d"; }
lane() { local me=$BASHPID i ln g
  while :; do
    i=$(flock $O/qpos$STG.lock bash -c "i=\$(cat $O/qpos$STG); echo \$((i + 1)) > $O/qpos$STG; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/$1/summary.json ] && continue
    g=$(claim_wait $me); run $1 $2 $g; rm -f $CL/$g
  done; }
for l in $(seq 1 $NG); do lane & sleep 2; done
wait
log "판정 판 끝 $STG"
# ---- 3. 단독 시간: 카드 하나 잡고 판마다 차례로 (진단·덤프 없음), 10 초마다 그 카드 계산 프로세스 수
g=$(claim_wait $$); log "단독 시간 카드 $g"
for v in $VARS; do n=z_$v
  [ -f $O/$n/summary.json ] && continue
  c="$(cmd_of $v)"; rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1 &
  P=$!; while kill -0 $P 2>/dev/null; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" >> $O/$n/gpu_procs.txt; sleep 10; done; wait $P
  log "$n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"; up $n
done
rm -f $CL/$g
$PY exe/wmk1fix_tab.py $O $WMD/wm.pt > $O/wmk1fix_table.md 2> $O/wmk1fix_table.err; rclone copy $O $R -q --include "wmk1fix_table.*" --include "gt_energy.json"
log "단계 끝 ${STG:-1}"
