#!/bin/bash
# 절 CM50 묶음 speed4: 수박 F 60 fps -- 단계별 시간 기록 (결과 불변) + 결과 불변 가속 (S1·X1) + 결과 바뀌는 후보. 잡 안, 커밋 디렉토리에서.
# exe/stab_escape4.sh (절 CM49) 틀 그대로 + 맨 앞 감싸개 exe/speed4_wrap.py ('@s4...' 인자) -> escape4_wrap.py ('@e5...') -> escape3 ('@e3...') -> escape2 ('@e2...') -> escape1/speed3 (나머지 '@').
# 낱말 F = 공통 후보 F (E + W̄ @e3w 1 + R2 @e5r 2 + A2 @e5a 2), E = D + 예산 T2 (+ 수박 @inv 1), D = 절 CM37 C + 절 CM38 X3c (+ 수박 --sp2 31).
# 사용: bash exe/stab_speed4.sh <출력> <R2> <단계> (판 목록 exe/speed4_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식: z 단독 시간 (a·b) · p 단독 단계 기록 (a 만) · q 단독 커널 프로파일 (speed1_prof --prof 1, a 만) · n 비결정 (동시, --erec) · h 결정적 해시 (새 실행기, 동시)
#   단독 판 (z·p·q): 계산 프로세스 0 인 카드 하나를 잡아 차례로 (판마다 시작·끝 nvidia-smi·부하를 판 디렉토리에 남김; 시작 때 그 카드 계산 프로세스 0 확인)
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
for t in L000 ficus bread wolf; do cm_data $t; done
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
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
dfl() { local s=$1 x=" $2 "
  x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  x="${x// FH / D @tb 1000 @tb_m 0.8 @tb_w 2 @e3w 1 @e5r 2 @e5a 2 }"                      # 해시 대조용: 예산이 안 걸리는 F (절 CM39.2 와 같은 꼴)
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in M*) case $md in q) echo "--erec --prof 1";; *) echo "--erec";; esac; return;; esac
  case $md in h) x="--det --fhash";; *) x="";; esac
  [ $s = W ] && [ $md = n ] && x="$x --dump --dump_part"
  [ $s = F ] && [ $md = n ] && x="$x --dump"
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/speed4_$STG.log; rclone copyto $O/speed4_$STG.log $RO/speed4_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt; }
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
run() {   # run 이름 씬 방식 카드 플래그
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre="" wo="" w2="" w3="" w4="" w5=""
  [ -f $O/$n/summary.json ] && return
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+-]*' | grep -v '^@\(e[235]\|s4\)' | sed 's/^@/--/' | tr '\n' ' ')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  c="exe/escape4_wrap.py $w5 $w3 $w2 $wo $c"
  [ -n "$w4" ] && c="exe/speed4_wrap.py $w4 $c"
  [ $md = q ] && c="exe/speed1_prof.py $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; echo "카드 $g 계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_start.txt; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  log "$n 카드 $g 시작"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_end.txt; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
[ -f exe/speed4_pre_$STG.sh ] && { log "앞 단계 스크립트 speed4_pre_$STG.sh"; bash exe/speed4_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
SPEC=$(grep -v '^#' exe/speed4_spec.txt | awk -v s=$STG '$1==s')
# ---- 단독 (z·p·q) 먼저: 계산 프로세스 0 인 카드 하나에 차례로 (a 다 돌고 b)
if echo "$SPEC" | awk '$4=="z" || $4=="p" || $4=="q"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG (계산 프로세스 0 인 카드 하나; 노드 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  for r in a b c; do
    while read -r st n s md x; do
      case $md in z) ;; p|q) [ $r != a ] && continue;; *) continue;; esac
      [ -f $O/${n}_$r/summary.json ] && continue
      while [ "$(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; do log "카드 $ZG 에 다른 계산 프로세스 -- 기다림"; sleep 30; done
      log "${n}_$r 단독 카드 $ZG 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
# ---- 동시 판 (n·h): 카드마다 하나
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z|p|q) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판 끝 $STG"
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나야 (컴파일·훅 확인), 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    case $md in z) nn=${n}_a;; p|q) nn=${n}_a;; *) nn=$n;; esac
    $PY -c "import json,sys; s=json.load(open('$O/$nn/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $nn 5 프레임 못 감/발산"; }
    case "$s $x " in M*" @s4"*) [ -s $O/$nn/s4.jsonl ] || { OK=0; log "[연기] $nn s4.jsonl 없음"; }; grep -q "절 CM50\] speed4" $O/$nn/log.txt || { OK=0; log "[연기] $nn CM50 줄 없음"; };; esac
    case " $x " in *" @s4p 1"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$nn/s4.jsonl')]; t=[q['sp']['tries'] for q in L if q['sp'].get('n')]; sys.exit(0 if len(t) >= 3 and all(len(q) >= 1 for q in t) else 1)" || { OK=0; log "[연기] $nn 단계 기록 비었음"; };; esac
    case " $x " in *" @s4x 3"*) $PY -c "
import json,sys
L=[json.loads(l) for l in open('$O/$nn/s4.jsonl')]
R=[r for q in L for r in q.get('x2c',[]) if r['kind']=='같은 시도']
bad=[]
for r in R:
    for k,v in r.items():
        if k in ('kind','coll','IP','bx','bx_snap'): continue
        if (isinstance(v,list) and (v[0]!=0 or v[3]!=0)) or (isinstance(v,int) and v!=0): bad.append((k,v))
    if r.get('bx_snap') is None or r['bx_snap'][0]!=0: bad.append(('bx_snap',r.get('bx_snap')))
print('x2c 같은 시도', len(R), '어긋남', bad[:5])
sys.exit(0 if R and not bad else 1)" >> $O/speed4_$STG.log 2>&1 || { OK=0; log "[연기] $nn X2 확인 실패"; };; esac
    case " $x " in *" @s4y 2"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$nn/s4.jsonl')]; b=sum(q['yl4']['bad'] for q in L); r=sum(q['yl4']['chk'] for q in L); print('yl4 bad', b, 'chk', r); sys.exit(0 if b == 0 and r > 0 else 1)" >> $O/speed4_$STG.log 2>&1 || { OK=0; log "[연기] $nn S2 확인 실패"; };; esac
    case " $x " in *" @s4w 2"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$nn/s4.jsonl')]; b=sum(q['ws4']['bad'] for q in L); r=sum(q['ws4']['round'] for q in L); print('ws4 bad', b, 'round', r); sys.exit(0 if b == 0 and r > 0 else 1)" >> $O/speed4_$STG.log 2>&1 || { OK=0; log "[연기] $nn S1 확인 실패"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/speed4.FAIL; rclone copy $O $RO -q --include speed4.FAIL; };;
esac
$PY exe/speed4_tab.py $O > $O/speed4_table.md 2> $O/speed4_table.err; rclone copy $O $RO -q --include "speed4_table.*"
log "단계 끝 $STG"
