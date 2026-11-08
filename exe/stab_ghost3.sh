#!/bin/bash
# 절 CM51 묶음 ghost3: ghost penalty 3 (고스트 질량 + 얇은 구조 면 빼기 + 칸별 커널; 감싸개 exe/ghost3_wrap.py '@g3 g', '@g3m 0|1', '@g3t 0|1', '@g3fd 프레임,..') 를 공통 후보 F 위에 -- 수박 E·V2NF, 다른 씬 회귀, 결정적 해시. 잡 안, 커밋 디렉토리에서.
# exe/stab_escape4.sh (절 CM49) 틀 그대로 + 맨 앞 감싸개 exe/ghost3_wrap.py ('@g3...' 인자) -> escape4_wrap ('@e5...') -> escape3 ('@e3...') -> escape2 ('@e2...') -> escape1/speed3 (나머지 '@'); 진단 판에 AF_G1_DG=1. 방식 d = 결정적 판정 (--det --fhash).
#   + 낱말 V2NF·V2N (절 CM44·CM46 글자 그대로, stab_wmshape4.sh 와 같음) + 수박 진단 판 (방식 g) 은 D1·D2·D5 (AF_E1_DG·AF_E2_DG·AF_E5_DG, 결과 불변; D3 는 안 켬 -- R 이 AF_BX 기준을 옮김).
# 단독 시간 (z): 계산 프로세스 0 인 카드 하나를 잡아 그 카드에 차례로 a · b (시작·끝 nvidia-smi·부하를 판 디렉토리에 남김).
# 사용: bash exe/stab_ghost3.sh <출력> <R2> <단계> (판 목록 exe/ghost3_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식: nd 새 실행기 비결정 + 진단 (wolf 덤프 -> bbox, ficus 덤프 -> 주기) · g 수박 --erec + D1·D2·D5 · z 단독 시간
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
BW4="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --sp2 31 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1 --rm_sub 16 --ws2_fc 20,20 --ws2_fcn 2 --ws2_win 4"   # 절 CM41 BW4 (절 CM44·CM46 바탕)
V2N="$BW4 --ws3 1 --ws3_win 1 --ws3_lj 1"; V2NF="$V2N --ws3_pf 1"                     # 절 CM44 V2N·V2NF 글자 그대로
DFL="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1"   # 공통 후보 D (절 CM37 C + 절 CM38 X3c)
base_of() { case $1 in
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M3) echo "${WMB/--cell 1.5/--cell 3}";;
esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
dfl() { local s=$1 x=" $2 "; x="${x// V2NF / $V2NF }"; x="${x// V2N / $V2N }"; case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac; x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M*) x="";; esac
  case $s in M*) case $md in b|g|z) x="--erec";; p) x="--erec --prof 1";; esac; echo "$x"; return;; esac
  case $md in nd|u) x="${x/--det /}"; [ $s = W ] && x="$x --dump --dump_part";; d) ;; n|z) x="";; p) x="--prof 1";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/ghost3_$STG.log; rclone copyto $O/ghost3_$STG.log $RO/ghost3_$STG.log -q; }
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
idle_wait() {   # 단독: 노드 7 장 모두 계산 프로세스 0 이고 1 분 부하 < 4 일 때까지 (기다린 초를 돌려줌)
  local t0=$(date +%s)
  while :; do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] && awk '{exit !($1 < 4)}' /proc/loadavg && break
    sleep 20; done
  echo $(( $(date +%s) - t0 )); }
run() {   # run 이름 씬 방식 카드 플래그
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre="" wo="" w2="" w3="" w5="" wg=""
  [ -f $O/$n/summary.json ] && return
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@e[235]' | grep -v '^@g3' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"   # '@tb 16.7' 따위 = 감싸개 인자
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = u ] && pre="AF_S3_UL=1"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E5_DG=1"                                          # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in z|p) ;; *) case " $5 " in *" @nodg "*) ;; *) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; esac;; esac   # @nodg 1 = 진단 훅 끔 (훅 무간섭 대조)                                   # 절 CM42 진단 (결과 불변; 단독 시간 판은 끔)
  c="exe/ghost3_wrap.py $wg $w5 $w3 $w2 $wo $c"
  [ $md = p ] && case $s in M*) c="exe/speed1_prof.py $c";; esac
  rm -rf $O/$n; mkdir -p $O/$n; echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  log "$n 카드 $g 시작"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1
    [ $md = u ] && $PY exe/speed3_wolfdiag.py $CELLW $O/$n > $O/$n/wolfdiag.json 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
[ -f exe/ghost3_pre_$STG.sh ] && { log "앞 단계 스크립트 ghost3_pre_$STG.sh"; bash exe/ghost3_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
SPEC=$(grep -v '^#' exe/ghost3_spec.txt | awk -v s=$STG '$1==s')
# ---- 판정·비결정·진단 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z|p) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 (프로파일 p · 시간 z): 노드 카드 모두를 잡고 (다른 묶음이 같은 표식 디렉토리로 카드를 고르면 비켜감) 노드가 비면 카드 하나에 차례로
if echo "$SPEC" | awk '$4=="z" || $4=="p"' | grep -q .; then
  ZG=$(claim_wait $$); GL=$ZG; log "단독 카드 $ZG (계산 프로세스 0 인 카드 하나)"
  for r in a b; do
    while read -r st n s md x; do
      case $md in z) ;; p) [ $r = b ] && continue;; *) continue;; esac
      [ -f $O/${n}_$r/summary.json ] && continue
      log "${n}_$r 단독 카드 $ZG 다른 계산 프로세스 $(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .) 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  for g in $GL; do rm -f $CL/$g; done
fi
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나야 (컴파일·훅 확인), 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $n 5 프레임 못 감/발산"; }
    [ $md = d ] || [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n 진단 e1 없음/오류"; }
    [ -s $O/$n/g3_err.txt ] && { OK=0; log "[연기] $n 검사 오류 (g3_err.txt)"; }
    case " $x " in *" @g3fd "*) [ -s $O/$n/g3fd.jsonl ] || { OK=0; log "[연기] $n g3fd.jsonl 없음"; };; esac
    case " $x " in *" @g3 "*) grep -q "절 CM51 ghost3\] γ̄" $O/$n/log.txt || { OK=0; log "[연기] $n CM51 ghost3 줄 없음"; };; esac
    [ $md != g ] || { [ -s $O/$n/e2.jsonl ] && [ ! -s $O/$n/e2_err.txt ] && [ -s $O/$n/e5.jsonl ] && [ ! -s $O/$n/e5_err.txt ]; } || { OK=0; log "[연기] $n 진단 e2·e5 없음/오류"; }
    case " $x " in *" @e3w"*) grep -q "절 CM45\] W̄ 고친 커널\|절 CM45\] 장벽 바닥 없는" $O/$n/log.txt || { OK=0; log "[연기] $n CM45 W̄ 줄 없음"; };; esac
    case " $x " in *" @e5"*) grep -q "절 CM49\] 표본점 기준 높이" $O/$n/log.txt || { OK=0; log "[연기] $n CM49 줄 없음"; };; esac
    case " $x " in *" @e5r"*) ;; *) continue;; esac
    [ $md != g ] || $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/e5.jsonl')]; L=[q for q in L if 'nq' in q]; e=[q.get('ref_dev_err') for q in L if q.get('ref_dev_err') is not None]; c=[q.get('car_dev_err') for q in L if q.get('car_dev_err') is not None]; print('ref', max(e) if e else None, 'car', max(c) if c else None); sys.exit(0 if L and e and max(e) < 1e-4 and (not c or max(c) < 1e-4) else 1)" >> $O/ghost3_$STG.log 2>&1 || { OK=0; log "[연기] $n D5 기준 높이 확인 (장치 = d_q + δ, R 장치 = 파이썬 재계산) 실패"; }
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/ghost3.FAIL; rclone copy $O $RO -q --include ghost3.FAIL; };;
esac
$PY exe/ghost3_tab.py $O > $O/ghost3_table.md 2> $O/ghost3_table.err; rclone copy $O $RO -q --include "ghost3_table.*"
log "단계 끝 $STG"
