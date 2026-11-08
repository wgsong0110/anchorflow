#!/bin/bash
# 절 CM45 묶음 escape3: 장벽 표본점 이동 기저 W̄ (감싸개 exe/escape3_wrap.py '@e3...') + 수박 E+G 진단 D3 + 다른 씬 회귀. 잡 안, 커밋 디렉토리에서.
# exe/stab_escape2.sh (절 CM43) 그대로 + 맨 앞 감싸개 exe/escape3_wrap.py ('@e3...' 인자; '@e2...' 는 escape2 감싸개, 나머지 '@' 는 escape1/speed3) + 방식 g 에 진단 D3 (AF_E3_DG=1).
# 단독 시간 (z): 노드 전체가 비기를 기다리지 않고 (다른 에이전트가 다른 카드를 씀) 계산 프로세스 0 인 카드 하나를 잡아 그 카드에 차례로 a · b (시작·끝 nvidia-smi·부하를 판 디렉토리에 남김).
# 사용: bash exe/stab_escape3.sh <출력> <R2> <단계> (판 목록 exe/escape3_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기, stab_wmredesign.sh 바탕 글자 그대로) · M3 수박 (옛 실행기 셀 3, 절 CM38 바탕 글자 그대로; D 에 --sp2 31 을 더함)
#   방식: d 결정적 판정 · nd 비결정 + 진단 (wolf 는 덤프 -> 입자 이탈 bbox) · n 비결정 진단 없음 · b 수박 --erec · u wolf 이탈 진단 (비결정 덤프 + AF_S3_UL=1 감싸개, 결과 불변)
#         p 단독 프로파일 (새 실행기 --prof 1 CUDA 이벤트; 수박 exe/speed1_prof.py 동기 벽시계) · z 단독 시간 -- p·z 는 판정 판이 모두 끝난 뒤, 노드 7 장 모두 계산 프로세스 0 일 때 카드 하나에 차례로 (a·b).
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
dfl() { local s=$1 x=" $2 "; case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac; x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M*) x="";; esac
  case $s in M*) case $md in b|g|z) x="--erec";; p) x="--erec --prof 1";; esac; echo "$x"; return;; esac
  case $md in nd|u) x="${x/--det /}"; [ $s = W ] && x="$x --dump --dump_part";; n|z) x="";; p) x="--prof 1";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/escape3_$STG.log; rclone copyto $O/escape3_$STG.log $RO/escape3_$STG.log -q; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre="" wo="" w2="" w3=""
  [ -f $O/$n/summary.json ] && return
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+-]*' | grep -v '^@e[23]' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+-]*//g')"   # '@tb 16.7' 따위 = 감싸개 인자 (@e2* 는 escape3 감싸개 앞)
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = u ] && pre="AF_S3_UL=1"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E3_DG=1"                                          # 절 CM43 진단 D2 + 절 CM45 진단 D3 (결과 불변)
  case $md in z|p) ;; *) case " $5 " in *" @nodg "*) ;; *) pre="$pre AF_E1_DG=1";; esac;; esac   # @nodg 1 = 진단 훅 끔 (훅 무간섭 대조)                                   # 절 CM42 진단 (결과 불변; 단독 시간 판은 끔)
  c="exe/escape3_wrap.py $w3 $w2 $wo $c"
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
[ -f exe/escape3_pre_$STG.sh ] && { log "앞 단계 스크립트 escape3_pre_$STG.sh"; bash exe/escape3_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
SPEC=$(grep -v '^#' exe/escape3_spec.txt | awk -v s=$STG '$1==s')
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
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || OK=0
    [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n 진단 e1 없음/오류"; }                       # 절 CM42 진단 훅 확인
    [ $md != g ] || { [ -s $O/$n/e2.jsonl ] && [ ! -s $O/$n/e2_err.txt ] && [ -s $O/$n/e3.jsonl ] && [ ! -s $O/$n/e3_err.txt ]; } || { OK=0; log "[연기] $n 진단 e2·e3 없음/오류"; }   # 절 CM43 D2·절 CM45 D3 확인
    case " $x " in *" @e3w"*) grep -q "절 CM45\] W̄ 고친 커널\|절 CM45\] 장벽 바닥 없는" $O/$n/log.txt || { OK=0; log "[연기] $n CM45 W̄ 줄 없음"; };; esac
    case " $x " in *" @e3w"*) [ $md != g ] || $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/e3.jsonl')]; L=[q for q in L if 'nq' in q]; sys.exit(0 if L and all(q.get('wsum_bad', 1) == 0 and q.get('dwq_max', 1) < 1e-4 for q in L) else 1)" || { OK=0; log "[연기] $n W̄ 확인 (ΣW̄ = 1, |d_wq − d_pm| < 1e-4) 실패"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/escape3.FAIL; rclone copy $O $RO -q --include escape3.FAIL; };;
esac
$PY exe/escape3_tab.py $O > $O/escape3_table.md 2> $O/escape3_table.err; rclone copy $O $RO -q --include "escape3_table.*"
log "단계 끝 $STG"
