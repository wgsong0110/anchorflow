#!/bin/bash
# 절 CM39 묶음 speed3: 공통 후보 D (C + X3c) 60 fps -- 최악 프레임 프로파일·결과 불변 가속·결과를 바꾸는 후보. 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_speed3.sh <출력> <R2> <단계>   -- 판 목록 = exe/speed3_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄 (플래그의 'D' 글자는 공통 후보 D 로 펼침).
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
dfl() { local s=$1 x=" $2 "; x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M*) x="";; esac
  case $s in M*) case $md in b|z) x="--erec";; p) x="--erec --prof 1";; esac; echo "$x"; return;; esac
  case $md in nd|u) x="${x/--det /}"; [ $s = W ] && x="$x --dump --dump_part";; n|z) x="";; p) x="--prof 1";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/speed3_$STG.log; rclone copyto $O/speed3_$STG.log $RO/speed3_$STG.log -q; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre=""
  [ -f $O/$n/summary.json ] && return
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = u ] && c="exe/speed3_wrap.py $c" && pre="AF_S3_UL=1"
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
SPEC=$(grep -v '^#' exe/speed3_spec.txt | awk -v s=$STG '$1==s')
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
  GL=""; for g in $(seq 0 $((NG - 1))); do gg=$(claim_wait $$); GL="$GL $gg"; done
  ZG=$(echo $GL | awk '{print $1}'); log "단독 카드 $ZG (잡은 카드$GL)"
  for r in a b; do
    while read -r st n s md x; do
      case $md in z) ;; p) [ $r = b ] && continue;; *) continue;; esac
      [ -f $O/${n}_$r/summary.json ] && continue
      W=$(idle_wait); log "${n}_$r 단독 기다림 ${W}s"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  for g in $GL; do rm -f $CL/$g; done
fi
$PY exe/speed3_tab.py $O > $O/speed3_table.md 2> $O/speed3_table.err; rclone copy $O $RO -q --include "speed3_table.*"
log "단계 끝 $STG"
