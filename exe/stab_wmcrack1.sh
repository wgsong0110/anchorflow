#!/bin/bash
# 절 CM59 묶음 wmcrack1 (수박이 녹지 않고 첫 충돌에 깨지게): exe/stab_ghost6.sh 틀 + 맨 앞 감싸개 exe/wmcrack1_wrap.py ('@wc1*' 인자) -> exe/ghost6_wrap.py ('@g4*·@g5*·@g6*') -> escape4 ('@e5*') -> escape3 ('@e3*') -> escape2 ('@e2*') -> speed3 (나머지 '@').
# 낱말: P1X = 공통 후보 F + 벌점 P1X (절 CM53, '@g4 1e-1 @g4t 1 @g4x 1'), P1XK = P1X 에서 쪼개기·X3c (--ksplit 3 --ks_big --wrx 3 --wrx_col 1) 만 뺀 것 (--K 고정 진단용).
# 수박 방식 r = 기록 판 (--erec + --wsh_rec 30 --wsh_sv 프레임 + '@wc1sv 1' 탄성 F 특이값 + 진단 D1·D2·D5, 모두 결과 불변), z = 단독 시간 (--erec 만).
# 다른 씬 (L 레고 L000 · W wolf · F ficus · B bread) 방식 nd·z 는 stab_ghost6.sh 그대로.
# 노드 / ·/tmp 안 씀 (자료·TMPDIR /dev/shm/dkta), /dev/shm 사용률 < 65 % 일 때만 판 시작, 덤프 (pred.pt) 는 R2 크기 확인 뒤 로컬 사본 지움.
# 사용: bash exe/stab_wmcrack1.sh <출력> <R2> <단계> (판 목록 exe/wmcrack1_spec.txt)
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
AG=/dev/shm/dkta/ag_data; mkdir -p $AG                                                         # 절 CM55: /tmp 사본 안 읽음
for t in L000 ficus bread wolf; do cm_data $t; done
WMD=/dev/shm/dkta/ag_data; mkdir -p $WMD                                                                 # 절 CM55: 노드 / 와 /tmp 안 씀
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
JP=/dev/shm/dkta/ag_data/gt_jp; mkdir -p $JP
[ $(ls $JP | wc -l) -ge 101 ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $JP -q --transfers 16
mkdir -p $O/gt; [ -s $O/gt/gt_wmshape.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape/gt/gt_wmshape.json $O/gt/gt_wmshape.json -q
[ -s $O/gt/gt_wmshape2.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape2/gt/gt_wmshape2.json $O/gt/gt_wmshape2.json -q
GH5=/home/dkta/work/wm2/gf/simulation_ply
SVF=3,4,5,6,8,10,15,20,30,50,100
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --gq_oct 0 --ws_bar 4 --ws_fi"
BW4="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --sp2 31 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1 --rm_sub 16 --ws2_fc 20,20 --ws2_fcn 2 --ws2_win 4"   # 절 CM41 BW4 (절 CM44·CM46 바탕)
V2N="$BW4 --ws3 1 --ws3_win 1 --ws3_lj 1"; V2NF="$V2N --ws3_pf 1"                     # 절 CM44 V2N·V2NF 글자 그대로
DFL="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1"   # 공통 후보 D (절 CM37 C + 절 CM38 X3c)
DFLK="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --xg 1"   # 진단: D 에서 쪼개기·X3c 만 뺌 (--K 고정)
P1="@e3w 1 @e5r 2 @e5a 2 @g4 1e-1 @g4t 1 @g4x 1"                          # 절 CM53 P1X (F 위)
base_of() { case $1 in
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M3) echo "${WMB/--cell 1.5/--cell 3}";;
esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
dfl() { local s=$1 x=" $2 "; x="${x// P1XK / EK $P1 }"; x="${x// P1X / E $P1 }"; x="${x// V2NF / $V2NF }"; x="${x// V2N / $V2N }"; case $s in M*) x="${x// EK / DK @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }"; x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// EK / DK @tb 16.7 @tb_m 0.8 @tb_w 2 }"; x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac; x="${x// DK / $DFLK }"; x="${x// D / $DFL }"; case $s in M*) x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M*) x="";; esac
  case $s in M*) case $md in r) x="--erec --wsh_rec 30 --wsh_sv $SVF";; b|g|z) x="--erec";; p) x="--erec --prof 1";; esac; echo "$x"; return;; esac
  case $md in nd|u|zd) x="${x/--det /}"; [ $s = W ] && x="$x --dump --dump_part"; [ $md = zd ] && [ $s = L ] && x="$x --dump";; d) ;; n|z) x="";; p) x="--prof 1";; esac
  echo "$x"; }
shm_use() { df --output=pcent /dev/shm | tail -1 | tr -dc '0-9'; }
shm_wait() { while [ "$(shm_use)" -ge 65 ]; do log "/dev/shm 사용률 $(shm_use) % -- 기다림"; sleep 120; done; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmcrack1_$STG.log; rclone copyto $O/wmcrack1_$STG.log $RO/wmcrack1_$STG.log -q; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre="" wo="" wc="" w2="" w3="" w5="" wg=""
  [ -f $O/$n/summary.json ] && return
  wc="$(echo " $x " | grep -o '@wc1[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ $md = r ] && wc="$wc --wc1sv 1"; wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@e[235]' | grep -v '^@g[456]' | grep -v '^@wc1' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g[456][a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"   # '@tb 16.7' 따위 = 감싸개 인자
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = u ] && pre="AF_S3_UL=1"
  case $md in g|r) pre="AF_E2_DG=1 AF_E5_DG=1";; esac                                          # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in z|p) ;; *) case " $5 " in *" @nodg "*) ;; *) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; esac;; esac   # @nodg 1 = 진단 훅 끔 (훅 무간섭 대조)                                   # 절 CM42 진단 (결과 불변; 단독 시간 판은 끔)
  c="exe/wmcrack1_wrap.py $wc $wg $w5 $w3 $w2 $wo $c"
  [ $md = p ] && case $s in M*) c="exe/speed1_prof.py $c";; esac
  rm -rf $O/$n; mkdir -p $O/$n; echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  shm_wait
  log "$n 카드 $g 시작 (/dev/shm $(shm_use) %)"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1
    [ $md = u ] && $PY exe/speed3_wolfdiag.py $CELLW $O/$n > $O/$n/wolfdiag.json 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  if [ -f $O/$n/pred.pt ]; then                                                              # /dev/shm 가 거의 참 (11-08 7.9 GB 남음): 덤프는 R2 에 올리고 크기 확인 뒤 로컬 사본만 지움 (R2 에 그대로 남음)
    rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
    [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && rm -f $O/$n/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"
  fi
  for z in $O/$n/g4cells_f*.npz; do [ -f "$z" ] || continue                                   # 절 CM55: 칸 덤프도 R2 크기 확인 뒤 로컬 사본 지움
    [ "$(rclone size $RO/$n/$(basename $z) --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $z)" ] && rm -f $z
  done; }
[ -f exe/wmcrack1_pre_$STG.sh ] && { log "앞 단계 스크립트 wmcrack1_pre_$STG.sh"; bash exe/wmcrack1_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
SPEC=$(grep -v '^#' exe/wmcrack1_spec.txt | awk -v s=$STG '$1==s')
# ---- 판정·비결정·진단 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z|zd|p) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 (프로파일 p · 시간 z): 노드 카드 모두를 잡고 (다른 묶음이 같은 표식 디렉토리로 카드를 고르면 비켜감) 노드가 비면 카드 하나에 차례로
if echo "$SPEC" | awk '$4=="z" || $4=="zd" || $4=="p"' | grep -q .; then
  ZG=$(claim_wait $$); GL=$ZG; log "단독 카드 $ZG (계산 프로세스 0 인 카드 하나)"
  for r in a b; do
    while read -r st n s md x; do
      case $md in z|zd) ;; p) [ $r = b ] && continue;; *) continue;; esac
      [ -f $O/${n}_$r/summary.json ] && continue
      log "${n}_$r 단독 카드 $ZG 다른 계산 프로세스 $(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .) 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  for g in $GL; do rm -f $CL/$g; done
fi
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나야, 감싸개 줄·기록 확인, 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $n 5 프레임 못 감/발산"; }
    [ -s $O/$n/wc1_err.txt ] && { OK=0; log "[연기] $n 감싸개 오류 (wc1_err.txt)"; }
    case " $x " in *" @wc1k 1 "*) grep -q "절 CM59\] --wc1k" $O/$n/log.txt && [ -s $O/$n/wc1.jsonl ] || { OK=0; log "[연기] $n wc1k 줄·wc1.jsonl 없음"; };; esac
    case "$s $md" in "M3 r") [ -s $O/$n/wsh/fs_004.npz ] && [ -s $O/$n/wsh/xs_004.npz ] || { OK=0; log "[연기] $n wsh fs/xs 없음"; };; esac
    case " $x " in *" @g4 "*) grep -q "절 CM57\] ghost penalty 6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM57 ghost6 줄 없음"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/wmcrack1.FAIL; rclone copy $O $RO -q --include wmcrack1.FAIL; };;
esac
[ -f exe/wmcrack1_post_$STG.sh ] && { log "뒤 단계 스크립트 wmcrack1_post_$STG.sh"; bash exe/wmcrack1_post_$STG.sh $O $RO $JP $WMD $GH5 > $O/post_$STG.out 2>&1; log "뒤 단계 rc=$?"; rclone copy $O $RO -q --include "post_$STG.out" --include "wmcrack1_prof*"; }
$PY exe/wmcrack1_tab.py $O $JP > $O/wmcrack1_table.md 2> $O/wmcrack1_table.err; rclone copy $O $RO -q --include "wmcrack1_table.*"
log "단계 끝 $STG"
