#!/bin/bash
# 절 CM79 묶음 wolfh7 (exe/stab_wolfh7.sh 복사): (1) H5 (= H4 + 결과 불변 가속 C·Y '@s9c 1 @s9y 1', 절 CM76) wolf 이탈 원인 가르기 -- 벽시계 예산 Rw ('@s6rw 1')·X3g ('@s8g 1e6')·속도 판 빼기 대조,
#   (2) (Y7) P1X 강성 × 면 항복 가중 ('@w7y 1', lib/anchorflow/fem_wolfh7.py) + S_구속·C ('@w5s 1 @w5c 1'). 잡 안, 코드 디렉토리에서.
# 감싸개 사슬: exe/wolfh7_wrap.py ('@w7') -> lightg4_wrap ('@l4') -> lightg3_wrap ('@l3') -> wolfh5_wrap ('@w5') -> [첫 mfix_wrap 자리를 speed9_wrap ('@s9') 또는 speed8_wrap 로] speed8_wrap ('@s8') -> speed7_wrap ('@s7'·'@s6') -> mfix_wrap ('@mf') -> wmfloor_wrap ('@wf') -> hhtc2_wrap ('@hc'·'@rr'·'@s4'·'@tc') -> ghost6_wrap ('@g4...') -> [렌더 판: rtrender1_wrap -> speed4_wrap] -> escape4_wrap ('@e5') -> escape3 ('@e3') -> escape2 -> escape1/speed3 (나머지 '@'). '@s6l 1' 가벼운 판은 실행기 사본 exe/s6/ + 진단 끔 (단독 z 판만).
# 노드 / ·/tmp 는 안 씀 (코드·자료·TMPDIR·결과 모두 /dev/shm/dkta), /dev/shm 사용률 65 % 이상이면 판 시작을 기다림, 덤프는 R2 크기 확인 뒤 로컬 사본만 지움.
# 사용: bash exe/stab_wolfh7.sh <출력> <R2> <단계> (판 목록 exe/wolfh7_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식: nd 새 실행기 비결정 + 진단 (wolf 덤프 -> bbox, ficus 덤프 -> 주기) · g 수박 --erec + D1·D2·D5 · r 수박 --erec 만 (재현용 짧은 판)
#         h 결정적 끝 해시 (--det --fhash, 예산 @tb 1000 = 안 걸림; 진단 끔; 새 실행기 씬만)
#         z 단독 (렌더 포함 '@rr 1'): 계산 프로세스 0 인 카드 하나에 a · b 차례로 (연기 단계는 a 만)
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
source exe/fem_cm2_scn.sh
AG=/dev/shm/dkta/ag_data; mkdir -p $AG                                                         # /tmp 사본 안 읽음
for t in L000 ficus bread wolf; do cm_data $t; done
WMD=$AG
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
# 낱말: F = 공통 후보 F (E + W̄ + R2 + A2, 절 CM49), P1X = 절 CM53 최종 후보 (γ̄ 1e-1, 3x3x3 열림, 구속 노드 면 빼기, 질량 켬), A3 = 절 CM50 결과 불변 가속
dfl() { local s=$1 x=" $2 "
  case $s in M*) ;; *) [[ "$x" == *" H4n "* || "$x" == *" H5n"* ]] && x="$x --cap_mul 4";; esac  # 새 실행기 첫 할당 배수 4 (절 CM72 H2) -- 속도 판에만 (Gk = 절 CM75 판은 없음)
  local SPD="@s6o 1 @s6b 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s9c 1 @s9y 1" KAP="@l4 1 @l3 10 @l3sc 1 @l3fr 1"   # 절 CM79: H5 = H4 + C·Y; R = 벽시계 예산 Rw '@s6rw 1', X = X3g '@s8g 1e6'
  x="${x// H5n / Gpp $SPD @s6rw 1 @s8g 1e6 $KAP }"; x="${x// H5nR / Gpp $SPD @s8g 1e6 $KAP }"; x="${x// H5nX / Gpp $SPD @s6rw 1 $KAP }"; x="${x// H5nRX / Gpp $SPD $KAP }"
  x="${x// Gk / Gpp $KAP }"                                                                    # 절 CM75 판 (G'' + 가벼움 벌점 × κ, 속도 판 없음)
  x="${x// H4n / Gpp @s6o 1 @s6b 1 @s6rw 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s8g 1e6 @l4 1 @l3 10 @l3sc 1 @l3fr 1 }"; x="${x// Gpp / Gp @mf2 1 }"; x="${x// Gp / F P1X A3 @hc 5 @rr 1 @tc 8 }"
  x="${x// P1X / @g4 1e-1 @g4t 1 @g4x 1 }"; x="${x// A3 / @s4w 1 @s4y 1 @s4x 2 }"; x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in M*) case $md in g|r|z) echo "--erec";; esac; return;; esac
  case $md in h) echo "--det --fhash"; return;; esac
  case $md in nd) case $s in L) x="--hgt --pdiag --qjd --bar_cnt";; W) x="--hgt --fhash --bar_cnt --dump --dump_part";; F) x="--pdiag --qjd --fhash --dump";; B) x="--pdiag --qjd --fhash";; esac;; *) x="";; esac
  echo "$x"; }
W=/home/dkta/work
RD=/dev/shm/dkta/wolfh7; mkdir -p $RD                                                          # 이 묶음이 만드는 정답 궤적 (렌더 카메라·0 프레임) 자리 (t2chain 것이 있으면 읽기만)
gtd() { local n=$1 d; for d in /dev/shm/dkta/ag_data/gt_$n /dev/shm/dkta/t2chain/gt_$n /dev/shm/dkta/rtrender1/gt_$n /dev/shm/dkta/hhtc/gt_$n /dev/shm/dkta/hhtc2/gt_$n /dev/shm/dkta/lightg3/gt_$n $RD/gt_$n; do [ -f $d/meta.json ] && [ -f $d/sim_0000000000.h5 ] && { echo $d; return; }; done; echo $RD/gt_$n; }
mkgt() { local n=$1 d=$2; shift 2; [ -f $d/meta.json ] && return                              # exe/stab_rtrender1.sh 와 같은 변환 (CPU, 앞 1 프레임)
  CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py "$@" --out_pt $RD/tmp_$n.pt --out_dir $d --out_dir_n 1 > $RD/mkgt_$n.log 2>&1; rm -f $RD/tmp_$n.pt; }
GTW=$(gtd wolf); GTB=$(gtd bread); GTM=$(gtd wm)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; WCF=$W/wm2/gf/config.json
FGT=$W/lbours/ficus_v0/gt
rr_of() { case $1 in                                                                         # 절 CM54 실시간 렌더 (가장 빠른 결과 불변 갱신: PG 1, 수박 2) -- 절 CM56 과 같은 글자
  L) echo "--rr pg --rr_shape lego --rr_traj $W/lbdata/test_low/L000 --rr_cfg $W/lbdata/config.json --rr_fast 1";;
  F) echo "--rr pg --rr_shape ficus --rr_traj $FGT --rr_cfg $FGT/config.json --rr_fast 1";;
  W) echo "--rr pg --rr_shape wolf --rr_traj $GTW --rr_cfg $GTW/config.json --rr_fast 1";;
  B) echo "--rr pg --rr_shape bread --rr_traj $GTB --rr_cfg $GTB/config.json --rr_fast 1";;
  M3) echo "--rr gf --rr_shape watermelon --rr_traj $GTM --rr_cfg $WCF --rr_model $WMM --rr_gf $W/GaussianFluent --rr_fast 2";;
esac; }
shm_use() { df --output=pcent /dev/shm | tail -1 | tr -dc '0-9'; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfh7_$STG.log; rclone copyto $O/wolfh7_$STG.log $RO/wolfh7_$STG.log -q; }
shm_wait() { while [ "$(shm_use)" -ge 65 ]; do log "/dev/shm 사용률 $(shm_use) % -- 기다림"; sleep 120; done; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL                                                   # 다른 묶음과 같은 카드 표식 디렉토리 (서로 비켜감)
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b pre="" wo="" w2="" w3="" w5="" wg="" hc="" rr="" w4="" tc="" wf="" mf="" l3="" l4="" w7="" s5="" s7="" s8="" s9="" lean=0
  [ -f $O/$n/summary.json ] && return
  hc="$(echo " $x " | grep -o '@hc [0-9]*' | awk '{print $2}')"; tc="$(echo " $x " | grep -o '@tc [0-9]*' | awk '{print $2}')"; rr="$(echo " $x " | grep -o '@rr [0-9]*' | awk '{print $2}')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  local w5x="$(echo " $x " | grep -o '@w5[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"           # 절 CM70 S_구속 · C (exe/wolfh5_wrap.py)
  w7="$(echo " $x " | grep -o '@w7[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"                   # 절 CM79 (Y7)
  s9="$(echo " $x " | grep -o '@s9[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"                # 절 CM76 C·Y
  s5="$(echo " $x " | grep -o '@s6[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s7="$(echo " $x " | grep -o '@s7[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s8="$(echo " $x " | grep -o '@s8[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; case " $x " in *" @s6l 1 "*) lean=1;; esac
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@\(e[235]\|g[456]\|hc\|rr\|s4\|s6\|s7\|s8\|s9\|tc\|wf\|mf\|l3\|l4\|w5\|w7\)' | sed 's/^@/--/' | tr '\n' ' ')"; wf="$(echo " $x " | grep -o '@wf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; mf="$(echo " $x " | grep -o '@mf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l3="$(echo " $x " | grep -o '@l3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l4="$(echo " $x " | grep -o '@l4[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g[456][a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E5_DG=1"                                                 # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in g|nd) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; z) pre="AF_E1_DG=1";; esac
  [ $lean = 1 ] && { pre=""; c="${c/exe\/fem_an_rt_sim.py/exe/s6/fem_an_rt_sim.py}"; c="${c/exe\/fem_ao_rt_sim.py/exe/s6/fem_ao_rt_sim.py}"; }   # 절 CM65 가벼운 판: 실행기 사본 + 진단 끔 (절 CM74 speed8 과 같음)           # 단독 판도 E1 (빠른 노드·이탈, 프레임 사이 읽기만; 절 CM56 단독 판과 같음)                                   # 절 CM42 진단 (결과 불변; 재현·단독 판은 끔)
  local hx=""; [ -n "$hc" ] && hx="--hc $hc"; [ -n "$tc" ] && hx="$hx --tc $tc"
  [ "${rr:-0}" = 1 ] && hx="$hx $(rr_of $s) $w4"
  c="exe/wolfh7_wrap.py $w7 $l4 $l3 $w5x $s9 $s8 $s7 $s5 $mf $wf $hx $wg $w5 $w3 $w2 $wo $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; echo "카드 $g 계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_start.txt; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  shm_wait
  log "$n 카드 $g 시작 (/dev/shm $(shm_use) %, 부하 $(cut -d' ' -f1 /proc/loadavg))"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_end.txt; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1
    $PY exe/wolfh1_an.py $AG/wolf.pt $O/$n/an.json $O/$n > $O/$n/an.log 2>&1                    # 절 CM52 진단 (높이·면적·회전 반경; 읽기만)
    $PY exe/wolfh4_relfall.py $AG/wolf.pt $n=$O/$n/pred.pt > $O/$n/relfall.md 2> $O/$n/relfall.err; fi   # 절 CM68·CM70 방출 뒤 낙하 몫 (읽기만)
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) hc 줄 $(wc -l < $O/$n/hc.jsonl 2>/dev/null) 렌더 $(wc -l < $O/$n/rr.jsonl 2>/dev/null) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  if [ -f $O/$n/pred.pt ]; then                                                              # 덤프: R2 에 올리고 크기 확인 뒤 로컬 사본만 지움 (R2 에 그대로 남음)
    rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
    [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && rm -f $O/$n/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"
  fi
  for z in $O/$n/e2_f*.npz; do [ -f "$z" ] || continue                                        # 진단 D2 덤프도 R2 크기 확인 뒤 로컬 사본 지움
    [ "$(rclone size $RO/$n/$(basename $z) --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $z)" ] && rm -f $z
  done; }
SPEC=$(grep -v '^#' exe/wolfh7_spec.txt | awk -v s=$STG '$1==s')
[ -f exe/wolfh7_pre_$STG.sh ] && { log "앞 단계 스크립트 wolfh7_pre_$STG.sh"; bash exe/wolfh7_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
if echo "$SPEC" | grep -q '@rr 1'; then
  mkgt wolf $GTW --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json
  mkgt bread $GTB --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json
  mkgt wm $GTM --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --allow_nonfinite
  log "렌더 자료: ficus $FGT wolf $GTW ($(ls $GTW/meta.json 2>&1)) bread $GTB ($(ls $GTB/meta.json 2>&1)) 수박 $GTM ($(ls $GTM/meta.json 2>&1)) 모델 $WMM"
fi
# ---- 판정·비결정·진단·재현 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 (렌더 포함): 계산 프로세스 0 인 카드 하나를 잡아 그 카드에 차례로 a · b (연기 단계는 a 만)
if echo "$SPEC" | awk '$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG (계산 프로세스 0 인 카드 하나)"
  RS="a b"; case $STG in *k) RS="a";; esac
  for r in $RS; do
    while read -r st n s md x; do
      [ "$md" = z ] || continue
      [ -f $O/${n}_$r/summary.json ] && continue
      log "${n}_$r 단독 카드 $ZG 다른 계산 프로세스 $(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .) 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나야 (컴파일·훅 확인), 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    [ $md = z ] && n=${n}_a
    x="$(dfl $s "$x")"                                                                       # 낱말 (H4n 등) 펼친 글자로 검사
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $n 5 프레임 못 감/발산"; }
    grep -q "절 CM61\] hhtc2 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM61 줄 없음"; }
    case " $x " in *" @s7nm 1 "*) ;; *" @mf"*) grep -q "절 CM67\] mfix 켬" $O/$n/log.txt && [ ! -s $O/$n/mf_err.txt ] && [ -s $O/$n/mf.jsonl ] || { OK=0; log "[연기] $n CM67 줄·mf.jsonl 없음/오류"; };; esac
    case "$s $x " in M*) ;; *" @w5"*) grep -q "절 CM70\] wolfh5 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM70 줄 없음"; };; esac
    case " $x " in *" @w7y 1"*) grep -q "절 CM79\] wolfh7 켬" $O/$n/log.txt && grep -q "절 CM79 wolfh7\] (Y7) 재질" $O/$n/log.txt && grep -q "절 CM79 wolfh7\] (Y7) 훅" $O/$n/log.txt && $PY -c "import json,sys; i=json.load(open('$O/$n/w7info.json')); print('$n w7', {k: i[k] for k in ('nsrc','nw7','fnset','miss','nlaunch','nalloc')}); sys.exit(0 if i['nsrc'] >= 1 and i['nw7'] >= 1 and i['fnset'] >= 5 and not i['miss'] and len(i['nlaunch']) >= 3 else 1)" >> $O/wolfh7_$STG.log 2>&1 || { OK=0; log "[연기] $n CM79 줄·w7info 없음/고침 못 함"; };; esac
    case " $x " in *" @s9"*) grep -q "절 CM76\] speed9 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM76 speed9 줄 없음"; };; esac
    case " $x " in *" @l4 1"*) grep -q "절 CM75\] lightg4 켬" $O/$n/log.txt && grep -q "절 CM75 lightg4\] 재질" $O/$n/log.txt && [ -s $O/$n/l4info.json ] || { OK=0; log "[연기] $n CM75 줄·l4info.json 없음"; };; esac
    case " $x " in *" @l3 "*) grep -q "절 CM73\] lightg3 켬" $O/$n/log.txt && [ ! -s $O/$n/lg3_err.txt ] || { OK=0; log "[연기] $n CM73 줄 없음/lg3_err"; };; esac
    case " $x " in *" @l3fd "*) case $s in M3|W|B) [ -s $O/$n/lg3fd.jsonl ] || { OK=0; log "[연기] $n lg3fd.jsonl 없음"; };; esac;; esac
    case " $x " in *" @l3ck 1"*) case $s in M3) grep -q "\"ck\": {" $O/$n/lg3.jsonl 2>/dev/null && ! grep -q "ck_err" $O/$n/lg3.jsonl || { OK=0; log "[연기] $n 금 면 진단 없음/오류"; };; esac;; esac
    case " $x " in *" @wfm "*) grep -q "절 CM64\] wmfloor 켬" $O/$n/log.txt && [ ! -s $O/$n/wf_err.txt ] && [ -s $O/$n/wf.jsonl ] || { OK=0; log "[연기] $n CM64 줄·wf.jsonl 없음/오류"; };; esac
    [ $md = z ] && continue                                                                  # 가벼운 판 단독: 진단 기록 없음이 정상 (위 훅 줄·w6info 만 봄)
    case $s in F|B) [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc 오류"; };; *) [ -s $O/$n/hc.jsonl ] && [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc.jsonl 없음/오류"; };; esac   # ficus·bread: 장벽 없음 -> 표시 커널 안 돎 (버퍼·기록 없음이 정상)
    case $s in M3|L|W) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/hc.jsonl')]; i=json.load(open('$O/$n/hcinfo.json')); print('$n', i, sum(q['nstep'] for q in L), sum(q['ncol'] for q in L), sum(q['new'] for q in L), sum(q['napp'] for q in L)); sys.exit(0 if i['nbar'] >= 1 and i['nhc'] >= 1 and sum(q['nstep'] for q in L) > 0 and all(len(q['mo']) == min(q['nstep'], 32) for q in L) and L[-1]['mo'][-1][0] > 0 else 1)" >> $O/wolfh7_$STG.log 2>&1 || { OK=0; log "[연기] $n 장벽 씬인데 표시 커널이 안 돎"; };; esac
    case " $x " in *" @g4 "*|*" P1X "*) grep -q "절 CM57\] ghost penalty 6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM57 ghost6 줄 없음"; };; esac
    case " $x " in *" @rr 1"*) [ "$(wc -l < $O/$n/rr.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n rr.jsonl 없음"; };; esac
    case $md in g|nd) [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n 진단 e1 없음/오류"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/wolfh7.FAIL; rclone copy $O $RO -q --include wolfh7.FAIL; };;
esac
$PY exe/wolfh7_tab.py $O > $O/wolfh7_table.md 2> $O/wolfh7_table.err; rclone copy $O $RO -q --include "wolfh7_table.*"
log "단계 끝 $STG"
