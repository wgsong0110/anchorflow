#!/bin/bash
# 절 CM78 묶음 speed10 (exe/stab_speed9.sh 복사; 감싸개 앞에 exe/speed10_wrap.py ('@s10n' 노드 상자 압축 · '@s10k' 목록 · '@s10g' 격자 · '@s10cap' 첫 할당 상자 배수) + 공통 후보 H5 = H4 (H3 + 가벼움 벌점 x κ '@l4 1 @l3 10 @l3sc 1 @l3fr 1', 절 CM75) + C·Y ('@s9c 1 @s9y 1', 절 CM76); 사슬 speed10_wrap -> lightg4 -> lightg3 -> wolfh5 -> [mfix 자리 speed9_wrap] -> speed8 -> speed7 -> mfix -> ...). 아래는 speed9 설명 그대로 --
# 절 CM74 묶음 speed8 (exe/stab_speed7.sh 복사; 감싸개 앞에 exe/speed8_wrap.py ('@s8w' 워프 모으기 · '@s8g' X3g)): 공통 후보 H2 = Hc '@s7nm 1 @s7z 1 @s7p 1' (+ 새 실행기 --cap_mul 4). 아래는 speed7 설명 그대로 --: 공통 후보 H = G'' ('Gpp' = G' + 절 CM67 (2) '@mf2 1') + O + L + B + Rw ('Hc' = Gpp '@s6o 1 @s6l 1 @s6b 1 @s6rw 1') 위 수박 커널 프로파일 ('@s7kp a,b') · 결과 불변 가속 ('@s7*') · 다섯 씬 렌더 포함 벽시계. 잡 안, 코드 디렉토리에서.
# 감싸개 사슬: exe/speed8_wrap.py ('@s8*') -> exe/speed7_wrap.py ('@s7*'·'@s6*') -> exe/mfix_wrap.py ('@mf') -> wmfloor_wrap -> hhtc2_wrap ('@hc'·'@rr'·'@s4'·'@tc') -> ghost6_wrap ('@g4...') -> [렌더 판: rtrender1_wrap -> speed4_wrap] -> escape4_wrap ('@e5') -> escape3 ('@e3') -> escape2 -> escape1/speed3 (나머지 '@'). 가벼운 판은 실행기 사본 exe/s6/ + 진단 E1 끔.
# 노드 / ·/tmp 는 안 씀 (코드·자료·TMPDIR·결과 모두 /dev/shm/dkta), /dev/shm 사용률 65 % 이상이면 판 시작을 기다림, 덤프는 R2 크기 확인 뒤 로컬 사본만 지움.
# 사용: bash exe/stab_speed10.sh <출력> <R2> <단계> (판 목록 exe/speed10_spec.txt)
#   씬: L 레고 · W wolf · F ficus · B bread (새 실행기) · M3 수박 (옛 실행기 셀 3)
#   방식: nd 새 실행기 비결정 + 진단 (wolf 덤프 -> bbox, ficus 덤프 -> 주기) · g 수박 --erec + D1·D2·D5 · r 수박 --erec 만 (재현용 짧은 판)
#         h 결정적 끝 해시 (--det --fhash, 새 실행기) · p 단계 기록 판 (단독 카드, a · b)
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
  case $s in M*) ;; *) [[ "$x" == *" H5 "* ]] && x="$x --cap_mul 4";; esac                     # 새 실행기 첫 할당 배수 4 (절 CM72 H2)
  local KL; KL="$(tr -d ' \n' < exe/speed10_klist.txt 2>/dev/null)"                      # 압축 목록 (s0 확인 판에서 다름 0 인 커널; exe/speed10_klist.txt)
  x="${x// H6 / H5 @s10n 1 @s10k $KL @s10cap 8 }"; x="${x// KL / $KL }"                     # H6 = H5 + 노드 상자 압축 N + 첫 할당 상자 배수 8
  x="${x// H5 / H3 @l4 1 @l3 10 @l3sc 1 @l3fr 1 @s9c 1 @s9y 1 }"
  x="${x// H3 / Hc @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s8g 1e6 }"; x="${x// Hc / Gpp @s6o 1 @s6l 1 @s6b 1 @s6rw 1 }"; x="${x// Gpp / Gp @mf2 1 }"; x="${x// Gp / F P1X A3 @hc 5 @rr 1 @tc 8 }"; x="${x// G / F P1X A3 @hc 4 @rr 1 @tc 8 }"; x="${x// P1X / @g4 1e-1 @g4t 1 @g4x 1 }"; x="${x// A3 / @s4w 1 @s4y 1 @s4x 2 }"; x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  x="${x// D / $DFL }"; case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in M*) case $md in g|r|z|p|q) echo "--erec";; esac; return;; esac
  case $md in h) echo "--det --fhash"; return;; esac
  case $md in nd) case $s in L) x="--hgt --pdiag --qjd --bar_cnt";; W) x="--hgt --fhash --bar_cnt --dump --dump_part";; F) x="--pdiag --qjd --fhash --dump";; B) x="--pdiag --qjd --fhash";; esac;; *) x="";; esac
  echo "$x"; }
W=/home/dkta/work
RD=/dev/shm/dkta/speed10; mkdir -p $RD                                                          # 이 묶음이 만드는 정답 궤적 (렌더 카메라·0 프레임) 자리 (t2chain 것이 있으면 읽기만)
gtd() { local n=$1 d; for d in /dev/shm/dkta/ag_data/gt_$n /dev/shm/dkta/t2chain/gt_$n /dev/shm/dkta/rtrender1/gt_$n /dev/shm/dkta/speed6/gt_$n /dev/shm/dkta/speed7/gt_$n /dev/shm/dkta/speed8/gt_$n /dev/shm/dkta/speed9/gt_$n $RD/gt_$n; do [ -f $d/meta.json ] && [ -f $d/sim_0000000000.h5 ] && { echo $d; return; }; done; echo $RD/gt_$n; }
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
log() { echo "[$(date '+%F %T')] $*" >> $O/speed10_$STG.log; rclone copyto $O/speed10_$STG.log $RO/speed10_$STG.log -q; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b s5="" s7="" s8="" s9="" s10="" l3="" l4="" mf="" pre="" lean=0 wo="" w2="" w3="" w5="" wg="" hc="" rr="" w4="" tc=""
  [ -f $O/$n/summary.json ] && return
  hc="$(echo " $x " | grep -o '@hc [0-9]*' | awk '{print $2}')"; tc="$(echo " $x " | grep -o '@tc [0-9]*' | awk '{print $2}')"; rr="$(echo " $x " | grep -o '@rr [0-9]*' | awk '{print $2}')"
  s5="$(echo " $x " | grep -o '@s6[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s7="$(echo " $x " | grep -o '@s7[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s8="$(echo " $x " | grep -o '@s8[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s9="$(echo " $x " | grep -o '@s9[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s10="$(echo " $x " | grep -o '@s10[a-z0-9_]* [0-9a-z_.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l3="$(echo " $x " | grep -o '@l3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l4="$(echo " $x " | grep -o '@l4[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; mf="$(echo " $x " | grep -o '@mf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; case " $x " in *" @s6l 1 "*) lean=1;; esac
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@\(e[235]\|g[456]\|hc\|rr\|s4\|s6\|s7\|s8\|s9\|s10\|l3\|l4\|mf\|tc\)' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g[456][a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9a-z_.e+,-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E5_DG=1"                                                 # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in g|nd) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; z|p|q) pre="AF_E1_DG=1";; esac
  [ $lean = 1 ] && { pre=""; c="${c/exe\/fem_an_rt_sim.py/exe/s6/fem_an_rt_sim.py}"; c="${c/exe\/fem_ao_rt_sim.py/exe/s6/fem_ao_rt_sim.py}"; }   # 절 CM65 가벼운 판: 실행기 사본 + 진단 끔           # 단독 판도 E1 (빠른 노드·이탈, 프레임 사이 읽기만; 절 CM56 단독 판과 같음)                                   # 절 CM42 진단 (결과 불변; 재현·단독 판은 끔)
  local hx=""; [ -n "$hc" ] && hx="--hc $hc"; [ -n "$tc" ] && hx="$hx --tc $tc"
  [ "${rr:-0}" = 1 ] && hx="$hx $(rr_of $s) $w4"
  c="exe/speed10_wrap.py $s10 $l4 $l3 $s9 $s8 $s7 $s5 $mf $hx $wg $w5 $w3 $w2 $wo $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1; echo "카드 $g 계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_start.txt; cat /proc/loadavg >> $O/$n/apps_at_start.txt
  shm_wait
  log "$n 카드 $g 시작 (/dev/shm $(shm_use) %, 부하 $(cut -d' ' -f1 /proc/loadavg))"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $O/$n/apps_at_end.txt; cat /proc/loadavg >> $O/$n/apps_at_end.txt
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then
    CELLW=$(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}')
    $PY exe/speed1_bbox.py $AG/wolf.pt $CELLW $O/$n/pred.pt > $O/$n/bbox.txt 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) s6 줄 $(wc -l < $O/$n/s6.jsonl 2>/dev/null) hc 줄 $(wc -l < $O/$n/hc.jsonl 2>/dev/null) 렌더 $(wc -l < $O/$n/rr.jsonl 2>/dev/null) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  if [ -f $O/$n/pred.pt ]; then                                                              # 덤프: R2 에 올리고 크기 확인 뒤 로컬 사본만 지움 (R2 에 그대로 남음)
    rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
    [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && rm -f $O/$n/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"
  fi
  for z in $O/$n/e2_f*.npz; do [ -f "$z" ] || continue                                        # 진단 D2 덤프도 R2 크기 확인 뒤 로컬 사본 지움
    [ "$(rclone size $RO/$n/$(basename $z) --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $z)" ] && rm -f $z
  done; }
SPEC=$(grep -v '^#' exe/speed10_spec.txt | awk -v s=$STG '$1==s')
[ -f exe/speed10_pre_$STG.sh ] && { log "앞 단계 스크립트 speed10_pre_$STG.sh"; bash exe/speed10_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
if echo "$SPEC" | grep -q '@rr 1'; then
  mkgt wolf $GTW --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json
  mkgt bread $GTB --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json
  mkgt wm $GTM --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --allow_nonfinite
  log "렌더 자료: ficus $FGT wolf $GTW ($(ls $GTW/meta.json 2>&1)) bread $GTB ($(ls $GTB/meta.json 2>&1)) 수박 $GTM ($(ls $GTM/meta.json 2>&1)) 모델 $WMM"
fi
# ---- 판정·비결정·진단·재현 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z|p|q) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 (렌더 포함): 계산 프로세스 0 인 카드 하나를 잡아 그 카드에 차례로 a · b (연기 단계는 a 만)
if echo "$SPEC" | awk '$4=="z"||$4=="p"||$4=="q"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG (계산 프로세스 0 인 카드 하나)"
  RS="a b"; case $STG in *k) RS="a";; esac
  for r in $RS; do
    while read -r st n s md x; do
      case $md in z|p) ;; q) [ $r = a ] || continue;; *) continue;; esac
      [ -f $O/${n}_$r/summary.json ] && continue
      log "${n}_$r 단독 카드 $ZG 다른 계산 프로세스 $(nvidia-smi -i $ZG --query-compute-apps=pid --format=csv,noheader | grep -c .) 부하 $(cut -d' ' -f1 /proc/loadavg)"
      run ${n}_$r $s $md $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 모든 판이 5 프레임 넘게 끝나야 (컴파일·훅 확인), 아니면 실패 표식 -> 다음 단계 안 함
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    x="$(dfl $s "$x")"                                                                       # 낱말 (H5 등) 펼친 글자로 검사
    case $md in z|p|q) n=${n}_a;; esac
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || { OK=0; log "[연기] $n 5 프레임 못 감/발산"; }
    grep -q "절 CM65\] speed6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM65 줄 없음"; }
    case " $x " in *" @s7"*) grep -q "절 CM72\] speed7 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM72 줄 없음"; };; esac
    case " $x " in *" @s8"*) grep -q "절 CM74\] speed8 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM74 줄 없음"; };; esac
    case " $x " in *" @s8w "*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s8w.jsonl')]; p=next((q['patched'] for q in L if 'patched' in q), []); b=next((q['bad'] for q in L if 'bad' in q), [1]); print('$n', p, b); sys.exit(0 if len(p) == (5 if ' @s9e 1' in ' $x ' else 4) else 1)" >> $O/speed10_$STG.log 2>&1 && grep -q "절 CM74\] 워프 모으기 W 켬" $O/$n/log.txt || { OK=0; log "[연기] $n 워프 모으기 고친 커널 4 개 아님/줄 없음"; };; esac
    case " $x " in *" @s8w 2"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s8w.jsonl')]; C=[q['chk'] for q in L if 'chk' in q]; n=sum(c['Aacc']['n'] for c in C); print('$n chk', len(C), n); sys.exit(0 if len(C) >= 5 and n > 0 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n 확인 판 셈 없음"; };; esac
    case " $x " in *" @s9"*) grep -q "절 CM76\] speed9 켬" $O/$n/log.txt && $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s9.jsonl')]; p=next((q['patched'] for q in L if 'patched' in q), []); print('$n s9', p); sys.exit(0 if len(p) == 3 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n CM76 줄·고친 커널 3 개 없음"; };; esac
    case " $x " in *" @s10n "*|*" @s10cap "*) grep -q "절 CM78\] speed10 켬" $O/$n/log.txt && $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s10.jsonl')]; t=next((q['tx'] for q in L if 'tx' in q), None); print('$n s10', len(L), t, next((q['skip'] for q in L if 'skip' in q), None), L[-1]); sys.exit(0 if len(L) >= 5 and (' @s10n ' not in ' $x ' or t) else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n CM78 줄·s10.jsonl·고친 커널 없음"; };; esac
    case " $x " in *" @s10n 2"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s10.jsonl')]; c=[q['chk'] for q in L if q.get('chk')]; n=sum(v[0] for d in c for v in d.values()); print('$n s10 확인', n, sum(v[1] for d in c for v in d.values())); sys.exit(0 if n > 0 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n s10 확인 셈 없음"; };; esac
    case " $x " in *" @s9r 1"*) grep -q "절 CM76\] 이어 하기 경계" $O/$n/log.txt || { OK=0; log "[연기] $n 이어 하기 경계 줄 없음"; };; esac
    case " $x " in *" @s9c 2"*|*" @s9dg 1"*) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/s9.jsonl')]; c=[q for q in L if 'chk' in q or 'bar' in q]; print('$n s9 셈', len(c), c[-1] if c else None); sys.exit(0 if len(c) >= 5 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n s9 셈 없음"; };; esac
    case " $x " in *" @s7z "*) [ -f $O/$n/s7z.jsonl ] && grep -q "절 CM72\] 지연 백업 Z 켬" $O/$n/log.txt || { OK=0; log "[연기] $n 지연 백업 줄·s7z.jsonl 없음"; };; esac
    case " $x " in *" @s7p "*) [ -s $O/$n/s7p.jsonl ] && grep -q "절 CM72\] 장벽 거르기 P 켬" $O/$n/log.txt && head -1 $O/$n/s7p.jsonl | grep -q '"fnwrap": true' || { OK=0; log "[연기] $n 장벽 거르기 줄·s7p.jsonl·fnwrap 없음 $(head -c 400 $O/$n/s7p.jsonl)"; };; esac
    case " $x " in *" @s7kp "*) [ -s $O/$n/s7trace.json.gz ] || { OK=0; log "[연기] $n s7trace 없음 $(head -c 600 $O/$n/s7kp_err.txt 2>/dev/null)"; };; esac
    case " $x " in *" @mf2 1"*|*" Gpp "*|*" Hc "*) grep -q "절 CM67\] mfix 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM67 mfix 줄 없음"; };; esac
    [ "$(wc -l < $O/$n/s6.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n s6.jsonl 없음"; }
    case " $x " in *" @s6prof "*) [ -s $O/$n/s6prof.json ] || { OK=0; log "[연기] $n s6prof.json 없음 $(head -c 600 $O/$n/s6prof_err.txt 2>/dev/null)"; };; esac
    case " $x " in *" @s6l 1 "*) grep -q "절 CM65\] 실행기 사본" $O/$n/log.txt || { OK=0; log "[연기] $n 실행기 사본 줄 없음"; }
        case " $x " in *" Gp "*) grep -q "절 CM57\] ghost penalty 6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM57 ghost6 줄 없음"; };; esac
        case " $x " in *" @rr 1"*|*" Gp "*) [ "$(wc -l < $O/$n/rr.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n rr.jsonl 없음"; };; esac
        case "$s $x " in M3*" Gp "*|L*" Gp "*|W*" Gp "*) $PY -c "import json,sys; i=json.load(open('$O/$n/hcinfo.json')); print('$n', i); sys.exit(0 if i['nhc'] >= 1 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n hcinfo 없음"; };; esac
        continue;; esac   # 가벼운 판은 hc.jsonl 기록 없음 (아래 hc 검사 안 함)
    case $s in F|B) [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc 오류"; };; *) case " $x " in *" @hc "*|*" G "*|*" Gp "*) [ -s $O/$n/hc.jsonl ] && [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc.jsonl 없음/오류"; };; esac;; esac   # ficus·bread: 장벽 없음 -> 표시 커널 안 돎 (버퍼·기록 없음이 정상)
    case " $x " in *" @hc "*|*" G "*|*" Gp "*) ;; *) continue;; esac
    case $s in M3|L|W) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/hc.jsonl')]; i=json.load(open('$O/$n/hcinfo.json')); print('$n', i, sum(q['nstep'] for q in L), sum(q['ncol'] for q in L), sum(q['new'] for q in L), sum(q['napp'] for q in L)); sys.exit(0 if i['nbar'] >= 1 and i['nhc'] >= 1 and sum(q['nstep'] for q in L) > 0 else 1)" >> $O/speed10_$STG.log 2>&1 || { OK=0; log "[연기] $n 장벽 씬인데 표시 커널이 안 돎"; };; esac
    case " $x " in *" @g4 "*|*" P1X "*|*" G "*|*" Gp "*) grep -q "절 CM57\] ghost penalty 6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM57 ghost6 줄 없음"; };; esac
    case " $x " in *" @rr 1"*|*" G "*|*" Gp "*) [ "$(wc -l < $O/$n/rr.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n rr.jsonl 없음"; };; esac
    case $md in g|nd) [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n 진단 e1 없음/오류"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/speed10.FAIL; rclone copy $O $RO -q --include speed10.FAIL; };;
esac
$PY exe/speed10_tab.py $O > $O/speed10_table.md 2> $O/speed10_table.err; rclone copy $O $RO -q --include "speed10_table.*"
log "단계 끝 $STG"
