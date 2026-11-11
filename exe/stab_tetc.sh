#!/bin/bash
# 절 CM106 묶음 tetc (exe/stab_tet2.sh 복사): 'TC' = T9 (절 CM102, 사면체 기저, B-스플라인 전용 뺌, 벌점·장벽 없음) + 사면체 단위 질량·운동량 (노드 ¼):
#   새 실행기 --gq_mass + --gq_oct 0 (κ 0; T9 의 0.1 대신), 수박 (옛 실행기) 은 H9 글자의 --gq_mass --gq_oct 0 을 그대로 두고 ρ 0.8·--sp2 23 (T9 수박과 같게);
#   가장 바깥 exe/tetc_wrap.py '--tcm 1' (정의 확인 + 기록 tetc.jsonl: 보존 합·노드 질량 대 강성), '@tcm 0' = 기록만 (T9S 대조).
# 덤프·판 디렉토리는 /home/dkta/work/tetc_dump (판정 판은 지표 뒤 덤프 지움, 영상 판 덤프는 남기고 R2 에도), 단계 로그·코드는 /dev/shm/dkta.
# 이하 exe/stab_tet2.sh 줄 (장벽 '@t2' 갈래는 이 묶음에서 안 씀).
O=$1; RO=$2; STG=$3; mkdir -p $O
NOSHM=1; O=/home/dkta/work/tetc_dump; mkdir -p $O   # 절 CM106 (사용자 11-10): 판 디렉토리·덤프는 /home/dkta/work/tetc_dump, 판정 판 덤프는 지표 뒤 지움, 영상 판 덤프는 남김, 올리기는 뒤에서
PY=/home/dkta/.conda/envs/af/bin/python
[ "$STG" = en ] && { bash exe/tetc_en.sh $RO; exit $?; }   # 절 CM106.7: 에너지 색 렌더 단계 (실행기·판 설정 안 부름)
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
dfl() { local s=$1 x=" $2 "
  local W8="@w5s 1 @w5c 1 @w8k 1 @w8e 1 @w8d 2 @w8t 1"                                       # 절 CM83: H6 = H5 + S_구속·C + (K) + T1E2 (절 CM81 T1E2 판 글자 그대로), H6np = H6 − 가벼움 벌점 × κ (KAP)
  x="${x// H8N / H8 @w11 1 }"; x="${x// H5KN / H5n @w8k 1 @w11 1 }"                             # 절 CM91: N = 노드별 뒤집힘 보폭 (옛 --ws_bar 4 --ws_fi 정의), H5KN = H5n + P1X 강성 × κ (확장 없음) + N
  x="${x// H9a / H8 @w9 1 }"; x="${x// H9b / H8 @w9 2 }"                                          # 절 CM87
  x="${x// TC / H9 @tet 1 @tnob 1 @tcm 1 }"; x="${x// X9 / H9 @tet 0 @tnob 1 }"; x="${x// T9S / H9 @tet 1 @tnob 1 @tsp 1 }"; x="${x// T9 / H9 @tet 1 @tnob 1 }"; x="${x// H9nb / H9 @tnob 1 }"; x="${x// H9 / H8 @s13x 1 }"   # 절 CM102: T9 = H9 − B-스플라인 전용 + 사면체 기저, T9S = T9 + 희소 사면체 처리; H9 = H8 + X (절 CM86)
  x="${x// H6np / H5nP $W8 }"; x="${x// H6 / H5n $W8 }"; x="${x// H7np / H5nP $W8 }"; x="${x// H8 / H5n $W8 @s12z 1 }"; x="${x// H7 / H5n $W8 }"   # 절 CM84: H7 = H6 와 같은 글자, 감싸개 ext2 가 κ 로 확장을 고름
  case $s in M*) ;; *) [[ "$x" == *" H4n "* || "$x" == *" H5n"* ]] && x="$x --cap_mul 4";; esac  # 새 실행기 첫 할당 배수 4 (절 CM72 H2) -- 속도 판에만 (Gk = 절 CM75 판은 없음)
  local SPD="@s6o 1 @s6b 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s9c 1 @s9y 1" KAP="@l4 1 @l3 10 @l3sc 1 @l3fr 1"   # 절 CM81: H5 = H4 + C·Y; R = 벽시계 예산 Rw '@s6rw 1', X = X3g '@s8g 1e6'
  x="${x// H5nP / Gpp $SPD @s6rw 1 @s8g 1e6 }"; x="${x// H5n / Gpp $SPD @s6rw 1 @s8g 1e6 $KAP }"; x="${x// H5nR / Gpp $SPD @s8g 1e6 $KAP }"; x="${x// H5nX / Gpp $SPD @s6rw 1 $KAP }"; x="${x// H5nRX / Gpp $SPD $KAP }"
  x="${x// Gk / Gpp $KAP }"                                                                    # 절 CM75 판 (G'' + 가벼움 벌점 × κ, 속도 판 없음)
  x="${x// H4n / Gpp @s6o 1 @s6b 1 @s6rw 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s8g 1e6 @l4 1 @l3 10 @l3sc 1 @l3fr 1 }"; x="${x// Gpp / Gp @mf2 1 }"; x="${x// Gp / F P1X A3 @hc 5 @rr 1 @tc 8 }"
  x="${x// P1X / @g4 1e-1 @g4t 1 @g4x 1 }"; x="${x// A3 / @s4w 1 @s4y 1 @s4x 2 }"; x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  case "$x " in *" @noext 1 "*) x="${x/@w8e 1 @w8d 2 @w8t 1/}"; x="${x// @noext 1 / }";; esac                         # 절 CM93 (b): 확장만 뺌 (P1X 강성 × κ 는 그대로)
  case "$x " in *" @nosc 1 "*) x="${x/@w5s 1 @w5c 1/}"; x="${x// @nosc 1 / }";; esac                                     # 절 CM93 (d): S_구속·C 만 뺌
  case "$x " in *" @tnob 1 "*) x="${x// @g4 1e-1 @g4t 1 @g4x 1 / }"; x="${x// @l4 1 @l3 10 @l3sc 1 @l3fr 1 / }"; x="${x// @w8k 1 / }"; x="${x// @w8e 1 @w8d 2 @w8t 1 / }"; x="${x// @tnob 1 / }";; esac   # 절 CM102: B-스플라인 전용 뺌 (ghost penalty P1X · 가벼움 벌점 · (K) · 확장)
  x="${x// D / $DFL }"; case "$x " in *" @nxit 1 "*) x="${x/--xit_from 2/}"; x="${x/--xg 1 / }"; x="${x// @nxit 1 / }";; esac   # 절 CM90 수렴 판: 반복 xit 문과 그 위 그래프 나눔 끔
  case "$x " in *" @nks 1 "*) x="${x/--ksplit 3 --ks_big/}"; x="${x// @nks 1 / }";; esac   # 절 CM87 s3 진단: 걸음 쪼개기 끔 (--K 2 고정은 --ksplit 이 K 1 을 요구)
  case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac
  x="${x// @tsp 1 / }"; echo "$x"; }   # 절 CM102 T9S: 점유 가중 κ 는 run 에서 실행기 글자 (--gq_oct) 를 바꿈
dg_of() { local s=$1 md=$2 x
  case $s in M*) case $md in g|r|z) echo "--erec";; v|d) echo "--erec --dump --dump_lj";; esac; return;; esac   # 절 CM102 d = 덤프 판 (영상 안 만듦)   # v = 영상 판 (exe/stab_ext2v.sh 와 같은 글자)
  case $md in h) echo "--det --fhash"; return;; v|d) case $s in W) echo "--hgt --fhash --bar_cnt --dump --dump_F --dump_part";; *) echo "--dump --dump_F --dump_part";; esac; return;; esac
  case $md in nd) case $s in L) x="--hgt --pdiag --qjd --bar_cnt";; W) x="--hgt --fhash --bar_cnt --dump --dump_part";; F) x="--pdiag --qjd --fhash --dump";; B) x="--pdiag --qjd --fhash";; esac;; *) x="";; esac
  echo "$x"; }
W=/home/dkta/work
RD=/dev/shm/dkta/tetc; mkdir -p $RD                                                          # 이 묶음이 만드는 정답 궤적 (렌더 카메라·0 프레임) 자리 (t2chain 것이 있으면 읽기만)
gtd() { local n=$1 d; for d in /dev/shm/dkta/ag_data/gt_$n /dev/shm/dkta/t2chain/gt_$n /dev/shm/dkta/rtrender1/gt_$n /dev/shm/dkta/hhtc/gt_$n /dev/shm/dkta/hhtc2/gt_$n /dev/shm/dkta/lightg3/gt_$n /dev/shm/dkta/gpw/gt_$n /dev/shm/dkta/gpw2/gt_$n $RD/gt_$n; do [ -f $d/meta.json ] && [ -f $d/sim_0000000000.h5 ] && { echo $d; return; }; done; echo $RD/gt_$n; }
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
vid() {   # 영상 판 뒤 3DGS 렌더 (같은 카드): 새 실행기 씬 = exe/lb_render.py (exe/fem_cm4.sh·stab_wolfh3_vid.sh 와 같은 절차), 수박 = 일반 + 손상 색 (logJp viridis −0.2~0.4; exe/stab_wmK.sh vid 와 같은 절차). h5 는 결과 디렉토리 밖 /dev/shm 에 두고 렌더 뒤 지움
  local n=$1 s=$2 g=$3 Od=$O/$1 V=$O/$1/vj HD=/dev/shm/dkta/tetc_vid/$1 CD=$(pwd -P) GTD CFG SH
  mkdir -p $V $HD
  case $s in
  M3)
    $PY - $Od/pred.pt $HD/pred_v.pt <<'PYX' > $V/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
    $PY -u exe/lb_roll_to_h5.py --dump $HD/pred_v.pt --traj $GTM --out $HD/h5_fem > $V/h5.log 2>&1; rm -f $HD/pred_v.pt
    $PY exe/wmdmg_stats.py --run $Od --sc_lj $HD/sc_lj > $V/sc.log 2>&1
    (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $WCF --sim $HD/h5_fem --out $V/plain.mp4 < /dev/null > $V/r_plain.log 2>&1)
    (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $WCF --sim $HD/h5_fem --scalar_dir $HD/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis --out $V/lj.mp4 < /dev/null > $V/r_lj.log 2>&1)
    log "[영상] $n $(tail -1 $V/pad.log) plain $(stat -c %s $V/plain.mp4 2>&1) lj $(stat -c %s $V/lj.mp4 2>&1) $(tail -1 $V/r_plain.log | cut -c1-160)";;
  *)
    case $s in L) GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; SH=lego;; F) GTD=$FGT; CFG=$FGT/config.json; SH=ficus;; W) GTD=$GTW; CFG=$GTW/config.json; SH=wolf;; B) GTD=$GTB; CFG=$GTB/config.json; SH=bread;; esac
    $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $HD/h5 > $V/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $CFG --shape $SH --out $V/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
    log "[영상] $n $(stat -c %s $V/fem_$SH.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160) $(tail -1 $V/h5.log | cut -c1-160)";;
  esac
  rm -rf $HD; }
shm_use() { df --output=pcent /dev/shm | tail -1 | tr -dc '0-9'; }
LG=/dev/shm/dkta/lbours/tetc; mkdir -p $LG
log() { echo "[$(date '+%F %T')] $*" >> $LG/tetc_$STG.log; rclone copyto $LG/tetc_$STG.log $RO/tetc_$STG.log -q; }
homeok() { [ $(du -s -BG $O 2>/dev/null | cut -f1 | tr -dc 0-9) -lt 90 ] && [ $(df -BG --output=avail /home | tail -1 | tr -dc 0-9) -gt 40 ]; }
shm_wait() { [ $NOSHM = 1 ] && { while ! homeok; do log "/home 덤프 $(du -sh $O | cut -f1) · 여유 $(df -h --output=avail /home | tail -1) -- 기다림"; sleep 60; done; return; }; while [ "$(shm_use)" -ge 65 ]; do log "/dev/shm 사용률 $(shm_use) % -- 기다림"; sleep 120; done; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b w11="" pre="" wo="" w2="" w3="" w5="" wg="" hc="" rr="" w4="" tc="" wf="" mf="" l3="" l4="" w8="" s5="" s7="" s8="" s9="" lean=0 s12="" w9="" fiv="" w12="" w13="" gpl=""
  [ -f $O/$n/summary.json ] && return
  hc="$(echo " $x " | grep -o '@hc [0-9]*' | awk '{print $2}')"; tc="$(echo " $x " | grep -o '@tc [0-9]*' | awk '{print $2}')"; rr="$(echo " $x " | grep -o '@rr [0-9]*' | awk '{print $2}')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  local w5x="$(echo " $x " | grep -o '@w5[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"           # 절 CM70 S_구속 · C (exe/wolfh5_wrap.py)
  w9="$(echo " $x " | grep -o '@w9[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w11="$(echo " $x " | grep -o '@w11[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$w11" ] && case $md in g|nd|v) w11="$w11 --w11dg 1";; esac   # 절 CM89 (exe/wolfh11_wrap.py; 진단 판은 w11.jsonl)      # 절 CM87 (exe/wolfh9_wrap.py)
  local tcm=""; tcm="$(echo " $x " | grep -o '@tcm [0-9]*' | awk '{print $2}')"   # 절 CM106 (exe/tetc_wrap.py)
  local t2=""; t2="$(echo " $x " | grep -o '@t2 [0-9]*' | awk '{print $2}')"   # 절 CM103 (exe/tet2_wrap.py)
  local tet="" s13=""; tet="$(echo " $x " | grep -o '@tet [0-9]*' | sed 's/^@/--/' | tr '\n' ' ')"; s13="$(echo " $x " | grep -o '@s13[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  gpl="$(echo " $x " | grep -o '@gpl [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$gpl" ] && case $md in g|nd|v) gpl="$gpl --gpldg 1";; esac   # 절 CM100 (exe/gpl_wrap.py)
  w13="$(echo " $x " | grep -o '@w13[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"   # 절 CM94 (exe/wolfh13_wrap.py)
  w12="$(echo " $x " | grep -o '@w12[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$w12" ] && case $md in g|nd|v) w12="$w12 --w12dg 1";; esac   # 절 CM93 (exe/wolfh13_wrap.py)
  fiv="$(echo " $x " | grep -o '@fi[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$fiv" ] && case $md in g|nd|v) fiv="$fiv --fidg 1";; esac   # 절 CM92 (exe/finterp_wrap.py)
  s12="$(echo " $x " | grep -o '@s12[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"   # 절 CM85 시작점 기준 고침·진단 (exe/speed12_wrap.py)
  w8="$(echo " $x " | grep -o '@\(w8\|x1\)[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"            # 절 CM81 확장 B-스플라인·(K) + 절 CM83 검사 '@x1ck' 
  s9="$(echo " $x " | grep -o '@s9[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"                # 절 CM76 C·Y
  s5="$(echo " $x " | grep -o '@s6[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s7="$(echo " $x " | grep -o '@s7[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s8="$(echo " $x " | grep -o '@s8[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; case " $x " in *" @s6l 1 "*) lean=1;; esac
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@\(tcm\|t2\|t3u\|tet\|s13\|gpl\|gphi\|gpw\|w13\|w12\|fi\|w11\|w9\|s12\|e[235]\|g[456]\|hc\|rr\|s4\|s6\|s7\|s8\|s9\|tc\|wf\|mf\|l3\|l4\|w5\|w8\|x1\)' | sed 's/^@/--/' | tr '\n' ' ')"; wf="$(echo " $x " | grep -o '@wf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; mf="$(echo " $x " | grep -o '@mf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l3="$(echo " $x " | grep -o '@l3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l4="$(echo " $x " | grep -o '@l4[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g[456][a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E5_DG=1"                                                 # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in g|nd) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; z) pre="AF_E1_DG=1";; esac
  [ $lean = 1 ] && { pre=""; c="${c/exe\/fem_an_rt_sim.py/exe/s6/fem_an_rt_sim.py}"; c="${c/exe\/fem_ao_rt_sim.py/exe/s6/fem_ao_rt_sim.py}"; }   # 절 CM65 가벼운 판: 실행기 사본 + 진단 끔 (절 CM74 speed8 과 같음)           # 단독 판도 E1 (빠른 노드·이탈, 프레임 사이 읽기만; 절 CM56 단독 판과 같음)                                   # 절 CM42 진단 (결과 불변; 재현·단독 판은 끔)
  local hx=""; [ -n "$hc" ] && hx="--hc $hc"; [ -n "$tc" ] && hx="$hx --tc $tc"
  [ "${rr:-0}" = 1 ] && hx="$hx $(rr_of $s) $w4"
  [[ " $5 " == *" T9S "* ]] && c="${c/--gq_oct 0.1 /--gq_oct 1 }"                                # 절 CM102 T9S 새 실행기: 점유 가중 κ 0.1 -> 1
  local toct=""
  if [ $s = M3 ] && [[ " $5 " == *" T9 "* || " $5 " == *" T9S "* || " $5 " == *" X9 "* ]]; then   # X9 = T9 에서 --tet 0 (이차 B-스플라인) 만 다름                                       # 절 CM102 (조정자 11-10): 수박 사면체 판 = 입자 -> 자기 사면체 λ 4 노드 질량·운동량 (--gq_mass 끔), HHT ρ 0.8 (다른 씬과 같게),
    c="${c/ --gq_mass / }"; c="${c/ --gq_oct 0 / }"; c="${c/--rho 0.7 /--rho 0.8 }"; c="${c/--sp2 31/--sp2 23}"   # --sp2 비트 8 (점->노드 칸 모으기, 결과 불변 가속) 은 --gq_mass 판 전용 -> 뺌      #   점유 가중 κ 는 tet1_wrap --toct (옛 실행기는 --gq_mass 없이 --gq_oct 경로가 없음): T9 0.1 · T9S 1 (새 실행기와 같은 값)
    toct="--toct 0.1"; [[ " $5 " == *" T9S "* ]] && toct="--toct 1"
  fi
  if [[ " $5 " == *" TC "* ]]; then                                                            # 절 CM106 TC: 사면체 단위 질량·운동량 (노드 ¼) = --gq_mass + κ 0
    if [ $s = M3 ]; then c="${c/--rho 0.7 /--rho 0.8 }"; c="${c/--sp2 31/--sp2 23}"            # 수박: H9 글자에 --gq_mass --gq_oct 0 이 이미 있음, ρ·sp2 는 T9 수박과 같게
    else c="${c/--gq_oct 0.1 /--gq_oct 0 } --gq_mass"; fi
  fi   # 절 CM102 T9S: 점유 가중 κ (새 실행기 0.1 · 수박 0) -> 1 (칸 입자 부피·질량을 6 사면체에 고르게)
  c="exe/speed13_wrap.py $s13 exe/speed12_wrap.py $s12 exe/ext2_wrap.py $w8 $l4 $l3 $w5x $s9 $s8 $s7 $s5 $mf $wf $hx $wg $w5 $w3 $w2 $wo $c"; [ -n "$tet" ] && c="exe/tet1_wrap.py $tet $toct $c"
  local tcdg=1; [[ " $5 " == *" @tcdg 2 "* ]] && tcdg=2                                   # 절 CM106.D 발산 노드 진단 (결과 불변)
  [ -n "$tcm" ] && c="exe/tetc_wrap.py --tcm $tcm --tcdg $tcdg $c"
  if [ -n "$t2" ]; then local dg_=1; [ $md = z ] && dg_=0                                       # 절 CM103: 단독 (시간) 판은 기록 끔 (프레임마다 동기화 없음)
    local tb_
    local GD=$O; [ -s $GD/gtdf_$s.json ] || GD=/dev/shm/dkta/lbours/tet2                     # 절 CM103 s1e: O 가 /home 이면 띠·범위는 /dev/shm 묶음 디렉토리에서
    tb_=$(O=$GD band_of $s) || { log "$n 띠 없음"; return; }
    local t3_=""; [ -s $GD/gtdf_$s.json ] && t3_="--t3b $($PY -c "import json; b=json.load(open('$GD/gtdf_$s.json'))['band3']; print('%.6e,%.6e' % (b['slo'], b['shi']))")"
    [[ " $5 " == *" @t3u 1 "* ]] && t3_="--t3b $($PY -c "import json; b=json.load(open('$GD/gtdfu_$s.json'))['band3u']; print('%.6e,%.6e' % (b['slo'], b['shi']))")"   # 판 u: 상한만   # 절 CM103 판 r: 정답 프레임 간 ΔF 범위 (단계 g2)
    [ "$t2" = 3 ] && [ -z "$t3_" ] && { log "$n ΔF 범위 없음 (단계 g2 먼저)"; return; }
    c="exe/tet2_wrap.py --t2 $t2 --t2b $tb_ $t3_ --t2dg $dg_ $c"; fi
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
  [ -s $O/$n/t2.jsonl ] && $PY -c "import json,sys; sys.path.insert(0,'lib'); from anchorflow import fem_tet2 as T; json.dump(T.summarize('$O/$n/t2.jsonl'), open('$O/$n/t2sum.json','w'), indent=1)" > $O/$n/t2sum.err 2>&1   # 절 CM103 기록 요약
  [ $md = v ] && [ -f $O/$n/pred.pt ] && vid $n $s $g
  if [ $NOSHM = 1 ] && [ -f $O/$n/pred.pt ]; then                                             # 절 CM106: 지표 (영상 판 포함), 판정 판만 덤프 지움
    case $s in W) HEAP_ROWS=$O/$n/heap_rows.json $PY exe/wolfh10_heap.py $AG/wolf.pt "$n=$O/$n" > $O/$n/heap.md 2> $O/$n/heap.err;;
               M3) TET1_WM_ROWS=$O/$n/wm_rows.json $PY exe/tet1_wm.py $WMD/wm.pt "$n=$O/$n" > $O/$n/wm.md 2> $O/$n/wm.err;; esac
    [ $md != v ] && rm -f $O/$n/pred.pt $O/$n/lj.pt && log "$n 지표 뒤 덤프 지움"; fi                                         # 영상 판: 덤프 렌더 (pred.pt 를 지우기 전; exe/stab_ext2v.sh 와 같음)
  [ $md != z ] && [ "$(cat $CL/$g 2>/dev/null)" = "$BASHPID" ] && rm -f $CL/$g && log "$n 카드 $g 놓음 (계산·렌더 끝, 올리기는 카드 없이)"   # 절 CM102: 덤프 올리기 (수박 6.6 GB) 동안 카드를 잡지 않음
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) hc 줄 $(wc -l < $O/$n/hc.jsonl 2>/dev/null) 렌더 $(wc -l < $O/$n/rr.jsonl 2>/dev/null) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  touch $O/$n/.upl                                                                           # 절 CM102: 올리기는 판 subshell 밖 (카드·동시 판 수 안 잡음), 단계 끝에 .upl 없어질 때까지 기다림
  ( trap "rm -f $O/$n/.upl" EXIT
    up $n
    if [ -f $O/$n/pred.pt ]; then                                                              # 덤프: R2 에 올리고 크기 확인 뒤 로컬 사본만 지움 (R2 에 그대로 남음)
      rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
      [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && log "$n pred.pt R2 확인 (로컬 /home 사본 그대로)"
      [ -f $O/$n/lj.pt ] && rclone copyto $O/$n/lj.pt $RO/$n/lj.pt -q
    fi
    for z in $O/$n/e2_f*.npz; do [ -f "$z" ] || continue                                        # 진단 D2 덤프도 R2 크기 확인 뒤 로컬 사본 지움
      [ "$(rclone size $RO/$n/$(basename $z) --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $z)" ] && rm -f $z
    done
  ) > /dev/null 2>&1 < /dev/null & disown; }
band_of() { local f=$O/gtsv_$1.json; [ -s $f ] || return 1; $PY -c "import json; b=json.load(open('$f'))['band']; print('%.6e,%.6e,%.6e,%.6e,%.6e' % (b['slo'], b['shi'], b['dlo'], b['dhi'], b['ja']))"; }
GX=""; [ $STG = g3 ] && { GX=u; O=/dev/shm/dkta/lbours/tet2; }
if [ $STG = g2 ] || [ $STG = g3 ]; then                                                                        # 절 CM103 단계 g2: 정답 프레임 간 ΔF 특이값 -> 판 r 범위 (씬마다 빈 카드, 동시)
  for sc in L W M3 F B; do
    ( me=$BASHPID; g=$(claim_wait $me)
      case $sc in L) D_=$SD/L000.pt; c_=1; n_=80; x_="";; W) D_=$AG/wolf.pt; c_=2; n_=50; x_="";; M3) D_=$WMD/wm.pt; c_=3; n_=100; x_="";; F) D_=$FI; c_=1; n_=125; x_="--region_density";; B) D_=$AG/bread.pt; c_=1; n_=300; x_="";; esac
      CUDA_VISIBLE_DEVICES=$g $PY -u exe/tet2_gtdf.py --data $D_ --cell $c_ --frames $n_ --tag $sc $x_ --out $O/gtdf${GX}_$sc.json < /dev/null > $O/gtdf${GX}_$sc.log 2>&1
      log "gtdf$GX $sc rc=$? $(grep -a '정답 프레임 간\|상한만' $O/gtdf${GX}_$sc.log | tr '\n' ' ')"
      rclone copy $O $RO -q --include "gtdf${GX}_$sc.*"
      [ "$(cat $CL/$g 2>/dev/null)" = "$me" ] && rm -f $CL/$g ) & sleep 3
  done
  wait; log "단계 끝 $STG"; exit 0
fi
if [ $STG = g ]; then                                                                         # 절 CM103 단계 g: 정답 사면체 F 특이값 범위 -> 띠 (씬마다 빈 카드 하나, 동시)
  for sc in L W M3 F B; do
    ( me=$BASHPID; g=$(claim_wait $me)
      case $sc in L) D_=$SD/L000.pt; c_=1; n_=80; x_="";; W) D_=$AG/wolf.pt; c_=2; n_=50; x_="";; M3) D_=$WMD/wm.pt; c_=3; n_=100; x_="";; F) D_=$FI; c_=1; n_=125; x_="--region_density";; B) D_=$AG/bread.pt; c_=1; n_=300; x_="";; esac
      log "gtsv $sc 카드 $g 시작"
      CUDA_VISIBLE_DEVICES=$g $PY -u exe/tet2_gtsv.py --data $D_ --cell $c_ --frames $n_ --tag $sc $x_ --out $O/gtsv_$sc.json < /dev/null > $O/gtsv_$sc.log 2>&1
      log "gtsv $sc rc=$? $(grep -a '정답 사면체' $O/gtsv_$sc.log | tail -1)"
      rclone copy $O $RO -q --include "gtsv_$sc.*"
      [ "$(cat $CL/$g 2>/dev/null)" = "$me" ] && rm -f $CL/$g ) & sleep 3
  done
  wait
  log "단계 끝 g"; exit 0
fi
SPEC=$(grep -v '^#' exe/tetc_spec.txt | awk -v s=$STG '$1==s')

if echo "$SPEC" | grep -q '@rr 1' || [ $STG = v ]; then
  mkgt wolf $GTW --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json
  mkgt bread $GTB --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json
  mkgt wm $GTM --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --allow_nonfinite
  log "렌더 자료: ficus $FGT wolf $GTW ($(ls $GTW/meta.json 2>&1)) bread $GTB ($(ls $GTB/meta.json 2>&1)) 수박 $GTM ($(ls $GTM/meta.json 2>&1)) 모델 $WMM"
fi
[ $STG = u1 ] || for d_ in $O/*/; do n_=$(basename $d_); [ -f $d_/summary.json ] && [ -f $d_/pred.pt ] && [ ! -f $d_/.upl ] || continue   # 절 CM103: 올리는 중 (.upl) 은 건너뜀, u1 단계는 하나씩 (앞 단계 스크립트)    # 절 CM102: 앞 단계가 끊겨 R2 확인 전에 남은 덤프 (카드 없이 뒤에서)
  touch $d_/.upl; ( trap "rm -f $d_/.upl" EXIT; up $n_; rclone copyto $d_/pred.pt $RO/$n_/pred.pt -q; [ "$(rclone size $RO/$n_/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $d_/pred.pt)" ] && rm -f $d_/pred.pt && log "$n_ 남은 pred.pt R2 확인 뒤 로컬 사본 지움" ) > /dev/null 2>&1 < /dev/null & disown
done
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in z) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; [ "$(cat $CL/$g 2>/dev/null)" = "$me" ] && rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
[ $NOSHM = 1 ] || while ls $O/*/.upl > /dev/null 2>&1; do sleep 20; done                                       # 절 CM102: 뒤에서 올리는 덤프 끝까지
log "판정 판 끝 $STG"
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
    case " $x " in *" @w8e 1"*) grep -q "절 CM84\] ext2 켬" $O/$n/log.txt && grep -q "절 CM84 ext2\] .* κ" $O/$n/log.txt && $PY -c "import json,sys,os; x2=json.load(open('$O/$n/x2info.json')); i=json.load(open('$O/$n/x1info.json')); w=json.load(open('$O/$n/w8info.json')) if os.path.exists('$O/$n/w8info.json') else {}; d=x2['dec']; print('$n x2', [(q.get('solver'), q.get('name'), q.get('kappa'), q.get('ext')) for q in d], 'alloc ext/base', x2.get('nalloc_ext'), x2.get('nalloc_base'), 'x1', {k: i.get(k) for k in ('nalloc','nclass','nfold','nexp','ngg','njz','nlog','nrows','err')}, 'w8', {k: w.get(k) for k in ('nalloc','nclass','nfold','nexp','ngg','err')}); W='$s' == 'W'; e=any(q['ext'] for q in d); z=lambda D, ks: all(int(D.get(k) or 0) == 0 for k in ks); L=[json.loads(l) for l in open('$O/$n/w8.jsonl')] if os.path.exists('$O/$n/w8.jsonl') else []; c=(e and w.get('nclass', 0) >= 1 and w.get('nfold', 0) >= 1 and w.get('nexp', 0) >= 1 and w.get('ngg', 0) >= 1 and i.get('nrows', 0) > 0 and max(q['outer'] for q in L) > 0 and not w.get('err')) if W else (len(d) > 0 and not e and z(i, ('nalloc','nclass','nfold','nexp','ngg','njz','nlog','nrows')) and z(w, ('nalloc','nclass','nfold','nexp','ngg')) and not L); sys.exit(0 if c and not i.get('err') else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM84 줄·x2info 결정 (wolf 확장 켬·나머지 커널 0) 아님"; };; esac
    case "$s $x " in M*" @s12"*) grep -q "절 CM85\] speed12 켬" $O/$n/log.txt && grep -q "절 CM85\] 시작점 기준 패치 \[" $O/$n/log.txt && [ ! -s $O/$n/s12_err.txt ] || { OK=0; log "[연기] $n CM85 줄 없음/s12_err"; };; esac
    case " $x " in *" @w12 "*) case $s in W) grep -q "절 CM93\] wolfh12 켬" $O/$n/log.txt && $PY -c "import json,sys,os; i=json.load(open('$O/$n/w12info.json')); print('$n w12', i); L=[json.loads(l) for l in open('$O/$n/w12.jsonl')] if os.path.exists('$O/$n/w12.jsonl') else []; print('$n w12.jsonl', len(L), L[-1] if L else None); sys.exit(0 if i['nsrc'] >= 1 and i['nalloc'] >= 1 and i['ncall'] > 0 and not i.get('err') and (len(L) >= 5 or not i['dg']) else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM93 줄·w12info·w12.jsonl 없음/오류"; };; esac;; esac
    case " $x " in *" @tet 1"*) grep -q "절 CM102\] tet1 켬 -- 모드 1" $O/$n/log.txt && $PY -c "import json,sys; i=json.load(open('$O/$n/tet1info.json')); R=[m for m in i['mods'] if m.get('rules')]; print('$n tet1', i.get('err'), [(m['n'], m['rules']) for m in R], i.get('tables')); ok = not i.get('err') and len(R) >= 1 and any('bwf' in m['rules'] for m in R) and any(k.startswith('rt_gauss') for m in R for k in m['rules']) and any(k.startswith('rt_coord') for m in R for k in m['rules']) and len(i.get('tables') or []) >= 1 and max(t['chk'] for t in i['tables']) < 1e-5; sys.exit(0 if ok else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM102 tet1 줄·tet1info (규칙·표) 없음/오류"; };; esac
    case " $x " in *" @tcm "*) grep -q "절 CM106\] tetc 켬" $O/$n/log.txt && $PY -c "import json,sys; i=json.load(open('$O/$n/tetcinfo.json')); L=[json.loads(l) for l in open('$O/$n/tetc.jsonl')]; R=[q['rat'] for q in L if 'rat' in q]; print('$n tetc', i, len(L), 'em', max([q['em'] for q in L if q.get('em') is not None] or [0]), 'eP', max([q['eP'] for q in L if q.get('eP') is not None] or [0]), 'rat', R[:1], [q.get('rat_err') for q in L if 'rat_err' in q][:1]); tc=' @tcm 1 ' in ' $x '; ok = not i.get('err') and i['n_inst'] >= 1 and len(L) >= 5 and min(q['m_n'] for q in L) > 0 and max([q['em'] for q in L if q.get('em') is not None] or [1]) < 1e-4 and max([q['eP'] for q in L if q.get('eP') is not None] or [0]) < 1e-3 and len(R) >= 1 and (not tc or (i['gqm'] and i['goct'] in (0.0, None))); sys.exit(0 if ok else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM106 줄·tetcinfo·tetc.jsonl (보존 합·기록) 없음/오류"; };; esac
    case " $x " in *" @t2 "*) $PY -c "import json,sys,os; i=json.load(open('$O/$n/tet2info.json')); m=[q for q in i['mods'] if q.get('rules')]; hk=i.get('hooks',{}); L=[json.loads(l) for l in open('$O/$n/t2.jsonl')] if os.path.exists('$O/$n/t2.jsonl') else []; print('$n tet2', i.get('err'), [q['rules'] for q in m], hk, len(L), L[-1]['steps'][-1] if L and L[-1]['steps'] else None); R=[k for q in m for k,v in q['rules'].items() if v > 0 and ':' in k]; ok = not i.get('err') and len(m) >= 1 and sum(v for k,v in hk.items() if k.startswith('iter:')) > 0 and sum(v for k,v in hk.items() if k.startswith('end:')) > 0 and sum(v for k,v in hk.items() if k.startswith('start:')) > 0 and len(L) >= 5 and (i['mode'] == 0 or len(R) >= 3); sys.exit(0 if ok else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM103 tet2info·t2.jsonl (규칙·훅·기록) 없음/오류"; };; esac
    case " $x " in *" @tet 0"*) $PY -c "import json,sys; i=json.load(open('$O/$n/tet1info.json')); print('$n tet0', [(m['n'], m['sha'], m.get('rules')) for m in i['mods']]); sys.exit(0 if i['mods'] and not any(m.get('rules') for m in i['mods']) else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n tet 0 인데 소스가 바뀜"; };; esac
    case " $x " in *" @gpl "*) grep -q "절 CM100\] gpl 켬" $O/$n/log.txt && grep -q "절 CM100 gpl\] 씬 결정" $O/$n/log.txt && $PY -c "import json,sys,os; i=json.load(open('$O/$n/gplinfo.json')); print('$n gpl', i); L=[json.loads(l) for l in open('$O/$n/gpl.jsonl')] if os.path.exists('$O/$n/gpl.jsonl') else []; print('$n gpl.jsonl', len(L), L[-1] if L else None); R=[q for q in L if 'rerr' in q]; ok=i['dec'] and i['nmul'] > 0 and i['nmul0'] > 0 and ((i.get('npre_tot') or 0) == 0 or i['mbar'] == 0.0) and len(L) >= 5 and max(q['werr'] for q in L) < 1e-5 and len(R) >= 1 and max(q['rerr'] for q in R) < 1e-4 and all(q.get('set_eq') for q in R) and L[-1]['np1x'] > 0 and not any('ref_err' in q for q in L); ok = ok and not os.path.exists('$O/$n/gpl_err.txt'); sys.exit(0 if ok else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM100 줄·gplinfo·gpl.jsonl 없음/오류 (가중 대 참값)"; };; esac
    case " $x " in *" @w13 "*) grep -q "절 CM94\] wolfh13 켬" $O/$n/log.txt && $PY -c "import json,sys; i=json.load(open('$O/$n/xpinfo.json')); print('$n xp', i); w=('$s'=='W'); ok=(i['nsrc'] >= 1 and not i.get('err')) and ((i['nalloc'] >= 1 and i['ngrad'] > 0 and i['nvc'] > 0 and i['npw0'] > 0 and (i['mode'] != 1 or (i['ncls'] > 0 and i['npwl'] + i['nbuild'] > 0)) and (i.get('dev') or {}).get('d0_mod', 0) > 0) if w else (i['nalloc'] == 0 and i['ngrad'] == 0 and i['nvc'] == 0 and i['npw0'] == 0 and i['ncls'] == 0)); sys.exit(0 if ok else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM94 줄·xpinfo 없음/오류 (wolf 는 훅 수 > 0, κ 1 씬은 0)"; };; esac
    case " $x " in *" @fi "*) grep -q "절 CM92\] finterp 켬" $O/$n/log.txt && $PY -c "import json,sys,os; i=json.load(open('$O/$n/fiinfo.json')); print('$n fi', i); L=[json.loads(l) for l in open('$O/$n/fi.jsonl')] if os.path.exists('$O/$n/fi.jsonl') else []; print('$n fi.jsonl', len(L), L[-1] if L else None); sys.exit(0 if i['nsrc'] >= 1 and i['nalloc'] >= 1 and i['nfpi'] > 0 and not i.get('err') and (len(L) >= 5 or not i['dg']) else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n CM92 줄·fiinfo·fi.jsonl 없음/오류"; };; esac
    case " $x " in *" @w11 "*) grep -q "절 CM91\] wolfh11 켬" $O/$n/log.txt && $PY -c "import json,sys; i=json.load(open('$O/$n/w11info.json')); print('$n w11', i); sys.exit(0 if i['nsrc'] >= 1 and i['nalloc'] >= 1 and i['nnewton'] > 0 and i['nstart'] > 0 and not i.get('err') else 1)" >> $O/wolfh11_$STG.log 2>&1 && [ -s $O/$n/w11.jsonl ] || { OK=0; log "[연기] $n CM89 줄·w11info (훅 수) 없음"; };; esac
    case " $x " in *" @w9 "*) grep -q "절 CM87\] wolfh9 켬" $O/$n/log.txt && grep -q "절 CM87 wolfh9\] " $O/$n/log.txt && [ -s $O/$n/w9info.json ] || { OK=0; log "[연기] $n CM87 줄·w9info 없음"; };; esac
    case " $x " in *" @w8k 1"*) grep -q "절 CM77 wolfh6\] (K) 재질" $O/$n/log.txt || { OK=0; log "[연기] $n CM77 (K) 줄 없음"; };; esac
    case " $x " in *" @s9"*) grep -q "절 CM76\] speed9 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM76 speed9 줄 없음"; };; esac
    case " $x " in *" @l4 1"*) grep -q "절 CM75\] lightg4 켬" $O/$n/log.txt && grep -q "절 CM75 lightg4\] 재질" $O/$n/log.txt && [ -s $O/$n/l4info.json ] || { OK=0; log "[연기] $n CM75 줄·l4info.json 없음"; };; esac
    case " $x " in *" @l3 "*) grep -q "절 CM73\] lightg3 켬" $O/$n/log.txt && [ ! -s $O/$n/lg3_err.txt ] || { OK=0; log "[연기] $n CM73 줄 없음/lg3_err"; };; esac
    case " $x " in *" @l3fd "*) case $s in M3|W|B) [ -s $O/$n/lg3fd.jsonl ] || { OK=0; log "[연기] $n lg3fd.jsonl 없음"; };; esac;; esac
    case " $x " in *" @l3ck 1"*) case $s in M3) grep -q "\"ck\": {" $O/$n/lg3.jsonl 2>/dev/null && ! grep -q "ck_err" $O/$n/lg3.jsonl || { OK=0; log "[연기] $n 금 면 진단 없음/오류"; };; esac;; esac
    case " $x " in *" @wfm "*) grep -q "절 CM64\] wmfloor 켬" $O/$n/log.txt && [ ! -s $O/$n/wf_err.txt ] && [ -s $O/$n/wf.jsonl ] || { OK=0; log "[연기] $n CM64 줄·wf.jsonl 없음/오류"; };; esac
    case " $x " in *" @s6l 1 "*) continue;; esac                                             # 절 CM83: 가벼운 판 (판정 수박 포함) 은 진단 기록 없음이 정상
    [ $md = z ] && continue                                                                  # 가벼운 판 단독: 진단 기록 없음이 정상 (위 훅 줄·w6info 만 봄)
    case $s in F|B) [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc 오류"; };; *) [ -s $O/$n/hc.jsonl ] && [ ! -s $O/$n/hc_err.txt ] || { OK=0; log "[연기] $n hc.jsonl 없음/오류"; };; esac   # ficus·bread: 장벽 없음 -> 표시 커널 안 돎 (버퍼·기록 없음이 정상)
    case $s in M3|L|W) $PY -c "import json,sys; L=[json.loads(l) for l in open('$O/$n/hc.jsonl')]; i=json.load(open('$O/$n/hcinfo.json')); print('$n', i, sum(q['nstep'] for q in L), sum(q['ncol'] for q in L), sum(q['new'] for q in L), sum(q['napp'] for q in L)); sys.exit(0 if i['nbar'] >= 1 and i['nhc'] >= 1 and sum(q['nstep'] for q in L) > 0 and all(len(q['mo']) == min(q['nstep'], 32) for q in L) and L[-1]['mo'][-1][0] > 0 else 1)" >> $LG/tetc_$STG.log 2>&1 || { OK=0; log "[연기] $n 장벽 씬인데 표시 커널이 안 돎"; };; esac
    case " $x " in *" @g4 "*|*" P1X "*) grep -q "절 CM57\] ghost penalty 6 켬" $O/$n/log.txt || { OK=0; log "[연기] $n CM57 ghost6 줄 없음"; };; esac
    case " $x " in *" @rr 1"*) [ "$(wc -l < $O/$n/rr.jsonl 2>/dev/null || echo 0)" -ge 5 ] || { OK=0; log "[연기] $n rr.jsonl 없음"; };; esac
    case $md in g|nd) [ -s $O/$n/e1.jsonl ] && [ ! -s $O/$n/e1_err.txt ] || { OK=0; log "[연기] $n 진단 e1 없음/오류"; };; esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $LG/tetc.FAIL; rclone copy $LG $RO -q --include tetc.FAIL; };;
esac
                                                                                            # 표는 로컬 (exe/wolfh13_tab.py, R2 내려받아)
for sc in L W; do                                                                            # 절 CM102 끈 판 = H9: 결정적 판 프레임별 입자 위치 해시 (fem.jsonl xh) -- h${sc}0 · h${sc}0b (같은 명령 두 번) · h${sc}1 (tet1_wrap --tet 0)
  [ -f $O/h${sc}1/fem.jsonl ] || continue
  $PY -c "
import json
L = lambda n: [json.loads(l).get('xh') for l in open('$O/' + n + '/fem.jsonl')]
a, b, c = L('h${sc}0'), L('h${sc}0b'), L('h${sc}1')
print('[해시 대조 $sc] 프레임', len(a), len(b), len(c), '같은 명령 두 번 같음', a == b, '끈 판 대 H9 같음', a == c, '끝', a[-1:], b[-1:], c[-1:])" >> $LG/tetc_$STG.log 2>&1
  rclone copyto $LG/tetc_$STG.log $RO/tetc_$STG.log -q
done
log "단계 끝 $STG"
