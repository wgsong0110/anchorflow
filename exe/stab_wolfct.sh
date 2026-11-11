#!/bin/bash
# 절 CM108 묶음 wolfct (exe/stab_condwhy.sh 복사; 사용자 11-11): 가장 바깥 exe/wolfct_wrap.py '@ct 1' = 모래 DP 반환 사상 일관 접선 (대칭 몫 + PSD 투영) 헤시안, '@ctfd 1' = 다시 풀기 판 가우스 점 유한 차분 대조 (lib/anchorflow/fem_wolfct.py, 결과 불변);
#   저장 상태 '@cwload 8|5' = 절 CM96 sv8·sv5, '@cwload 9' = 이 묶음 저장 판 sv9 (H8F 궤적). 판정 판 덤프 (pred.pt·e2 npz) 는 노드에서 지표 (wolf heap.md·heap_rows.json, ficus period) 만 남기고 R2 에 안 올리고 지움; 영상 판 (v) 만 덤프 올림. 이하 원래 설명:
# 절 CM98 묶음 condwhy (exe/stab_convwhy.sh 복사 + '@nok 1' = H8 에서 P1X 강성 × κ (@w8k) 만 뺌, 저장 상태 /dev/shm/dkta/condwhy_st/sv8·sv5 ('@cwload 8|5', 출력 옆 링크); 감싸개 exe/condwhy_wrap.py) -- 이하 절 CM96 설명: 절 CM96 묶음 convwhy (exe/stab_wolfh13.sh 복사; '@cwsave fi,..' 저장 판 · '@cwload 디렉토리' 다시 풀기 판 · '@cwks c' 장벽 강성 × c · '@cwnopl 1' 소성 끔 = lib/anchorflow/fem_condwhy.py (감싸개 exe/condwhy_wrap.py -> exe/wolfh13_wrap.py -> ...); 판·연기 검사는 아래 새로) -- 이하 절 CM94 설명:
# 절 CM94 묶음 wolfh13 (exe/stab_wolfh12.sh 복사; '@w13 모드' = lib/anchorflow/fem_wolfh13.py 확장 반영 MG 전처리기 (감싸개 exe/wolfh13_wrap.py -> exe/wolfh12_wrap.py -> ...), 1 = 층 0 + 거친 층, 2 = 층 0 만) -- 이하 절 CM93 설명:
# 절 CM93 묶음 wolfh13 (exe/stab_finterp.sh 복사; '@w12 모드' = lib/anchorflow/fem_wolfh13.py (감싸개 exe/wolfh13_wrap.py -> exe/finterp_wrap.py), 진단 판 (g·nd·v) 은 '--w12dg 1' (w12.jsonl); '@noext 1' = H8 에서 확장 T1E2 (@w8e·@w8d·@w8t) 만 뺌, '@nosc 1' = S_구속·C (@w5s·@w5c) 만 뺌, '@nxit 1' = 반복 xit 문 끔 (절 CM90)) -- 이하 절 CM92 설명: 절 CM92 묶음 finterp (exe/stab_wolfh11.sh 복사; '@fi 0|1|2' = 입자 F 보간 교체의 부피 진단·부피 보존 보간 (lib/anchorflow/fem_finterp.py, 감싸개 exe/finterp_wrap.py -> exe/wolfh11_wrap.py -> ...): 0 지금 정의 (진단만), 1 J 산술, 2 J 증분; 진단 판 (g·nd·v) 은 '--fidg 1' (fi.jsonl)) -- 이하 절 CM91 설명: 절 CM91 묶음 wolfh11 (exe/stab_wolfh10.sh 복사; N = '@w11 1' 새 실행기 노드별 뒤집힘 보폭 (lib/anchorflow/fem_wolfh11.py, 감싸개 exe/wolfh11_wrap.py -> exe/wolfh9_wrap.py), H8N = H8 + N, H5KN = H5n + '@w8k 1' (P1X 강성 × κ, 확장 없음) + N) -- 이하 절 CM89 설명: (exe/stab_wolfh9.sh 복사; L = '@w11 1' 새 실행기 뒤집힘 없는 보폭 필터 (lib/anchorflow/fem_wolfh11.py, 감싸개 exe/wolfh11_wrap.py -> exe/wolfh9_wrap.py), H8L = H8 + L, H5L = H5n + L) -- 이하 절 CM87 설명: (exe/stab_speed12.sh 복사; H9a = H8 + '@w9 1' (κ 0 씬 P1X 강성 그대로), H9b = H8 + '@w9 2' (× α_DP), lib/anchorflow/fem_wolfh9.py, 감싸개 exe/wolfh9_wrap.py -> exe/speed12_wrap.py -> ...) -- 이하 절 CM85 설명: (exe/stab_ext2.sh 복사; H8 = H7 + '@s12z 1' 시작점 기준 고침·X2 신호 따로, H7np = H7 − 가벼움 벌점 × κ, '@s12dg 1' 진단) -- 이하 절 CM84 설명: 공통 후보 H7 = H5 + S_구속·C + [노드 κ 0 일 때만 확장 T1E2 + P1X 강성 × κ] (lib/anchorflow/fem_ext2.py, 감싸개 exe/ext2_wrap.py; 플래그 글자는 절 CM83 H6 와 같음)
#   를 두 실행기에 (lib/anchorflow/fem_ext2.py: 새 실행기 = fem_wolfh8 훅, 옛 실행기 수박 = 이 묶음 이식) + H6np (H6 − 가벼움 벌점 × κ) -- 다섯 씬 판정 + 렌더 포함 단독. 잡 안, 코드 디렉토리에서.
# 감싸개 사슬: exe/ext2_wrap.py ('@w8'·'@x1') -> lightg4_wrap ('@l4') -> lightg3_wrap ('@l3') -> wolfh5_wrap ('@w5') -> [첫 mfix_wrap 자리를 speed9_wrap ('@s9')] speed8_wrap ('@s8') -> speed7_wrap ('@s7'·'@s6') -> mfix_wrap ('@mf') -> wmfloor_wrap ('@wf') -> hhtc2_wrap ('@hc'·'@rr'·'@s4'·'@tc') -> ghost6_wrap ('@g4...') -> [렌더 판: rtrender1_wrap -> speed4_wrap] -> escape4_wrap ('@e5') -> escape3 ('@e3') -> escape2 -> escape1/speed3 (나머지 '@'). '@s6l 1' 가벼운 판은 실행기 사본 exe/s6/ + 진단 끔.
# 노드 / ·/tmp 는 안 씀 (코드·자료·TMPDIR·결과 모두 /dev/shm/dkta), /dev/shm 사용률 65 % 이상이면 판 시작을 기다림, 덤프는 R2 크기 확인 뒤 로컬 사본만 지움.
# 사용: bash exe/stab_ext2.sh <출력> <R2> <단계> (판 목록 exe/wolfct_spec.txt)
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
CST=/dev/shm/dkta/condwhy_st                                                                    # 절 CM98: 절 CM96 저장 상태 (R2 lbours/convwhy/sv8·sv5 의 st_f*.pt 만, 읽기)
for v in sv8 sv5; do [ "$(ls $CST/$v/st_f*.pt 2>/dev/null | wc -l)" = 6 ] || rclone copy r2:storage/result/anchorflow/lbours/convwhy/$v $CST/$v --include "st_f*.pt" -q; ln -sfn $CST/$v $O/$v; done   # '@cwload 8|5' = 출력 디렉토리 옆 sv8·sv5 (숫자 표기; 경로 글자는 묶음 셸이 못 넘김) -> 심볼릭 링크 (rclone 은 안 따라감)
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
  local W8="@w5s 1 @w5c 1 @w8k 1 @w8e 1 @w8d 2 @w8t 1"                                       # 절 CM83: H6 = H5 + S_구속·C + (K) + T1E2 (절 CM81 T1E2 판 글자 그대로), H6np = H6 − 가벼움 벌점 × κ (KAP)
  x="${x// H8N / H8 @w11 1 }"; x="${x// H5KN / H5n @w8k 1 @w11 1 }"                             # 절 CM91: N = 노드별 뒤집힘 보폭 (옛 --ws_bar 4 --ws_fi 정의), H5KN = H5n + P1X 강성 × κ (확장 없음) + N
  x="${x// H9a / H8 @w9 1 }"; x="${x// H9b / H8 @w9 2 }"                                          # 절 CM87
  x="${x// H6np / H5nP $W8 }"; x="${x// H6 / H5n $W8 }"; x="${x// H7np / H5nP $W8 }"; x="${x// H8 / H5n $W8 @s12z 1 }"; x="${x// H7 / H5n $W8 }"   # 절 CM84: H7 = H6 와 같은 글자, 감싸개 ext2 가 κ 로 확장을 고름
  case $s in M*) ;; *) [[ "$x" == *" H4n "* || "$x" == *" H5n"* ]] && x="$x --cap_mul 4";; esac  # 새 실행기 첫 할당 배수 4 (절 CM72 H2) -- 속도 판에만 (Gk = 절 CM75 판은 없음)
  local SPD="@s6o 1 @s6b 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s9c 1 @s9y 1" KAP="@l4 1 @l3 10 @l3sc 1 @l3fr 1"   # 절 CM81: H5 = H4 + C·Y; R = 벽시계 예산 Rw '@s6rw 1', X = X3g '@s8g 1e6'
  x="${x// H5nP / Gpp $SPD @s6rw 1 @s8g 1e6 }"; x="${x// H5n / Gpp $SPD @s6rw 1 @s8g 1e6 $KAP }"; x="${x// H5nR / Gpp $SPD @s8g 1e6 $KAP }"; x="${x// H5nX / Gpp $SPD @s6rw 1 $KAP }"; x="${x// H5nRX / Gpp $SPD $KAP }"
  x="${x// Gk / Gpp $KAP }"                                                                    # 절 CM75 판 (G'' + 가벼움 벌점 × κ, 속도 판 없음)
  x="${x// H4n / Gpp @s6o 1 @s6b 1 @s6rw 1 @s7nm 1 @s7z 1 @s7p 1 @s8w 1 @s8g 1e6 @l4 1 @l3 10 @l3sc 1 @l3fr 1 }"; x="${x// Gpp / Gp @mf2 1 }"; x="${x// Gp / F P1X A3 @hc 5 @rr 1 @tc 8 }"
  x="${x// P1X / @g4 1e-1 @g4t 1 @g4x 1 }"; x="${x// A3 / @s4w 1 @s4y 1 @s4x 2 }"; x="${x// F / E @e3w 1 @e5r 2 @e5a 2 }"
  case $s in M*) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 @inv 1 }";; *) x="${x// E / D @tb 16.7 @tb_m 0.8 @tb_w 2 }";; esac
  case "$x " in *" @noext 1 "*) x="${x/@w8e 1 @w8d 2 @w8t 1/}"; x="${x// @noext 1 / }";; esac                         # 절 CM93 (b): 확장만 뺌 (P1X 강성 × κ 는 그대로)
  case "$x " in *" @xdeg "[0-9]" "*) local xd=$(echo " $x " | grep -o '@xdeg [0-9]' | awk '{print $2}'); x="${x/@w8d 2/@w8d $xd}"; x="${x// @xdeg $xd / }";; esac   # 절 CM98: 확장 외삽 차수 (@w8d 2 -> n)
  case "$x " in *" @nok 1 "*) x="${x/@w8k 1/}"; x="${x// @nok 1 / }";; esac                                                 # 절 CM98: P1X 강성 × κ (K) 만 뺌
  case "$x " in *" @nosc 1 "*) x="${x/@w5s 1 @w5c 1/}"; x="${x// @nosc 1 / }";; esac                                     # 절 CM93 (d): S_구속·C 만 뺌
  x="${x// D / $DFL }"; case "$x " in *" @nxit 1 "*) x="${x/--xit_from 2/}"; x="${x/--xg 1 / }"; x="${x// @nxit 1 / }";; esac   # 절 CM90 수렴 판: 반복 xit 문과 그 위 그래프 나눔 끔
  case "$x " in *" @nks 1 "*) x="${x/--ksplit 3 --ks_big/}"; x="${x// @nks 1 / }";; esac   # 절 CM87 s3 진단: 걸음 쪼개기 끔 (--K 2 고정은 --ksplit 이 K 1 을 요구)
  case $s in M*) [[ "$x" == *" $DFL "* ]] && x="$x --sp2 31";; esac; echo "$x"; }
dg_of() { local s=$1 md=$2 x
  case $s in M*) case $md in g|r|z) echo "--erec";; v) echo "--erec --dump --dump_lj";; esac; return;; esac   # v = 영상 판 (exe/stab_ext2v.sh 와 같은 글자)
  case $md in h) echo "--det --fhash"; return;; v) echo "--dump --dump_F --dump_part"; return;; esac
  case $md in nd) case $s in L) x="--hgt --pdiag --qjd --bar_cnt";; W) x="--hgt --fhash --bar_cnt --dump --dump_part";; F) x="--pdiag --qjd --fhash --dump";; B) x="--pdiag --qjd --fhash";; esac;; *) x="";; esac
  echo "$x"; }
W=/home/dkta/work
RD=/dev/shm/dkta/wolfct; mkdir -p $RD                                                          # 이 묶음이 만드는 정답 궤적 (렌더 카메라·0 프레임) 자리 (t2chain 것이 있으면 읽기만)
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
vid() {   # 영상 판 뒤 3DGS 렌더 (같은 카드): 새 실행기 씬 = exe/lb_render.py (exe/fem_cm4.sh·stab_wolfh3_vid.sh 와 같은 절차), 수박 = 일반 + 손상 색 (logJp viridis −0.2~0.4; exe/stab_wmK.sh vid 와 같은 절차). h5 는 결과 디렉토리 밖 /dev/shm 에 두고 렌더 뒤 지움
  local n=$1 s=$2 g=$3 Od=$O/$1 V=$O/$1/vj HD=/dev/shm/dkta/wolfct_vid/$1 CD=$(pwd -P) GTD CFG SH
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
log() { echo "[$(date '+%F %T')] $*" >> $O/wolfct_$STG.log; rclone copyto $O/wolfct_$STG.log $RO/wolfct_$STG.log -q; }
shm_wait() { while [ "$(shm_use)" -ge 92 ]; do log "/dev/shm 사용률 $(shm_use) % -- 기다림"; sleep 120; done; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt --exclude "e2_f*.npz"; }
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
  local n=$1 s=$2 md=$3 g=$4 x="$(dfl $2 "$5")" c b w11="" pre="" wo="" w2="" w3="" w5="" wg="" hc="" rr="" w4="" tc="" wf="" mf="" l3="" l4="" w8="" s5="" s7="" s8="" s9="" lean=0 s12="" w9="" fiv="" w12="" w13=""
  [ -f $O/$n/summary.json ] && return
  hc="$(echo " $x " | grep -o '@hc [0-9]*' | awk '{print $2}')"; tc="$(echo " $x " | grep -o '@tc [0-9]*' | awk '{print $2}')"; rr="$(echo " $x " | grep -o '@rr [0-9]*' | awk '{print $2}')"
  w4="$(echo " $x " | grep -o '@s4[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  local w5x="$(echo " $x " | grep -o '@w5[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"           # 절 CM70 S_구속 · C (exe/wolfh5_wrap.py)
  w9="$(echo " $x " | grep -o '@w9[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  w11="$(echo " $x " | grep -o '@w11[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$w11" ] && case $md in g|nd|v) w11="$w11 --w11dg 1";; esac   # 절 CM89 (exe/wolfh11_wrap.py; 진단 판은 w11.jsonl)      # 절 CM87 (exe/wolfh9_wrap.py)
  w13="$(echo " $x " | grep -o '@w13[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"
  local ct="$(echo " $x " | grep -o '@ct[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"   # 절 CM108 (exe/wolfct_wrap.py)
  local cw="$(echo " $x " | grep -o '@cw[a-z]* [^ ]*' | sed 's/^@/--/' | tr '\n' ' ')"   # 절 CM96 (exe/condwhy_wrap.py)   # 절 CM94 (exe/wolfh13_wrap.py)
  w12="$(echo " $x " | grep -o '@w12[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$w12" ] && case $md in g|nd|v) w12="$w12 --w12dg 1";; esac   # 절 CM93 (exe/wolfh13_wrap.py)
  fiv="$(echo " $x " | grep -o '@fi[a-z]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; [ -n "$fiv" ] && case $md in g|nd|v) fiv="$fiv --fidg 1";; esac   # 절 CM92 (exe/finterp_wrap.py)
  s12="$(echo " $x " | grep -o '@s12[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"   # 절 CM85 시작점 기준 고침·진단 (exe/speed12_wrap.py)
  w8="$(echo " $x " | grep -o '@\(w8\|x1\)[a-z0-9]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"            # 절 CM81 확장 B-스플라인·(K) + 절 CM83 검사 '@x1ck' 
  s9="$(echo " $x " | grep -o '@s9[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"                # 절 CM76 C·Y
  s5="$(echo " $x " | grep -o '@s6[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s7="$(echo " $x " | grep -o '@s7[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; s8="$(echo " $x " | grep -o '@s8[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; case " $x " in *" @s6l 1 "*) lean=1;; esac
  wo="$(echo " $x " | grep -o '@[a-z0-9_]* [0-9.e+,-]*' | grep -v '^@\(ct\|cw\|w13\|w12\|fi\|w11\|w9\|s12\|e[235]\|g[456]\|hc\|rr\|s4\|s6\|s7\|s8\|s9\|tc\|wf\|mf\|l3\|l4\|w5\|w8\|x1\)' | sed 's/^@/--/' | tr '\n' ' ')"; wf="$(echo " $x " | grep -o '@wf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; mf="$(echo " $x " | grep -o '@mf[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l3="$(echo " $x " | grep -o '@l3[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; l4="$(echo " $x " | grep -o '@l4[a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; wg="$(echo " $x " | grep -o '@g[456][a-z0-9_]* [0-9.e+,-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w5="$(echo " $x " | grep -o '@e5[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w2="$(echo " $x " | grep -o '@e2[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; w3="$(echo " $x " | grep -o '@e3[a-z0-9_]* [0-9.e+-]*' | sed 's/^@/--/' | tr '\n' ' ')"; x="$(echo " $x " | sed 's/@[a-z0-9_]* [0-9.e+,-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="$b $(dg_of $s $md) $x"
  [ $md = g ] && pre="AF_E2_DG=1 AF_E5_DG=1"                                                 # 절 CM43 진단 D2 + 절 CM49 진단 D5 (결과 불변)
  case $md in g|nd) pre="$pre AF_E1_DG=1 AF_G1_DG=1";; z) pre="AF_E1_DG=1";; esac
  [ $lean = 1 ] && { pre=""; c="${c/exe\/fem_an_rt_sim.py/exe/s6/fem_an_rt_sim.py}"; c="${c/exe\/fem_ao_rt_sim.py/exe/s6/fem_ao_rt_sim.py}"; }   # 절 CM65 가벼운 판: 실행기 사본 + 진단 끔 (절 CM74 speed8 과 같음)           # 단독 판도 E1 (빠른 노드·이탈, 프레임 사이 읽기만; 절 CM56 단독 판과 같음)                                   # 절 CM42 진단 (결과 불변; 재현·단독 판은 끔)
  local hx=""; [ -n "$hc" ] && hx="--hc $hc"; [ -n "$tc" ] && hx="$hx --tc $tc"
  [ "${rr:-0}" = 1 ] && hx="$hx $(rr_of $s) $w4"
  c="exe/wolfct_wrap.py $ct exe/condwhy_wrap.py $cw exe/wolfh13_wrap.py $w13 exe/wolfh12_wrap.py $w12 exe/finterp_wrap.py $fiv exe/wolfh11_wrap.py $w11 exe/wolfh9_wrap.py $w9 exe/speed12_wrap.py $s12 exe/ext2_wrap.py $w8 $l4 $l3 $w5x $s9 $s8 $s7 $s5 $mf $wf $hx $wg $w5 $w3 $w2 $wo $c"
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
  [ $md = v ] && [ -f $O/$n/pred.pt ] && vid $n $s $g                                         # 영상 판: 덤프 렌더 (pred.pt 를 지우기 전; exe/stab_ext2v.sh 와 같음)
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) hc 줄 $(wc -l < $O/$n/hc.jsonl 2>/dev/null) 렌더 $(wc -l < $O/$n/rr.jsonl 2>/dev/null) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then HEAP_ROWS=$O/$n/heap_rows.json $PY exe/wolfh10_heap.py $AG/wolf.pt $n=$O/$n > $O/$n/heap.md 2> $O/$n/heap.err; up $n; fi   # 절 CM108: 노드에서 더미 지표 (질량중심·꼭대기·안식각·채움·det F·이탈)
  if [ -f $O/$n/pred.pt ]; then
    if [ $md = v ]; then                                                                     # 영상 판만 덤프를 R2 에 (크기 확인 뒤 로컬 사본 지움)
      rclone copyto $O/$n/pred.pt $RO/$n/pred.pt -q
      [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $O/$n/pred.pt)" ] && rm -f $O/$n/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"
    else rm -f $O/$n/pred.pt; log "$n 판정 판 -- 지표만 남기고 pred.pt 지움 (안 올림)"; fi
  fi
  rm -f $O/$n/e2_f*.npz; }
SPEC=$(grep -v '^#' exe/wolfct_spec.txt | awk -v s=$STG '$1==s')
[ -f exe/wolfct_pre_$STG.sh ] && { log "앞 단계 스크립트 wolfct_pre_$STG.sh"; bash exe/wolfct_pre_$STG.sh $O $RO > $O/pre_$STG.out 2>&1; rclone copy $O $RO -q --include "pre_$STG.out"; }
if echo "$SPEC" | grep -q '@rr 1' || [ $STG = v ]; then
  mkgt wolf $GTW --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json
  mkgt bread $GTB --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json
  mkgt wm $GTM --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --allow_nonfinite
  log "렌더 자료: ficus $FGT wolf $GTW ($(ls $GTW/meta.json 2>&1)) bread $GTB ($(ls $GTB/meta.json 2>&1)) 수박 $GTM ($(ls $GTM/meta.json 2>&1)) 모델 $WMM"
fi
# ---- 판 (카드마다 하나; 계산 프로세스 0 인 카드를 공용 표식 디렉토리로 잡음 -- 다른 묶음 카드 안 건드림)
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판 끝 $STG"
for n in $(echo "$SPEC" | awk '{print $2}'); do                                               # 판마다 cwinfo 요약 (훅 수·오류)
  log "[ct] $n $(cat $O/$n/ctinfo.json 2>/dev/null | head -c 300)"; log "[cw] $n $(cat $O/$n/cwinfo.json 2>/dev/null | $PY -c "import json,sys; i=json.load(sys.stdin); print({k: i.get(k) for k in ('nsave','nload','nframe','nctl3','nctl7','nroll','nalloc','nopl_src','KS0','KS1','err')})" 2>&1) cw줄 $(wc -l < $O/$n/cw.jsonl 2>/dev/null) 상태 $(ls $O/$n/st_f*.pt 2>/dev/null | wc -l) $(grep -a 'rror' $O/$n/log.txt | tail -1 | cut -c1-200)"
done
case $STG in *k)                                                                            # 연기 단계 (이름 끝 k): 저장 판은 상태 파일, 다시 풀기 판은 cw.jsonl 줄 = 상태 수·훅 수 > 0·오류 없음 -- 아니면 실패 표식
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    NOPL=0; KSX=0; case " $x " in *" @cwnopl 1 "*) NOPL=1;; esac; case " $x " in *" @cwks "*) KSX=1;; esac
    case " $x " in *" @ct 1 "*|*" @ctfd 1 "*) $PY - $O/$n "$x" >> $O/wolfct_$STG.log 2>&1 <<'PYX' || { OK=0; log "[연기] $n ct 기록 없음/오류"; };;
import json, sys
d, x = sys.argv[1], " " + sys.argv[2] + " "
i = json.load(open(d + "/ctinfo.json"))
ok = not i.get("err") and (" @ct 1 " not in x or i["n_src_ct"] > 0) and (" @ctfd 1 " not in x or i["n_fd_launch"] > 0)
if " @ctfd 1 " in x:
    L = [json.loads(l) for l in open(d + "/cw.jsonl")]
    fd = [o["fd"] for q in L for st in q["steps"] for o in st["its"] if "fd" in o]
    ok = ok and len(fd) > 0 and sum(c[0] for f in fd for c in f) > 0
    print(d, "fd 반복", len(fd), "첫", [[round(v, 3) if v < 1e3 else "%.3g" % v for v in c[:9]] for c in fd[0]] if fd else None)
print(d, "ct", ok, {k: i.get(k) for k in ("ct", "fd", "n_src_ct", "n_src_fd", "n_fd_launch", "n_alloc", "err")})
sys.exit(0 if ok else 1)
PYX
    esac
    case " $x " in
      *" @cwsave "*) [ "$(ls $O/$n/st_f*.pt 2>/dev/null | wc -l)" -ge 1 ] || { OK=0; log "[연기] $n 상태 파일 없음"; };;
      *" @cwload "*) $PY - $O/$n $NOPL $KSX >> $O/wolfct_$STG.log 2>&1 <<'PYX' || { OK=0; log "[연기] $n 다시 풀기 기록 없음/오류"; };;
import json, sys
d, nopl, ksx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
i = json.load(open(d + "/cwinfo.json")); L = [json.loads(l) for l in open(d + "/cw.jsonl")]
ok = i["nload"] >= 1 and i["nctl3"] > 0 and i["nctl7"] > 0 and i["nroll"] > 0 and not i.get("err") and len(L) >= 1 and all(q["steps"] and q["steps"][0]["its"] for q in L)
ok = ok and (not nopl or i["nopl_src"] > 0) and (not ksx or i.get("KS1") is not None)
print(d, ok, [(q["f"], len(q["steps"][0]["its"]), q["steps"][0]["rs"][:3], q["steps"][0]["its"][0].get("kap")) for q in L])
sys.exit(0 if ok else 1)
PYX
    esac
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $STG $(date '+%F %T')" > $O/wolfct.FAIL; rclone copy $O $RO -q --include wolfct.FAIL; };;
esac
log "단계 끝 $STG"
