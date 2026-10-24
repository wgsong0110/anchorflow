#!/bin/bash
# 망 파라미터 프레임별 최적화 새 판 다섯 (사용자 10-23): 맞춤 -> 3DGS 렌더 -> 탄성 에너지 색 -> R2 통째 (한 궤적).
#   bash exe/pfnew.sh <판> <궤적 L000|0004> <GPU>
#   판:  nodestar     = --ov_obj nodestar (매 프레임 모델 상태 L-BFGS 500 해 u* 를 노드 L2 로 따라가기)
#        nodealt      = --ov_obj nodealt  (L-BFGS 10 -> Adam 50 을 40 바퀴)
#        floorpe      = --ov_obj ip + --bc_feat distpe (바닥 부호 거리 + sin/cos, K=--bc_pe 6)
#        floorcontact = --ov_obj ip + --bc_feat contact (접촉 셀 꼭짓점만 바닥 정보)
#        noin         = --ov_obj ip (바닥 입력 원래 그대로) -- 위 둘의 기준판
#        p2g          = noin + --node_feat p2g (노드 입력의 입자 부분 = MPM P2G 질량·속도·F, 2 차 B-스플라인; 사용자 10-23)
#        apic         = noin + --node_feat p2g --apic --p2g_stress --bc_feat near (APIC C 상태·입력 + 응력 입력 + near 바닥; 사용자 10-24)
#        p2gnear      = noin + --node_feat p2g --bc_feat near (apic 의 비교판)
#        lattn        = noin + --gnn_layer lattn --attn_heads 8 (몸통 = 국소 어텐션 8 층 128; 사용자 10-23)
#        lattn4       = lattn 과 같고 4 층
#        gunet        = noin + --gnn_layer unet --gunet_levels 3 (몸통 = Graph U-Net, 격자 2h·4h 묶기 풀링 + 스킵; 사용자 10-24)
#        gunet_near   = gunet + --bc_feat near --bc_near_h 1.0 (바닥 = 부호 거리 < h 꼭짓점이 있는 셀의 꼭짓점만 [플래그, 거리/h], 법선 없음, 거친 수준에도)
#        gunet_cf     = gunet + --bc_feat contactflag (바닥 = 교차 셀 꼭짓점 플래그 1 채널만)
#        prediag      = noin + --out_precond diag  (망 출력 u = z·sqrt(median d / d_i), d = M/h² + 탄성 헤시안 대각; 사용자 10-23)
#        prems        = noin + --out_precond ms    (다중 해상도 계층 평활 u = (z + Σ 2^l I_l R_l z)/(1+Σ 2^l), 2h·4h·8h)
#        prelbfgs     = noin + --out_precond lbfgs (프레임마다 노드 출력 L-BFGS 50 회 곡률 쌍 -> 출력 기울기에 H⁻¹/γ)
#   PFNEW_LEN=N (기본 80): 짧은 시험 -- 프레임 N 개만 맞추고 렌더·psi 없이 끝 (출력 lbours/<디렉토리>_lenN/<T>)
#   공통 명령 = lbours/pf3obj/{ip,nl2}/<T>/cmd.txt 의 명령에서 --net_inertial 을 빼고 (사용자 규칙: 새 실행에 관성 예측 더하기 금지)
#   --ov_obj·바닥 입력·출력 경로만 바꾼 것. 출력 lbours/<디렉토리>/<T>, R2 r2:storage/result/anchorflow/lbours/<디렉토리>/<T>.
#   이어 돌기: AF_OVNET_FIT 의 frames/frame_XX.pt (다시 띄우면 마지막 프레임 다음부터). 끝 표식 <출력>/.done
set -u
K=${1:?판}; T=${2:?궤적}; G=${3:?GPU}
W=/home/dkta/work; WT=$(cd "$(dirname "$0")/.." && pwd); C=$(cd $WT && git rev-parse --short HEAD)
case $K in
  nodestar) DIR=nodestar; OBJ=nodestar; XF=""; XE="" ;;
  nodealt) DIR=nodealt; OBJ=nodealt; XF=""; XE="AF_ALT_ROUNDS=${AF_ALT_ROUNDS:-40} AF_ALT_LBFGS=${AF_ALT_LBFGS:-10} AF_ALT_ADAM=${AF_ALT_ADAM:-50}" ;;
  floorpe) DIR=pfip_floorpe; OBJ=ip; XF="--bc_feat distpe --bc_pe 6"; XE="" ;;
  floorcontact) DIR=pfip_floorcontact; OBJ=ip; XF="--bc_feat contact"; XE="" ;;
  noin) DIR=pfip_noin; OBJ=ip; XF=""; XE="" ;;
  p2g) DIR=pfip_p2g; OBJ=ip; XF="--node_feat p2g"; XE="" ;;
  apic) DIR=pfip_apic; OBJ=ip; XF="--node_feat p2g --apic --p2g_stress --bc_feat near --bc_near_h 1.0"; XE="" ;;   # 사용자 10-24: (A) P2G + APIC 아핀 속도 C (상태·입력) + 응력 (내부력·평균 τ) + near 바닥
  p2gnear) DIR=pfip_p2gnear; OBJ=ip; XF="--node_feat p2g --bc_feat near --bc_near_h 1.0"; XE="" ;;   # 사용자 10-24: (B) apic 의 비교판 = P2G + near 바닥 (APIC·응력 없이)
  p2gstress) DIR=pfip_p2gstress; OBJ=ip; XF="--node_feat p2g --p2g_stress --bc_feat near --bc_near_h 1.0"; XE="" ;;   # 사용자 10-24: 비교판(p2gnear) + 응력 (내부력·평균 τ), APIC 없음
  lattn) DIR=pfip_lattn; OBJ=ip; XF="--gnn_layer lattn --attn_heads 8"; XE="" ;;
  gunet_near) DIR=pfip_gunet_near; OBJ=ip; XF="--gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0"; XE="" ;;   # 사용자 10-24: + 바닥 입력 = 접촉 여지 셀만 [플래그, 거리/h], 법선 없음, 거친 수준에도
  gunet_cf) DIR=pfip_gunet_contactflag; OBJ=ip; XF="--gnn_layer unet --gunet_levels 3 --bc_feat contactflag"; XE="" ;;   # 사용자 10-24: + 바닥 입력 = 교차 셀 플래그만
  gunet) DIR=pfip_gunet; OBJ=ip; XF="--gnn_layer unet --gunet_levels 3"; XE="" ;;   # 사용자 10-24: 몸통만 Graph U-Net (h·2h·4h, 층 2+2+2+1+1)
  lattn4) DIR=pfip_lattn4; OBJ=ip; XF="--gnn_layer lattn --attn_heads 8 --gnn_layers 4"; XE="" ;;   # 사용자 10-23: 국소 어텐션 층수 낮춤 (8 -> 4, 뒤의 --gnn_layers 가 앞 값을 덮는다)
  prediag) DIR=pfip_prediag; OBJ=ip; XF="--out_precond diag"; XE="AF_PRE_HUTCH=${AF_PRE_HUTCH:-0}" ;;
  prems) DIR=pfip_prems; OBJ=ip; XF="--out_precond ms"; XE="AF_PRE_MS_LEVELS=${AF_PRE_MS_LEVELS:-3}" ;;
  prelbfgs) DIR=pfip_prelbfgs; OBJ=ip; XF="--out_precond lbfgs"; XE="AF_PRE_LBFGS=${AF_PRE_LBFGS:-50}" ;;
  *) echo "[오류] 판 $K"; exit 1 ;;
esac
NL=${PFNEW_LEN:-80}; [ "$NL" = 80 ] || DIR=${DIR}_len$NL
O=$W/lbours/$DIR/$T; R=r2:storage/result/anchorflow/lbours/$DIR/$T; mkdir -p $O
S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick; WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; set -u
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=$HOME/bin:$PATH
up(){ rclone copy $O $R -q; }
echo "[시작] $K $T GPU$G 커밋 $C $(date '+%F %T') 노드 $(hostname)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
[ -f $O/.done ] && { echo "[끝나 있음] $O/.done"; exit 0; }
# 1) 맞춤 (기준 명령 = pf3obj ip 명령; --net_inertial 빼고 --ov_obj·바닥 입력·경로만 바꾼다)
BASE=$(sed -n 's/^\[실행 명령[^]]*\] //p' $W/lbours/pf3obj/ip/$T/cmd.txt | head -1)
[ -n "$BASE" ] || { echo "[오류] 기준 명령 없음"; exit 1; }
CMD=$(sed -e "s#/home/dkta/work/lbours/pf3obj/ip/$T#$O#g" -e "s#r2:storage/result/anchorflow/lbours/pf3obj/ip/$T#$R#g" \
          -e "s/CUDA_VISIBLE_DEVICES=[0-9]*/$XE CUDA_VISIBLE_DEVICES=$G/" -e 's/ --net_inertial / /' \
          -e "s/ --ov_obj ip / --ov_obj $OBJ /" -e "s# --bc_stick_xn # --bc_stick_xn $XF #" -e "s/ --eval_len 80 / --eval_len $NL /" <<<"$BASE" | tr -s ' ')
grep -q -- "--net_inertial" <<<"$CMD" && { echo "[오류] --net_inertial 이 남았다"; exit 1; }
grep -q -- "--ov_obj $OBJ " <<<"$CMD" || { echo "[오류] --ov_obj 바꾸기 실패"; exit 1; }
[ -z "$XF" ] || grep -q -- "$XF" <<<"$CMD" || { echo "[오류] 추가 인자 넣기 실패 ($XF)"; exit 1; }
grep -q -- "--eval_len $NL " <<<"$CMD" || { echo "[오류] --eval_len $NL 바꾸기 실패"; exit 1; }
echo "[기준] $BASE" > $O/cmd_base.txt
echo "[실행 명령 $(date '+%F_%T') 커밋 $C] $CMD" >> $O/cmd.txt
echo "[명령] $CMD"
if [ ! -f $O/roll_net.pt ]; then
  ( cd $WT && $CMD >> $O/run.log 2>&1 < /dev/null )
  grep -q OVNET_FIT_DONE $O/run.log || { echo "[오류] 맞춤 안 끝남"; tail -5 $O/run.log; up; exit 1; }
fi
up
[ "$NL" = 80 ] || { touch $O/.done; up; echo "PFNEW_DONE $K $T 짧은 시험 $NL 프레임 $(date '+%F %T')"; exit 0; }
# 2) 3DGS 렌더 (pf3obj.sh render 와 같은 명령)
cd $WT
[ -f $O/h5_net/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_net.pt --traj $(gt_of $T) --out $O/h5_net > $O/h5_net.log 2>&1
[ -f $O/net.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_net --config $W/lbdata/config.json --shape lego --out $O/net.mp4 > $O/r_net.log 2>&1 < /dev/null
[ -f $O/net.mp4 ] || { echo "[오류] 렌더"; up; exit 1; }
up
# 3) 탄성 에너지 색 (pf3obj.sh psi 와 같은 명령, 고정 범위)
D=$O/psi; mkdir -p $D
( cd $WP && { [ -f $D/col_$K.npy ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump sp=$S/roll_ov_$T.pt \
      --dump $K=$O/roll_net.pt --out $D --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
  grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
  [ -f $D/h5_$K/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_net.pt --traj $(gt_of $T) --out $D/h5_$K > $D/h5_$K.log 2>&1
  [ -f $D/psi_${K}_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $D/h5_$K --config $W/lbdata/config.json --shape lego \
      --colors $D/col_$K.npy --out $D/psi_${K}_lego.mp4 > $D/r_$K.log 2>&1 < /dev/null
  [ -f $D/psi_${K}_lego.mp4 ] ) || { echo "[오류] psi"; up; exit 1; }
for n in gt sp; do cp $NS/$T/psi_${n}_lego.mp4 $D/psi_${n}_lego.mp4; done; cp $NS/$T/range.json $D/range_spstick.json
cat $D/defstats_$T.txt
touch $O/.done; up
rclone check $O $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo "PFNEW_DONE $K $T $(date '+%F %T')"
