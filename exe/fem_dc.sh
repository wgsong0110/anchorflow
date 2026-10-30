#!/bin/bash
# 사용자 10-30 "발산이 일어난 셀을 영상에서 표시 -- 발산한 셀에 들어 있던 입자가 어떤 입자인지":
#   (1) 수박: 절 AG 와 같은 명령 (공통 h_s, N480, 셀 증분 RM) + --lagdiag --divtrack (계산 불변 진단) + --dump_f32 --dump_div
#   (2) lego: 절 U 와 같은 설정 (절 R 셀 집중 질량 --mass tet, 클램프 끔, 속도 바닥, L000 N192) + --lagdump 3 --divtrack + --dump_f32 --dump_div
#   -> exe/fem_dc_cells.py (발산 프레임·처음 터진 셀·그 셀 입자 번호, 위치·F 특이값·이웃 퍼짐)
#   -> 3DGS 알파블렌딩 렌더 (정답·우리 x 보통·투시, 표시 입자 빨강) -> exe/fem_dc_compose.py 합침 -> v20261009/diverge_cells_{wm,lego}.mp4
# 비측정, 빈 카드 하나 (카드 0 이면 gpu0.lock 을 잡아 다른 갈래 단독 측정과 겹치지 않게). 출력 /tmp/dkta/lbours/fem_dc -> R2 lbours/fem_dc.
cd "$(dirname "$0")/.." || exit 1
WT=$(pwd)
export PYTHONUTF8=1 OMP_NUM_THREADS=8; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
W=/home/dkta/work; RR=r2:storage/result/anchorflow; O=/tmp/dkta/lbours/fem_dc; mkdir -p $O
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
exec 9>$W/gpu0.lock
while true; do
  C=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 500 {print $1}')
  G=$(echo "$C" | grep -v '^0$' | head -1)
  [ -n "$G" ] && break
  if echo "$C" | grep -q '^0$' && flock -n 9; then
    [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && { G=0; echo "[잠금] gpu0.lock $(date +%T)"; break; }
    flock -u 9
  fi
  echo "[대기] 빈 카드 없음 $(date +%T)"; sleep 60
done
echo "[카드] $G $(date +%T)"
# ---- (1)(2) 시뮬 ----
ag_scene wm
EX_ROOT=fem_dc bash exe/fem_ab_run.sh dc_wm $G $SARGS $REC --lagdiag --divtrack --dump_f32 --dump_div --N $NC < /dev/null
EX_ROOT=fem_dc bash exe/fem_ab_run.sh dc_lego $G --mode lag --opt --mass tet --lagdiag --lagdump 3 --divtrack --floor vel --N 192 --dump_f32 --dump_div < /dev/null
# ---- 발산 셀 ----
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_dc_cells.py --run $O/dc_wm --data $AGD/wm.pt < /dev/null > $O/dc_wm/dc.log 2>&1; tail -12 $O/dc_wm/dc.log
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_dc_cells.py --run $O/dc_lego --data $W/one/L000/L000.pt < /dev/null > $O/dc_lego/dc.log 2>&1; tail -12 $O/dc_lego/dc.log
rclone copy $O $RR/lbours/fem_dc -q
# ---- 렌더·합침 ----
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon
rend() {   # <씬> <실행 디렉토리> <정답 h5 디렉토리> <h5 변환 traj> <fps>
  local S=$1 D=$2 GH=$3 TR=$4 FPS=$5
  local K=$(python -c "import json;print(json.load(open('$D/dc/dc.json'))['td'])")
  local TT=$(python -c "import json,numpy as np;d=json.load(open('$D/dc/dc.json'));print(f\"발산 셀 입자 {len(np.load('$D/dc/mark_idx.npy'))} 개 (프레임 {d['td']+1}, 서브스텝 {d['s_div']+1}/{d['N']})\")")
  local SB=$(python -c "import json;d=json.load(open('$D/dc/dc.json'));l=d['loc'];print(f\"상태 0~{d['td']} (상태 k = k 프레임 뒤), 셀 {l['n_cells']} 개 · 바닥 접촉 {l['floor_contact']} · 표면 {l['surface']} · 내부 {l['interior']}\")")
  mkdir -p $D/vj
  [ -f $D/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $D/roll_fem.pt --traj $TR --out $D/h5_fem > $D/h5_fem.log 2>&1
  local MKA="--mark $D/dc/mark_idx.npy"
  for v in gt ours; do
    local SIM=$([ $v = gt ] && echo $GH || echo $D/h5_fem)
    for f in 1 0.08; do
      local TAG=${v}$([ $f = 1 ] || echo _fade)
      [ -f $D/vj/$TAG.mp4 ] && continue
      if [ $S = wm ]; then
        (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$G python -u $WT/exe/wm_render.py --model_path $WMM --config $CFG --sim $SIM --out $D/vj/$TAG.mp4 --frames $K \
           $MKA --mark_fade $f --mark_px $D/vj/$TAG.json < /dev/null > $D/vj/r_$TAG.log 2>&1)
      else
        CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $SIM --config $W/lbdata/config.json --shape lego --out $D/vj/$TAG.mp4 --frames $K \
           $MKA --mark_fade $f --mark_px $D/vj/$TAG.json < /dev/null > $D/vj/r_$TAG.log 2>&1
      fi
      tail -2 $D/vj/r_$TAG.log
    done
  done
  python -u exe/fem_dc_compose.py --gt $D/vj/gt.mp4 --ours $D/vj/ours.mp4 --gt_fade $D/vj/gt_fade.mp4 --ours_fade $D/vj/ours_fade.mp4 \
     --gt_px $D/vj/gt_fade.json --ours_px $D/vj/ours_fade.json --title "$TT" --sub "$SB" --fps $FPS --out $O/diverge_cells_$S.mp4 < /dev/null > $D/vj/compose.log 2>&1
  tail -2 $D/vj/compose.log
  rclone copyto $O/diverge_cells_$S.mp4 $RR/v20261009/diverge_cells_$S.mp4 -q && echo "[R2] v20261009/diverge_cells_$S.mp4"
}
ag_scene wm
rend wm $O/dc_wm $GTH5 $AGD/gt_wm 2
rend lego $O/dc_lego $W/lbdata/test_low/L000 $W/lbdata/test_low/L000 6
[ "$G" = 0 ] && flock -u 9
# ---- R2 통째로 -> 확인 뒤 /tmp 큰 덤프만 지움 ----
rclone copy $O $RR/lbours/fem_dc -q
for D in $O/dc_wm $O/dc_lego; do
  if [ -f $O/diverge_cells_$(basename $D | cut -c4-).mp4 ] && rclone check $D $RR/lbours/fem_dc/$(basename $D) --one-way --size-only -q 2>/dev/null; then
    rm -rf $D/roll_fem.pt $D/h5_fem $D/lagdump.pt; echo "[R2 확인·/tmp 정리] $D"
  else echo "[R2 확인 실패 또는 영상 없음 -- /tmp 유지] $D"; fi
done
touch $O/DC_DONE; rclone copy $O $RR/lbours/fem_dc -q
echo "DC_DONE $(date +%T)"
