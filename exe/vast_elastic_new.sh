#!/bin/bash
# elastic_new (사용자 11-15 "pillow2sofa 대신 다른 탄성 씬"): mic·ship 을 PG jelly (pillow2sofa 공식 값) 로, 레고 L000 기울기로 0.3 m 낙하, 50 프레임.
#   vast_nshape_pg.sh 와 같은 사슬. 바뀐 것: (1) 3DGS 없으면 공식 3DGS 기본값 학습 (vast_nshape.sh 와 같은 판) (1b) 바닥 z = 최저점 - 0.3 (exe/elastic_new_floor.py)
#   (2) 정답 = 원본 PhysGaussian 8339ed6 과 i-physgaussian 사본을 같이 띄워 원본이 끝나면 원본, 아니면 사본 ("정답 (PG 사본)")
#   (3) 우리 판 K=1 만 (4) 단독 시간 판 (5) 렌더 = 위에서 내려다보는 카메라 (--elev) 2 칸, 칸 제목에 오차·ms. destroy 안 함.
#   env: R2_*, XS_SC, XS_OBJ, XS_CODE, (NS_ELEV 40, NS_BACK2 1.1, NS_CAMBACK 1.6)
set -u
SC=${XS_SC:?}; OBJ=${XS_OBJ:?}; TAG=elastic_new_$OBJ
R=r2:storage/result/anchorflow; RX=$R/lbours/elastic_new; DS=r2:storage/datasets/anchorflow/nshape; VD=$R/v20261009; MI=$R/_vastmirror
mkdir -p /root/sv; LOG=/root/sv/$TAG.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
say "[시작] $(hostname) 장면 $SC 물체 $OBJ"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then say "[전력 제한] $PL W < $PD W"; echo "전력 제한" | rclone rcat $RX/$TAG.END; exit 1; fi
export HOME=/root
python -c "import torch, cupy, h5py, warp, lpips, diff_gaussian_rasterization, simple_knn, taichi, plyfile" 2>/dev/null || { say "[세팅 실패] pip"; echo "pip 실패" | rclone rcat $RX/$TAG.END; exit 1; }
W=/home/dkta/work; AF=/root/sv/af_$OBJ; AG=/dev/shm/dkta/ag_$OBJ; PGU=$W/PhysGaussian_upstream; IPG=$W/i-physgaussian; GF=$W/GaussianFluent
mkdir -p $W/pgmodel $AF $AG /home/dkta/.conda/envs/af/bin /dev/shm/dkta/tmp
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
[ -f $PGU/gs_simulation.py ] || rclone copy $MI/PhysGaussian_upstream_8339ed6 $PGU --transfers 32 -q &
[ -f $IPG/gs_simulation.py ] || rclone copy $MI/i-physgaussian $IPG --exclude ".git/**" --transfers 32 -q &
[ -f $GF/gaussian-splatting/train.py ] || rclone copy $MI/GaussianFluent $GF --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q &
rclone cat ${XS_CODE:?}/af_code.tar | tar -x -C $AF
[ -f $W/font.otf ] || rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
wait
export PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp
# ---- (1) 3DGS
MD=$W/pgmodel/$OBJ
if rclone lsf $DS/model/$OBJ/point_cloud/iteration_30000/ 2>/dev/null | grep -q point_cloud.ply; then
  rclone copy $DS/model/$OBJ $MD --transfers 16 -q; say "[3DGS] R2 에 있음 -> 받음"
else
  rclone copy r2:storage/datasets/nerf_synthetic/$OBJ $W/data/$OBJ --transfers 32 -q
  python $AF/exe/patch_3dgs_train.py $GF/gaussian-splatting | tee -a $LOG
  t0=$(date +%s)
  (cd $GF/gaussian-splatting && python -u train.py -s $W/data/$OBJ -m $MD -w --eval < /dev/null > $W/train_$OBJ.log 2>&1); rc=$?
  say "[3DGS] 학습 rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -a 'Evaluating test\|Evaluating train' $W/train_$OBJ.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(grep -a 'rror' $W/train_$OBJ.log | tail -1 | cut -c1-200)"
  cp $W/train_$OBJ.log $MD/train.log 2>/dev/null
  [ -s $MD/point_cloud/iteration_30000/point_cloud.ply ] || { say "[3DGS 실패]"; echo "3DGS 실패" | rclone rcat $RX/$TAG.END; exit 1; }
  rclone copy $MD $DS/model/$OBJ --transfers 16 -q; say "[3DGS] R2 올림 $DS/model/$OBJ"
fi
CFG=$W/cfg_$SC.json; cp $AF/exe/nshape_cfg/$SC.json $CFG; NF=$(python -c "import json;print(json.load(open('$CFG'))['frame_num'])")
say "$(python $AF/exe/elastic_new_floor.py $MD $CFG 0.3)"
# ---- (2) 정답: 원본과 사본을 같이
O=$W/gt/$SC; OC=$W/gt/${SC}_ipg; mkdir -p $O $OC; t0=$(date +%s)
(cd $PGU && WARP_CACHE_PATH=$W/wpc timeout 1200 python -u gs_simulation.py --model_path $MD --config $CFG --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1) & PU=$!
(cd $IPG && WARP_CACHE_PATH=$W/wpc2 timeout 3600 python -u gs_simulation.py --model_path $MD --config $CFG --output_path $OC --output_h5 < /dev/null > $OC/run.log 2>&1) & PC=$!
wait $PU; rcu=$?; nu=$(ls $O/simulation_ply 2>/dev/null | grep -c h5)
say "[정답 원본] rc=$rcu 프레임 $nu $(( $(date +%s) - t0 )) 초 $(grep -a 'rror' $O/run.log | tail -1 | cut -c1-200)"
if [ $nu -ge $((NF + 1)) ]; then GTL="정답 (PhysGaussian)"; echo "PhysGaussian 8339ed6 gs_simulation.py (원본 그대로)" > $O/RUNNER; pkill -P $PC 2>/dev/null; kill $PC 2>/dev/null; wait $PC 2>/dev/null
else
  wait $PC; rcc=$?; nc=$(ls $OC/simulation_ply 2>/dev/null | grep -c h5)
  say "[정답 사본] rc=$rcc 프레임 $nc $(( $(date +%s) - t0 )) 초 $(grep -a 'rror' $OC/run.log | tail -1 | cut -c1-200)"
  [ $nc -ge $((NF + 1)) ] || { say "[정답 실패]"; rclone copy $O $DS/gt_pg/${SC}_fail -q; rclone copy $OC $DS/gt_pg/${SC}_ipg_fail -q; echo "정답 실패" | rclone rcat $RX/$TAG.END; exit 1; }
  echo "i-physgaussian 사본 gs_simulation.py (원본 8339ed6 이 rc=$rcu 프레임 $nu 로 안 끝남)" > $OC/RUNNER; cp $O/run.log $OC/run_upstream.log
  O=$OC; GTL="정답 (PG 사본)"
fi
cp $CFG $O/config.json; rclone copy $O $DS/gt_pg/$SC --transfers 16 -q; say "[정답] $GTL R2 올림 $DS/gt_pg/$SC"
cd $AF
python -u exe/pg_h5_to_pt.py --h5_dir $O/simulation_ply --config $CFG --out_pt $AG/$SC.pt --out_dir $AG/gt_$SC > $W/cv_$SC.log 2>&1 || { say "[변환 실패] $(tail -1 $W/cv_$SC.log)"; echo "변환 실패" | rclone rcat $RX/$TAG.END; exit 1; }
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $W/BA_1.txt
# ---- (3) 우리 판 K=1, (4) 단독 시간 판
run1() { local sfx=$1 drop=$2 nodump=$3; local D=$W/ours/${SC}_K1${sfx}; mkdir -p $D
  local CMD=$(NS_DROP="$drop" NS_NODUMP=$nodump python exe/nshape_cmd.py $W/BA_1.txt $AG/$SC.pt $NF $SC $D $AG/gt_$SC $AG/gt_$SC/config.json 1)
  echo "$CMD" > $D/cmd.txt; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 bash -c "$CMD" < /dev/null > $D/run.log 2>&1; local rc=$?
  say "  [$SC K=1$sfx] rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $D/run.log | tail -1 | cut -c1-200) $(grep -a 'rror' $D/run.log | tail -1 | cut -c1-160)"
  rclone copy $D $RX/ours/${SC}_K1${sfx} -q --exclude "pred.pt" --transfers 8
  return $rc; }
run1 "" "" 0
D=$W/ours/${SC}_K1
run1 _t "--pdiag --qjd --fhash --hgt --bar_cnt" 1
read ERR MS < <(python -c "
import json
s=json.load(open('$D/summary.json')); t=json.load(open('$W/ours/${SC}_K1_t/summary.json'))
e=s.get('err_mean_done'); print('%.2f' % (100*e) if e is not None else 'nan', '%.1f' % t['ms_mean'] if t.get('ms_mean') is not None else 'nan')")
say "  [$SC] 오차 $ERR % 단독 $MS ms 발산 $(python -c "import json;print(json.load(open('$D/summary.json')).get('diverged'))")"
# ---- (5) 렌더
CB=${NS_CAMBACK:-1.6}; EL=${NS_ELEV:-40}; B2=${NS_BACK2:-1.1}
python -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $AG/gt_$SC --out $D/h5_ours > $D/h5.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim $AG/gt_$SC --config $CFG --shape $SC --out $D/gt.mp4 --cam_back $CB --elev $EL --back2 $B2 --png $D/gt_png < /dev/null > $D/r_gt.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim $D/h5_ours --config $CFG --shape $SC --out $D/ours.mp4 --cam_back $CB --elev $EL --back2 $B2 --png $D/ours_png < /dev/null > $D/r_ours.log 2>&1
python -u exe/gfs_post.py vid --gt_label "$GTL" --our_label "우리 K=1  오차 ${ERR}%  ${MS} ms" --gt_png $D/gt_png --our_png $D/ours_png --out_json $D/visual.json --out_mp4 $D/cmp_raw.mp4 --font $W/font.otf --fps 25 > $D/vid.log 2>&1
V=$D/${OBJ}_K1_cmp.mp4
ffmpeg -y -loglevel error -i $D/cmp_raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $V
DEC=$(ffmpeg -v error -i $V -f null - 2>&1 | wc -l)
say "  [$SC] 렌더 정답 $(ls $D/gt_png 2>/dev/null | wc -l) 우리 $(ls $D/ours_png 2>/dev/null | wc -l) $(tail -1 $D/vid.log | cut -c1-200) 영상 $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $V 2>&1) 디코드 오류 줄 $DEC"
rclone copyto $V $VD/${OBJ}_K1_cmp.mp4 -q
rclone copy $D $RX/ours/${SC}_K1 -q --include "*.json" --include "*.log" --include "cmd.txt" --include "*.mp4"
say "[끝] 오차 $ERR % ms $MS 영상 $VD/${OBJ}_K1_cmp.mp4"; echo "끝 $(date) err=$ERR ms=$MS" | rclone rcat $RX/$TAG.END
