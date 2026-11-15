#!/bin/bash
# 절 CM186 (PG 경로, 사용자 11-15 "GF 말고 PG"): 새 형상 장면 하나 = vast.ai 인스턴스 하나 (A6000 1 장).
#   (1) 3DGS 모델 R2 datasets/anchorflow/nshape/model/<물체> (없으면 공식 3DGS 학습, exe/vast_nshape.sh 와 같음)
#   (2) 정답 = **원본 PhysGaussian** (XPandora/PhysGaussian 8339ed6 + 하위 모듈 gaussian-splatting d9fad7b, R2 _vastmirror/PhysGaussian_upstream_8339ed6) gs_simulation.py --output_h5 그대로
#       (채우기가 30 분 넘게 안 끝나면 -- taichi f32 sym_eig 무한대 (메모리 physgaussian-filling-hangs-symeig) -- exe/fix_pg_filling.py 를 사본에만 적용해 다시, 로그에 남김)
#   (3) 우리 판 = PG 장면용 기본 판 (exe/nshape_cmd.py: bread BA_1 명령 + 레고 바닥, 렌더 훅 --rr pg), 발산하면 K 2 -> 4 -> 8
#   (4) 단독 시간 판 (진단 플래그 뺌, 덤프 없음) (5) 위상 지표 (exe/gfs_post.py topo) (6) PG 렌더 (exe/lb_roll_to_h5.py + exe/lb_render.py --cam_back NS_CAMBACK) 비교 영상
#   (7) R2 확인 뒤 destroy.   env: R2_*, XS_TAG, XS_SC, XS_OBJ, XS_CODE
set -u
TAG=${XS_TAG:-nshape}; SC=${XS_SC:?}; OBJ=${XS_OBJ:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/nshape/pg; DS=r2:storage/datasets/anchorflow/nshape; VD=$R/nshape/vid; MI=$R/_vastmirror
mkdir -p /root/sv; LOG=/root/sv/$TAG.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?} 장면 $SC 물체 $OBJ"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < 기본 $PD" | rclone rcat $R/lbours/nshape/$TAG.POWERCAP; say "[전력 제한] $PL W < $PD W"; exit 1; fi
export HOME=/root
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
if ! python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization, simple_knn, mcubes, taichi" 2>/dev/null; then
  for try in 1 2 3; do
    pip install -q --default-timeout=120 "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib lpips vastai 2>&1 | tail -1 | tee -a $LOG
    pip install -q --default-timeout=120 --only-binary=:all: PyMCubes 2>&1 | tail -1 | tee -a $LOG
    cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
      curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
    pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -1 | tee -a $LOG
    python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization, simple_knn, mcubes, taichi; assert torch.cuda.is_available()" >> $LOG 2>&1 && break
    say "[pip] 시도 $try 실패"
  done
fi
python -c "import torch, cupy, h5py, warp, lpips, diff_gaussian_rasterization, mcubes, taichi" 2>/dev/null || { say "[세팅 실패] pip"; echo "pip 실패" | rclone rcat $R/lbours/nshape/$TAG.END; exit 1; }
W=/home/dkta/work; AF=/root/sv/af; AG=/dev/shm/dkta/ag_data; PGU=$W/PhysGaussian_upstream; mkdir -p $W/pgmodel $AF $AG /home/dkta/.conda/envs/af/bin /dev/shm/dkta/tmp
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
rclone copy $MI/PhysGaussian_upstream_8339ed6 $PGU --transfers 32 -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --exclude ".git/**" --transfers 32 -q &      # 렌더 (lb_render·rtrender1 의 GaussianModel 읽기) 전용 -- 정답 시뮬에는 안 씀
rclone copy $DS/model/$OBJ $W/pgmodel/$OBJ --transfers 16 -q &
rclone cat ${XS_CODE:?}/af_code.tar | tar -x -C $AF
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
wait
CFG=$W/cfg_$SC.json; cp $AF/exe/nshape_cfg/$SC.json $CFG; NF=$(python -c "import json;print(json.load(open('$CFG'))['frame_num'])")
say "[세팅 끝] PG 원본 $(ls $PGU | tr '\n' ' ') 모델 $(ls $W/pgmodel/$OBJ | tr '\n' ' ') 프레임 $NF"
[ -d $W/pgmodel/$OBJ/point_cloud ] || { say "[모델 없음]"; echo "모델 없음" | rclone rcat $R/lbours/nshape/$TAG.END; exit 1; }
export PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp
# ---- (2) 정답: 원본 PhysGaussian 그대로
O=$W/gt/$SC; mkdir -p $O
HAVE=$(rclone lsf $DS/gt_pg/$SC/simulation_ply 2>/dev/null | grep -c "h5$")
if [ "$HAVE" -ge $((NF + 1)) ]; then rclone copy $DS/gt_pg/$SC $O --transfers 16 -q; say "[정답] R2 에 있음 ($HAVE) -> 받음"; else
  t0=$(date +%s)
  (cd $PGU && WARP_CACHE_PATH=$W/wpc timeout 1800 python -u gs_simulation.py --model_path $W/pgmodel/$OBJ --config $CFG --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1); rc=$?
  echo "PhysGaussian 8339ed6 gs_simulation.py (원본 그대로)" > $O/RUNNER
  if [ $rc = 124 ] && ! grep -q "after dense grids" $O/run.log; then
    say "[정답] 채우기가 30 분 안에 안 끝남 (taichi f32 sym_eig) -> 사본에 exe/fix_pg_filling.py 적용 뒤 다시 (풀이기만 바꿈, 정의 그대로)"
    cp -r $PGU ${PGU}_fill; python $AF/exe/fix_pg_filling.py --pg ${PGU}_fill >> $O/fix.log 2>&1; rm -rf $O/simulation_ply
    (cd ${PGU}_fill && WARP_CACHE_PATH=$W/wpc timeout 5400 python -u gs_simulation.py --model_path $W/pgmodel/$OBJ --config $CFG --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1); rc=$?
    echo "PhysGaussian 8339ed6 gs_simulation.py + exe/fix_pg_filling.py (채우기 고유분해만 f64 토치)" > $O/RUNNER
  fi
  n=$(ls $O/simulation_ply 2>/dev/null | grep -c h5)
  say "[정답] rc=$rc 프레임 $n $(( $(date +%s) - t0 )) 초 $(grep -a 'Total particles\|after dense\|after internal' $O/run.log | tr '\n' ' ' | cut -c1-200) $(grep -a 'rror' $O/run.log | tail -1 | cut -c1-200)"
  cp $CFG $O/config.json
  [ $n -gt 1 ] && (cd $AF && python -u exe/measure_gf_tearing.py --h5_dir $O/simulation_ply --tag $SC --out $O/tear > $O/tear.log 2>&1; say "[정답 위상] $(grep -a '판정' $O/tear.log | cut -c1-200)")
  rclone copy $O $DS/gt_pg/$SC --transfers 16 -q; say "[정답] R2 올림 $DS/gt_pg/$SC"
  [ $n -ge $((NF + 1)) ] || { say "[정답 실패] 프레임 $n/$((NF + 1))"; echo "정답 실패" | rclone rcat $R/lbours/nshape/$TAG.END; exit 1; }
fi
cd $AF
python -u exe/pg_h5_to_pt.py --h5_dir $O/simulation_ply --config $CFG --out_pt $AG/$SC.pt --out_dir $AG/gt_$SC > $W/cv_$SC.log 2>&1 || { say "[변환 실패] $(tail -1 $W/cv_$SC.log)"; echo "변환 실패" | rclone rcat $R/lbours/nshape/$TAG.END; exit 1; }
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $W/BA_1.txt
# ---- (3) 우리 판
run1() { local K=$1 sfx=$2 drop=$3 nodump=$4; local D=$W/ours/${SC}_K${K}${sfx}; mkdir -p $D
  local CMD=$(NS_DROP="$drop" NS_NODUMP=$nodump python exe/nshape_cmd.py $W/BA_1.txt $AG/$SC.pt $NF $SC $D $AG/gt_$SC $AG/gt_$SC/config.json $K)
  echo "$CMD" > $D/cmd.txt; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 bash -c "$CMD" < /dev/null > $D/run.log 2>&1; local rc=$?
  say "  [$SC K=$K$sfx] rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $D/run.log | tail -1 | cut -c1-200) $(grep -a 'rror' $D/run.log | tail -1 | cut -c1-160)"
  rclone copy $D $RX/ours/${SC}_K${K}${sfx} -q --exclude "pred.pt" --transfers 8
  return $rc; }
KOK=""
for K in 1 2 4 8; do
  run1 $K "" "" 0; rc=$?
  if [ $rc != 0 ]; then say "  [$SC] K=$K 실행 오류 rc=$rc -> 멈춤"; break; fi
  if python -c "import json,sys;s=json.load(open('$W/ours/${SC}_K$K/summary.json'));sys.exit(0 if s.get('diverged') else 1)"; then say "  [$SC] K=$K 발산 -> 다음 K"; continue; fi
  KOK=$K; break
done
if [ -n "$KOK" ]; then
  D=$W/ours/${SC}_K$KOK
  run1 $KOK _t "--pdiag --qjd --fhash --hgt --bar_cnt" 1      # (4) 단독 시간 판: 카드에 이 판 하나, 진단·덤프 없음
  python -u exe/gfs_post.py topo --gt_pt $AG/$SC.pt --pred $D/pred.pt --out $D/topo.json > $D/topo.log 2>&1; say "  [$SC] $(tail -1 $D/topo.log | cut -c1-400)"
  # ---- (6) PG 렌더: 정답 h5 와 우리 (lb_roll_to_h5) 를 같은 lb_render (PG 3DGS 렌더, 카메라 = 모델 cameras.json 첫 장 x cam_back) 로
  CB=${NS_CAMBACK:-2.4}
  python -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $AG/gt_$SC --out $D/h5_ours > $D/h5.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim $AG/gt_$SC --config $CFG --shape $SC --out $D/gt.mp4 --cam_back $CB --png $D/gt_png < /dev/null > $D/r_gt.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim $D/h5_ours --config $CFG --shape $SC --out $D/ours.mp4 --cam_back $CB --png $D/ours_png < /dev/null > $D/r_ours.log 2>&1
  python -u exe/gfs_post.py vid --gt_label "정답 (PhysGaussian)" --gt_png $D/gt_png --our_png $D/ours_png --out_json $D/visual.json --out_mp4 $D/cmp_raw.mp4 --font $W/font.otf --fps 25 > $D/vid.log 2>&1
  ffmpeg -y -loglevel error -i $D/cmp_raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $D/nshape_${SC}_cmp.mp4
  say "  [$SC] 렌더 정답 $(ls $D/gt_png 2>/dev/null | wc -l) 우리 $(ls $D/ours_png 2>/dev/null | wc -l) $(tail -1 $D/vid.log | cut -c1-200) 영상 $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $D/nshape_${SC}_cmp.mp4 2>&1)"
  rclone copyto $D/nshape_${SC}_cmp.mp4 $VD/nshape_${SC}_cmp.mp4 -q
  rclone copy $D $RX/ours/${SC}_K$KOK -q --include "*.json" --include "*.log" --include "cmd.txt" --include "*.mp4"
  python - $D/summary.json $W/ours/${SC}_K${KOK}_t/summary.json $D/topo.json <<'PYX' | tee -a $LOG
import json, sys, os
s = json.load(open(sys.argv[1])); t = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
try: tp = json.load(open(sys.argv[3]))
except Exception: tp = {}
g, o = tp.get("gt", {}), tp.get("ours", {})
print("[요약] 완주", s.get("frames_done"), "/", s["args"]["frames"], "발산", s.get("diverged"), "오차 %", None if s.get("err_mean_done") is None else round(100 * s["err_mean_done"], 2),
      "ms 본 판", s.get("ms_mean"), "ms 단독", t.get("ms_mean"), "이탈 정답/우리", g.get("esc_max"), o.get("esc_max"), "조각 정답/우리", g.get("pieces_last"), o.get("pieces_last"))
PYX
fi
rclone copyto $LOG $RX/$TAG.log -q
BAD=0
for d in $(ls $W/ours); do rclone check $W/ours/$d $RX/ours/$d --one-way --size-only -q --exclude "pred.pt" --exclude "h5_ours/**" --exclude "*_png/**" --exclude "cmp_raw.mp4" --exclude "gt.mp4" --exclude "ours.mp4" 2>>/root/sv/check.txt || BAD=1; done
say "[끝] K=${KOK:-없음} R2 확인 $([ $BAD = 0 ] && echo 차이 0 || echo 차이 있음)"; echo "끝 $(date) K=${KOK:-none} check=$BAD" | rclone rcat $R/lbours/nshape/$TAG.END
if [ $BAD = 0 ] && [ -n "${CONTAINER_API_KEY:-}" ]; then vastai destroy instance ${CONTAINER_ID:-0} -y --api-key $CONTAINER_API_KEY 2>&1 | tee -a $LOG; fi
