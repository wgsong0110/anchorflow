#!/bin/bash
# 절 CM186 새 형상 (NeRF-synthetic chair·drums·hotdog·materials) x 위상 변화 재질: vast.ai 인스턴스 (A6000 1 장) 하나 = 장면 하나.
#   순서: (1) 3DGS 학습 (공식 3DGS train.py 기본값 30k, -w 흰 배경) (2) 정답 궤적 (PG 재질 = i-PhysGaussian 공식 러너, GF 유체 = GF 공식 러너)
#         (3) 우리 판 (exe/gfs_ours.sh = 공통 기본 H9 명령 + CM184 hexff + CM185 칸 상한; 발산하면 K 1->2->4->8) + 위상·시각 지표 + 비교 영상
#         (4) 단독 시간 판 (진단 플래그 뺌, 같은 카드 혼자) (5) R2 확인 뒤 스스로 destroy
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT, XS_TAG, XS_SC (장면 = exe/nshape_cfg/<XS_SC>.json), XS_OBJ (nerf_synthetic 물체), XS_SOLVER (pg|gf), XS_CODE (코드 묶음 R2 경로)
#   결과: 모델·정답 R2 datasets/anchorflow/nshape/{model,gt}, 우리 판 R2 result/anchorflow/lbours/nshape/ours/<장면>_K<K>(_t), 영상 result/anchorflow/nshape/vid/
set -u
TAG=${XS_TAG:-nshape}; SC=${XS_SC:?}; OBJ=${XS_OBJ:?}; SOLV=${XS_SOLVER:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/nshape; DS=r2:storage/datasets/anchorflow/nshape; VD=$R/nshape/vid; MI=$R/_vastmirror
mkdir -p /root/sv; LOG=/root/sv/$TAG.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?} 장면 $SC 물체 $OBJ 러너 $SOLV"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < 기본 $PD" | rclone rcat $RX/$TAG.POWERCAP; say "[전력 제한] $PL W < $PD W -- 일 안 함"; exit 1; fi
export HOME=/root
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
if ! python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization, simple_knn, mcubes, taichi" 2>/dev/null; then
  for try in 1 2 3; do
    pip install -q --default-timeout=120 "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib lpips vastai 2>&1 | tail -2 | tee -a $LOG
    pip install -q --default-timeout=120 --only-binary=:all: PyMCubes 2>&1 | tail -1 | tee -a $LOG
    cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
      curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
    pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
    python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization, simple_knn, mcubes, taichi; assert torch.cuda.is_available(); print('[pip] torch', torch.__version__)" >> $LOG 2>&1 && break
    say "[pip] 시도 $try 실패"
  done
fi
python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization, simple_knn" 2>/dev/null || { say "[세팅 실패] pip"; echo "pip 실패" | rclone rcat $RX/$TAG.END; exit 1; }
W=/root/gfs; AF=/root/sv/af; GF=$W/GaussianFluent; PG=$W/i-physgaussian; mkdir -p $W/hf/model $W/hf/config $AF
rclone copy $MI/GaussianFluent $GF --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q &
[ $SOLV = pg ] && rclone copy $MI/i-physgaussian $PG --exclude ".git/**" --transfers 32 -q &
rclone cat ${XS_CODE:?}/af_code.tar | tar -x -C $AF
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
wait
python $AF/exe/patch_gf_bg.py $GF/gs_simulation.py.orig $GF/gs_orig_run.py
CFG=$W/hf/config/$SC.json; cp $AF/exe/nshape_cfg/$SC.json $CFG; NF=$(python -c "import json;print(json.load(open('$CFG'))['frame_num'])")
say "[세팅 끝] GF $(ls $GF | wc -l) PG $(ls $PG 2>/dev/null | wc -l) 코드 $(ls $AF/exe | wc -l) 설정 $(ls -la $CFG | cut -c1-200) 프레임 $NF"
export PYTHONUTF8=1 OMP_NUM_THREADS=8
# ---- (1) 3DGS 학습: 공식 3DGS (GF 저장소에 든 Inria 2023 판 train.py, 기본 하이퍼파라미터 그대로 30000 반복), NeRF-synthetic 흰 배경 (-w)
MD=$W/hf/model/$OBJ
if rclone lsf $DS/model/$OBJ/point_cloud/iteration_30000/ 2>/dev/null | grep -q point_cloud.ply; then
  rclone copy $DS/model/$OBJ $MD --transfers 16 -q; say "[3DGS] R2 에 있음 -> 받음"
else
  rclone copy r2:storage/datasets/nerf_synthetic/$OBJ $W/data/$OBJ --transfers 32 -q
  python $AF/exe/patch_3dgs_train.py $GF/gaussian-splatting | tee -a $LOG      # 호환 고침만 (np.byte->uint8, 래스터라이저 반환 2 개)
  t0=$(date +%s)
  (cd $GF/gaussian-splatting && python -u train.py -s $W/data/$OBJ -m $MD -w --eval < /dev/null > $W/train_$OBJ.log 2>&1); rc=$?
  say "[3DGS] 학습 rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -a 'Evaluating test\|Evaluating train' $W/train_$OBJ.log | tail -2 | tr '\n' ' ' | cut -c1-300) $(grep -a 'rror' $W/train_$OBJ.log | tail -1 | cut -c1-200)"
  cp $W/train_$OBJ.log $MD/train.log 2>/dev/null
  [ -s $MD/point_cloud/iteration_30000/point_cloud.ply ] || { say "[3DGS 실패]"; rclone copy $W/train_$OBJ.log $RX/ -q; echo "3DGS 실패" | rclone rcat $RX/$TAG.END; exit 1; }
  rclone copy $MD $DS/model/$OBJ --transfers 16 -q; say "[3DGS] R2 올림 $DS/model/$OBJ (가우시안 $(python -c "from plyfile import PlyData;print(PlyData.read('$MD/point_cloud/iteration_30000/point_cloud.ply')['vertex'].count)"))"
fi
# ---- (2) 정답 궤적
O=$W/gtrun/$SC; mkdir -p $O
HAVE=$(rclone lsf $DS/gt/$SC/simulation_ply 2>/dev/null | grep -c "h5$")
if [ "$HAVE" -ge $((NF + 1)) ]; then say "[정답] R2 에 있음 ($HAVE)"; else
  t0=$(date +%s)
  if [ $SOLV = pg ]; then
    grep -q "AF_PR2_IMP" $PG/gs_simulation.py || sed -i '1i import os as _af_os, numpy as _af_np, torch as _af_t  # AF_PR2_IMP' $PG/gs_simulation.py      # exe/pr2_gt.sh 와 같은 import 결함 고침
    (cd $PG && AF_PGFILL_NPY=$O/fill.npy WARP_CACHE_PATH=$W/wpc PYTHONIOENCODING=utf-8 timeout 14400 python -u gs_simulation.py --model_path $MD --config $CFG --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1); rc=$?
    mkdir -p $O/simulation_ply; ls $O/sim_*.h5 >/dev/null 2>&1 && mv $O/sim_*.h5 $O/simulation_ply/
    echo "i-physgaussian gs_simulation.py" > $O/RUNNER
  else
    (cd $GF && PYTHONPATH=$GF:$GF/gaussian-splatting timeout 14400 python -u gs_orig_run.py --model_path $MD --config $CFG --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1); rc=$?
    echo "GaussianFluent gs_simulation.py.orig (배경 읽기만 감쌈)" > $O/RUNNER
  fi
  n=$(ls $O/simulation_ply 2>/dev/null | grep -c h5)
  say "[정답] rc=$rc 프레임 $n $(( $(date +%s) - t0 )) 초 입자 $(grep -a -i 'Total particles\|particles' $O/run.log | tail -1 | cut -c1-120) $(grep -a 'rror' $O/run.log | tail -1 | cut -c1-200)"
  cp $CFG $O/config.json
  [ $n -gt 1 ] && (cd $AF && python -u exe/measure_gf_tearing.py --h5_dir $O/simulation_ply --tag $SC --out $O/tear > $O/tear.log 2>&1; say "[정답 위상] $(tail -3 $O/tear.log | tr '\n' ' ' | cut -c1-400)")
  rclone copy $O $DS/gt/$SC --transfers 16 -q --exclude "fill.npy"; say "[정답] R2 올림 $DS/gt/$SC"
  [ $n -ge $((NF + 1)) ] || { say "[정답 실패] 프레임 $n/$((NF + 1))"; echo "정답 실패" | rclone rcat $RX/$TAG.END; exit 1; }
fi
rclone cat ${XS_CODE}/af_code.tar | tar -x -C $AF      # 절 CM186: 우리 판 직전에 코드 묶음 다시 (학습·정답 도중 공통 수정이 올라왔으면 반영)
# ---- (3) 우리 판 (발산하면 K 1 -> 2 -> 4 -> 8, 사용자 11-15 규칙)
export W AF DS R RX LOG TAG GF; export -f say; mkdir -p $W/ag
export GFS_VIDDIR=$VD; [ $SOLV = pg ] && export GFS_KEEP_DT=1 GFS_GTLABEL="정답 (PhysGaussian)"
KOK=""
for K in 1 2 4 8; do last=""; [ $K = 8 ] && last=1
  XS_SWEEP=1 XS_SWEEP_LAST=$last XS_SFX=_K$K bash $AF/exe/gfs_ours.sh $SC $OBJ $SC.json $NF "--K=$K"; rc=$?
  say "  [$SC] K=$K rc=$rc"; [ $rc = 3 ] && continue
  KOK=$K; break
done
# ---- (4) 단독 시간 판: 같은 명령에서 진단 플래그만 뺌 (절 CM185 와 같은 목록), 카드에 이 판 하나
if [ -n "$KOK" ]; then
  GFS_DROP="--pdiag --qjd --fhash --hgt --bar_cnt" XS_SFX=_K${KOK}_t bash $AF/exe/gfs_ours.sh $SC $OBJ $SC.json $NF "--K=$KOK"; say "  [$SC] 단독 시간 판 rc=$?"
  python - $W/ours/${SC}_K${KOK}/run/summary.json $W/ours/${SC}_K${KOK}_t/run/summary.json $W/ours/${SC}_K${KOK}/topo.json <<'PYX' | tee -a $LOG
import json, sys
s = json.load(open(sys.argv[1])); t = json.load(open(sys.argv[2])) if __import__("os").path.exists(sys.argv[2]) else {}
try: tp = json.load(open(sys.argv[3]))
except Exception: tp = {}
print("[요약] 완주", s.get("frames_done"), "/", s["args"]["frames"], "발산", s.get("diverged"), "오차 %", None if s.get("err_mean_done") is None else round(100 * s["err_mean_done"], 2),
      "ms (본 판)", s.get("ms_mean"), "ms (단독 시간 판)", t.get("ms_mean"), "위상 정답", {k: v for k, v in tp.get("gt", {}).items() if k != "series"}, "우리", {k: v for k, v in tp.get("ours", {}).items() if k != "series"})
PYX
fi
rclone copyto $LOG $RX/$TAG.log -q
# ---- (5) R2 확인 뒤 destroy
BAD=0
for d in $(ls $W/ours); do rclone check $W/ours/$d/run $RX/ours/$d --one-way --size-only -q --exclude "pred.pt" --exclude "pred_v.pt" 2>>/root/sv/check.txt || BAD=1; done
say "[끝] K=${KOK:-없음} R2 확인 $([ $BAD = 0 ] && echo 차이 0 || echo 차이 있음)"; echo "끝 $(date) K=${KOK:-none} check=$BAD" | rclone rcat $RX/$TAG.END
if [ $BAD = 0 ] && [ -n "${CONTAINER_API_KEY:-}" ]; then vastai destroy instance ${CONTAINER_ID:-0} -y --api-key $CONTAINER_API_KEY 2>&1 | tee -a $LOG; fi
