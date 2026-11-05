#!/bin/bash
# 절 CM13.4 묶음 wmJd -- 수박 손상 변수 (NACC 경화 logJp) 색 렌더 (vast.ai A6000 1 장, onstart -> tmux wmd; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, CONTAINER_ID·CONTAINER_API_KEY):
#   (가) 우리 (2) SNH = 절 CM13 s_a 명령 그대로 + --dump_lj (결과 불변, 프레임 끝 입자 logJp -> lj.pt)
#   (나) 정답 = GF 공식 러너 gs_simulation_watermelon.py 를 같은 config·3DGS 로 다시 + 프레임별 particle_Jp 덤프 (exe/gf_wm_jp.py, GF 소스 수정 없음);
#       원래 정답 h5 와 위치 차를 통계에 기록 (정답 데이터는 바꾸지 않음 -- 다시 돌린 궤적은 손상 색 렌더·곡선에만)
#   (다) 통계 exe/wmd_stats.py, 단면 exe/wmd_slices.py, 손상 색 렌더 wm_render.py --scalar_dir (viridis, logJp -0.2..0.4 공통 범위) x 정답·우리 + 정답 다시 일반 렌더
# 출력 R2 lbours/fem_cm/wmJd, 로그 marks/wmJd.log, 끝 표식 marks/wmJd.END. 합성 영상·곡선·뷰어는 로컬에서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 marks/wmJd_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/wd
LOG=/root/wd/wmJd.log
say() { echo "[wmJd $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/wmJd.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/wmJd.END; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
free -g | head -2 | tee -a $LOG; df -h /workspace | tail -1 | tee -a $LOG
# ---- 클러스터 경로 그대로 (exe/vast_nstep.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/wd/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib scikit-learn vastai 2>&1 | tail -2 | tee -a $LOG; pip install -q --only-binary=:all: PyMCubes==0.1.6 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/wd/pip.done
fi
[ -f /root/wd/pip.done ] || fail "[세팅 실패] pip"
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
# ---- 자료 (GPU 일 전에 모두)
W=/home/dkta/work; MI=$R/_vastmirror
export CM_SCN=exe/fem_cm9_scn.sh
set +u
source exe/fem_cm9_scn.sh
cm_data wm
rclone copy $MI/gfmodel/model/watermelon $W/gfmodel/model/watermelon --transfers 16 -q
rclone copy $MI/GaussianFluent $W/GaussianFluent --transfers 32 -q
rclone copyto $MI/wm2/gf/config.json $W/wm2/gf/config.json -q
rclone copy $MI/wm2/gf/simulation_ply $W/wm2/gf/simulation_ply --transfers 16 -q
B=/dev/shm/dkta/lbours/fem_cm/wmJd; RR=$R/lbours/fem_cm/wmJd; M=/dev/shm/dkta/lbours/fem_cm/marks; mkdir -p $B $M
CFG=$W/wm2/gf/config.json; WMM=$W/gfmodel/model/watermelon; GTD=/dev/shm/dkta/ag_data/gt_wm; mkdir -p /dev/shm/dkta/ag_data
[ -f $GTD/meta.json ] || $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt
# GF 러너 import 연기 (--help: import 만 하고 끝남) -- 빠진 모듈은 바이너리 휠로 (최대 6 번)
GFR=gs_simulation/watermelon/gs_simulation_watermelon.py
for k in 1 2 3 4 5 6; do
  (cd $W/GaussianFluent && GF_MODEL_ROOT=$W/GaussianFluent/model $PY $D/exe/gf_wm_jp.py --jp_out /tmp/jpx -- $GFR --help) > /root/wd/gfhelp.log 2>&1 && break
  mm=$(grep -o "No module named '[^']*'" /root/wd/gfhelp.log | head -1 | sed "s/No module named '\([^'.]*\).*/\1/")
  [ -n "$mm" ] || break
  case $mm in cv2) pk=opencv-python-headless;; sklearn) pk=scikit-learn;; mcubes) pk=PyMCubes==0.1.6;; PIL) pk=pillow;; *) pk=$mm;; esac
  say "  [GF import] $mm 없음 -> $pk"; pip install -q --only-binary=:all: $pk 2>&1 | tail -1 | tee -a $LOG
done
say "[자료] wm.pt $(ls -la $WMD/wm.pt 2>&1 | awk '{print $5}') 3DGS $(du -sh $WMM $W/GaussianFluent 2>&1 | awk '{print $1}' | tr '\n' ' ') 정답 h5 $(ls $W/wm2/gf/simulation_ply | wc -l) GTD $(ls $GTD | wc -l) GF --help: $(tail -2 /root/wd/gfhelp.log | tr '\n' ' ' | cut -c1-200)"
[ -s $WMD/wm.pt ] && grep -q "output_h5" /root/wd/gfhelp.log && [ -f $GTD/meta.json ] || fail "[자료 실패] $(tail -5 /root/wd/gfhelp.log | tr '\n' ' ' | cut -c1-600)"
say "[준비 끝] GPU 일 시작"
# ---- (가) 우리 (2) SNH: 절 CM13 s_a 명령 그대로 + --dump_lj (결과 불변 출력만)
O=$B/s_d; mkdir -p $O
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmd; mkdir -p $CUPY_CACHE_DIR
cm_scene wm --gq_mass --gq_chk --dump --nan_step --jdiag --elx snh --dump_lj
echo "$(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
if [ -s $O/lj.pt ] && [ -s $O/pred.pt ]; then say "  [우리] 이미 있음 (이어 돌리기) $(grep -a '롤아웃' $O/log.txt | tail -1 | cut -c1-200)"; else
CUDA_VISIBLE_DEVICES=0 timeout 3600 $PY -u $SIM --out $O --tag wm $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt; fi
say "  [우리] $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|발산\|rror' $O/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
rclone copy $O $RR/s_d -q --transfers 8 &
# ---- (나) 정답 GF 공식 러너 다시 (같은 config·러너·3DGS) + 프레임별 logJp 덤프
G=/workspace/gfn; mkdir -p $G
if [ "$(ls $G/jp 2>/dev/null | wc -l)" = 101 ] && grep -q "^rc=0" $G/gf.log 2>/dev/null; then say "  [정답 다시] 이미 있음 (이어 돌리기)"; else
(cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=0 GF_MODEL_ROOT=$W/GaussianFluent/model timeout 7200 $PY -u $D/exe/gf_wm_jp.py --jp_out $G/jp -- $GFR --model_path $WMM --config $CFG --output_path $G/out --output_h5 < /dev/null > $G/gf.log 2>&1); echo "rc=$?" >> $G/gf.log; fi
say "  [정답 다시] $(tail -1 $G/gf.log) h5 $(ls $G/out/simulation_ply 2>/dev/null | wc -l) jp $(ls $G/jp 2>/dev/null | wc -l) $(grep -a '^\[jp\]' $G/gf.log | tail -1)"
wait
# ---- (다) 통계 + 렌더용 궤적 (GPU 렌더 둘 동시에)
$PY -u exe/wmd_stats.py --ours_lj $O/lj.pt --ours_pred $O/pred.pt --ours_sc $O/sc --gt_jp $G/jp --gt_new $G/out/simulation_ply --gt_orig $W/wm2/gf/simulation_ply --data $WMD/wm.pt --out $B/wmd_stats.json > $B/wmd_stats.log 2>&1
say "  [통계] $(tail -1 $B/wmd_stats.log | cut -c1-300)"
$PY -u exe/wmd_slices.py --ours_pred $O/pred.pt --ours_sc $O/sc --x0_h5 $GTD/sim_0000000000.h5 --gt_h5 $G/out/simulation_ply --gt_sc $G/jp --out $B/wmd_slices.npz > $B/wmd_slices.log 2>&1
say "  [단면] $(tail -1 $B/wmd_slices.log | cut -c1-300)"
$PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
$PY -u exe/pg_h5_to_pt.py --h5_dir $G/out/simulation_ply --config $CFG --out_pt $G/gfn_tmp.pt --out_dir $G/rend --allow_nonfinite > $G/cv.log 2>&1; rm -f $G/gfn_tmp.pt
VR="--vmin -0.2 --vmax 0.4 --cmap viridis"
(cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=0 $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --scalar_dir $O/sc $VR --out $B/ours_lj.mp4 < /dev/null > $B/r_ours.log 2>&1) &
(cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=0 $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $G/rend --scalar_dir $G/jp $VR --out $B/gt_lj.mp4 < /dev/null > $B/r_gt.log 2>&1) &
(cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=0 $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $G/rend --out $B/gt_new_plain.mp4 < /dev/null > $B/r_gtp.log 2>&1) &
wait
say "  [렌더] $(ls -la $B/*.mp4 2>&1 | awk '{print $5, $9}' | tr '\n' ' ') $(tail -1 $B/r_ours.log $B/r_gt.log 2>&1 | tr '\n' ' ' | cut -c1-300)"
# ---- 올림: 결과 폴더 + 정답 다시 돌린 jp·log (h5 13 GB 는 올리지 않음 -- 위치 차는 통계에)
rm -rf $O/h5_fem $O/sc
cp $G/gf.log $G/cv.log $B/ 2>/dev/null; mkdir -p $B/gt_jp; cp $G/jp/*.npy $B/gt_jp/
rclone copy $B $RR -q --transfers 16
echo "[wmJd 끝] $(date '+%F %T')" > $M/wmJd.END; rclone copy $M/wmJd.END $MR -q
say "[wmJd 끝]"
if rclone check $B $RR --one-way --size-only -q 2>/root/wd/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/wd/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/wmJd_CHECK_FAIL
fi
