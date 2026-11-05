#!/bin/bash
# 절 CM13 영상 보충 (vast.ai A6000 1 장, onstart -> tmux wmjv; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, CONTAINER_ID·CONTAINER_API_KEY):
#   묶음 wmJ 의 렌더가 GaussianFluent particle_filling 의 mcubes (PyMCubes) 없음으로 실패 -> 같은 덤프 (R2 lbours/fem_cm/wmJ) 로 다시 렌더·합성만 (시뮬 다시 안 돌림).
#   칸 고르는 규칙: 판마다 완주한 판 a 먼저, 없으면 b, 둘 다 발산이면 더 오래 간 판 -> (1) J 장벽 = j_b (x73, j_a x23), (2) SNH = s_a (완주).
#   발산한 판 칸은 발산 전 마지막 유한 프레임에서 멈춤 + 표시. 영상 정답 | (1) | (2) -> result/anchorflow/v20261009/fem_wmJ_cmp.mp4.
# 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 marks/wmJv_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/wv
LOG=/root/wv/wmJv.log
say() { echo "[wmJv $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/wmJv.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/wmJv.END; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,memory.total --format=csv | tee -a $LOG
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/wv/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  pip install -q --only-binary=:all: PyMCubes==0.1.6 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, h5py, mcubes, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'mcubes ok')" 2>&1 | tail -1 | tee -a $LOG && touch /root/wv/pip.done
fi
[ -f /root/wv/pip.done ] || fail "[세팅 실패] pip"
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/wmJ; RR=$R/lbours/fem_cm/wmJ
W=/home/dkta/work; MI=$R/_vastmirror
PJ=j_b; PS=s_a
for r in $PJ $PS; do rclone copy $RR/$r $B/$r -q --transfers 8 --include "pred.pt" --include "summary.json"; done
rclone copy $MI/gfmodel/model/watermelon $W/gfmodel/model/watermelon --transfers 16 -q
rclone copy $MI/GaussianFluent $W/GaussianFluent --transfers 32 -q
rclone copyto $MI/wm2/gf/config.json $W/wm2/gf/config.json -q
rclone copy $MI/wm2/gf/simulation_ply $W/wm2/gf/simulation_ply --transfers 16 -q
TF=/dev/shm/dkta/wmjcmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wm/gt/gt_watermelon.mp4 $TF/gt.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/font.otf -q
(cd $W/GaussianFluent && $PY -c "import sys; sys.path.insert(0, '.'); sys.path.insert(1, 'gaussian-splatting'); from utils.decode_param import *; from utils.transformation_utils import *; from utils.render_utils import *; print('GF import ok')") > /root/wv/gfimp.log 2>&1
say "[자료] $(ls -la $B/$PJ/pred.pt $B/$PS/pred.pt 2>&1 | awk '{print $5}' | tr '\n' ' ') 정답 h5 $(ls $W/wm2/gf/simulation_ply | wc -l) 영상 $(ls $TF | tr '\n' ' ') GF $(tail -1 /root/wv/gfimp.log)"
[ -s $B/$PJ/pred.pt ] && [ -s $B/$PS/pred.pt ] && grep -q "GF import ok" /root/wv/gfimp.log || fail "[자료 실패] $(tail -3 /root/wv/gfimp.log | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
CFG=$W/wm2/gf/config.json; WMM=$W/gfmodel/model/watermelon; GTD=/dev/shm/dkta/ag_data/gt_wm; mkdir -p /dev/shm/dkta/ag_data
[ -f $GTD/meta.json ] || $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt
rend1() { local r=$1; local O=$B/$r; mkdir -p $O/vj
  $PY - $O/pred.pt $O/pred_v.pt <<'PYX' > $O/vj/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
ok = [i for i in range(P.shape[0]) if bool(torch.isfinite(P[i]).all()) and bool(torch.isfinite(F[i]).all())]
n = 0
while n in ok: n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred_v.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=0 $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --out $O/vj/fem_watermelon.mp4 < /dev/null > $O/vj/r_fem.log 2>&1)
  rm -f $O/pred_v.pt
  say "  [렌더] $r $(tail -1 $O/vj/pad.log) $(ls -la $O/vj/fem_watermelon.mp4 2>&1 | cut -c1-160)"; }
rend1 $PJ; rend1 $PS
lab() { $PY -c "import json; s=json.load(open('$B/$1/summary.json')); dv=s.get('diverged'); print('$2 $1 (%s, 오차 %.2f %%%%)' % ('완주' if dv is None else '프레임 %d 발산 -- 그 뒤 멈춤' % dv, 100*s['err_mean_done']))"; }
LJ=$(lab $PJ "(1) J 로그 장벽"); LS=$(lab $PS "(2) Stable Neo-Hookean")
H=$($PY -c "import imageio.v3 as iio; print(iio.improps('$TF/gt.mp4', plugin='pyav').shape[1])" 2>/dev/null || echo 720)
if [ -f $B/$PJ/vj/fem_watermelon.mp4 ] && [ -f $B/$PS/vj/fem_watermelon.mp4 ]; then
  ffmpeg -y -loglevel error -i $TF/gt.mp4 -i $B/$PJ/vj/fem_watermelon.mp4 -i $B/$PS/vj/fem_watermelon.mp4 -filter_complex \
    "[0:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='정답':x=20:y=20:fontsize=34:fontcolor=black[a];[1:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='$LJ':x=20:y=20:fontsize=28:fontcolor=black[b];[2:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='$LS':x=20:y=20:fontsize=28:fontcolor=black[c];[a][b][c]hstack=inputs=3:shortest=1" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 $TF/fem_wmJ_cmp.mp4 > $TF/ff.log 2>&1
fi
for r in $PJ $PS; do rclone copy $B/$r/vj $RR/$r/vj -q; rclone copy $B/$r/h5_fem.log $RR/$r -q; done
if [ -s $TF/fem_wmJ_cmp.mp4 ]; then rclone copyto $TF/fem_wmJ_cmp.mp4 $RR/fem_wmJ_cmp.mp4 -q; rclone copyto $TF/fem_wmJ_cmp.mp4 $R/v20261009/fem_wmJ_cmp.mp4 -q && say "[영상] result/anchorflow/v20261009/fem_wmJ_cmp.mp4 ($PJ | $PS)"
else say "[영상 실패] $(for q in /dev/shm/dkta/ag_data/cv_wm_rend.log $B/$PJ/h5_fem.log $B/$PJ/vj/r_fem.log $B/$PS/h5_fem.log $B/$PS/vj/r_fem.log $TF/ff.log; do [ -f $q ] && tail -n 3 $q; done | tr '\n' ' ' | cut -c1-900)"; fi
echo "[wmJv 끝] $(date '+%F %T')" | rclone rcat $MR/wmJv.END
say "[wmJv 끝]"
if rclone check $B/$PJ/vj $RR/$PJ/vj --one-way --size-only -q 2>/root/wv/check.txt && rclone check $B/$PS/vj $RR/$PS/vj --one-way --size-only -q 2>>/root/wv/check.txt && { [ ! -s $TF/fem_wmJ_cmp.mp4 ] || rclone check $TF $R/v20261009 --one-way --size-only --include fem_wmJ_cmp.mp4 -q 2>>/root/wv/check.txt; }; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/wv/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/wmJv_CHECK_FAIL
fi
