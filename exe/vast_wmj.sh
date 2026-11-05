#!/bin/bash
# 절 CM13 묶음 wmJ (vast.ai A6000 1~2 장 인스턴스 onstart -> tmux wmj; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   수박 cm9 w1a 명령 그대로 (exe/fem_cm9_scn.sh wm + --gq_mass --gq_chk --dump) + 정의 변경 하나 (사용자 승인 11-05):
#     판 (1) --elx jbar (가우스 점 J 로그 장벽 κ_J V_q (−log J + J − 1), κ_J = μ),  판 (2) --elx snh (탄성 = Stable Neo-Hookean, RM 그대로)
#   진단 (결과 불변): --nan_step (발산 프레임 걸음·반복별 처음 비유한), --jdiag (프레임 끝 F_q·입자 F det).
#   비결정이라 판마다 두 번 (j_a·j_b, s_a·s_b), 그 뒤 판마다 단독 시간 (zj·zs: 진단·덤프 없이, 그 카드에 판 하나만).
#   기준 (끈 판) = 절 CM9 cm9 w1a·w1b (R2 재사용; w1a 덤프 pred.pt 에서 프레임별 입자 F det 최소만 여기서 잼).
#   완주한 판이 있으면 영상: 정답 | 판 (1) | 판 (2) (3DGS, GF wm_render; 발산한 판 칸은 발산 전 마지막 프레임에서 멈춤 + 표시).
# 출력 R2 lbours/fem_cm/wmJ/<판>, 표 lbours/fem_cm/marks/wmJ_table.md, 로그 marks/wmJ.log, 끝 표식 marks/wmJ.END, 영상 result/anchorflow/v20261009/fem_wmJ_cmp.mp4.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/wmJ_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/wj
LOG=/root/wj/wmJ.log
say() { echo "[wmJ $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/wmJ.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/wmJ.END; exit 1; }
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
if [ ! -f /root/wj/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/wj/pip.done
fi
[ -f /root/wj/pip.done ] || fail "[세팅 실패] pip"
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/wmJ; RR=$R/lbours/fem_cm/wmJ; M=/dev/shm/dkta/lbours/fem_cm/marks
C9=/dev/shm/dkta/lbours/fem_cm/cm9
mkdir -p $B $M $C9
# ---- 자료 (GPU 일 전에 모두): 수박 정답 wm.pt, 기준 cm9 w1a·w1b, 영상용 3DGS·GF·정답 h5·정답 영상·글꼴
W=/home/dkta/work; MI=$R/_vastmirror
export CM_SCN=exe/fem_cm9_scn.sh
set +u
source exe/fem_cm9_scn.sh
cm_data wm
for v in w1a_wm w1b_wm; do rclone copy $R/lbours/fem_cm/cm9/$v $C9/$v -q --transfers 8; done
rclone copy $MI/gfmodel/model/watermelon $W/gfmodel/model/watermelon --transfers 16 -q
rclone copy $MI/GaussianFluent $W/GaussianFluent --transfers 32 -q
rclone copyto $MI/wm2/gf/config.json $W/wm2/gf/config.json -q
rclone copy $MI/wm2/gf/simulation_ply $W/wm2/gf/simulation_ply --transfers 16 -q
TF=/dev/shm/dkta/wmjcmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wm/gt/gt_watermelon.mp4 $TF/gt.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/font.otf -q
say "[자료] wm.pt $(ls -la $WMD/wm.pt 2>&1 | awk '{print $5}') cm9 $(ls $C9/w1a_wm | tr '\n' ' ') 3DGS $(du -sh $W/gfmodel/model/watermelon $W/GaussianFluent 2>&1 | awk '{print $1}' | tr '\n' ' ') 정답 h5 $(ls $W/wm2/gf/simulation_ply | wc -l) 영상 $(ls $TF | tr '\n' ' ')"
[ -s $WMD/wm.pt ] && [ -s $C9/w1a_wm/fem.jsonl ] || fail "[자료 실패] $(ls -la $WMD 2>&1 | tr '\n' ' ') $(ls $C9/w1a_wm 2>&1 | tr '\n' ' ')"
NG=$(nvidia-smi -L | wc -l); [ $NG -gt 2 ] && NG=2
say "[준비 끝] GPU 일 시작 (카드 $NG 장)"
table() { $PY exe/fem_wmj_table.py $B $C9 $B/base_det.json > $M/wmJ_table.md.tmp 2>$M/wmJ_table.err
  { echo "# wmJ ($C) -- 절 CM13 수박 J 로그 장벽 (1) / Stable Neo-Hookean (2) ($(date '+%F %T') UTC 기준)"; echo; cat $M/wmJ_table.md.tmp; } > $M/wmJ_table.md; rclone copy $M $MR -q --include "wmJ_table.*"; }
( while [ ! -f /root/wj/runs.done ]; do sleep 300; table; done ) &
PT=$!
# 판 하나: $1 = 이름, $2 = 카드, 나머지 = 목록 인자. z* 는 단독 시간 (그 카드 다른 계산 프로세스 수를 10 초마다 셈)
run1() { local v=$1 cd=$2; shift 2; local O=$B/$v; mkdir -p $O
  export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmj_$cd; mkdir -p $CUPY_CACHE_DIR
  cm_scene wm "$@"
  local UU=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c=$cd '$1==c {print $2}')
  echo "카드 $cd ($UU) $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UU" > $O/apps_at_start.txt
  CUDA_VISIBLE_DEVICES=$cd timeout 3600 $PY -u $SIM --out $O --tag wm $DA $XA < /dev/null > $O/log.txt 2>&1 &
  local PP=$! mx=0
  while kill -0 $PP 2>/dev/null; do sleep 10
    local me=$(pgrep -P $PP | head -1); local n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$UU" | grep -vc ", ${me:-x}$"); [ "$n" -gt "$mx" ] && mx=$n; done
  wait $PP; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt; echo $mx > $O/apps_during.txt
  say "  $v 카드 $cd $(tail -1 $O/run_info.txt) 그 카드 다른 프로세스 최대 $mx $(grep -a '절 CM13\]\|롤아웃\|rror\|발산\|CM9.3 덤프' $O/log.txt | tail -4 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O $RR/$v -q --transfers 8; }
X="--gq_mass --gq_chk --dump --nan_step --jdiag"
if [ $NG -ge 2 ]; then
  ( run1 j_a 0 $X --elx jbar; run1 j_b 0 $X --elx jbar; run1 zj 0 --gq_mass --elx jbar ) &
  P0=$!; sleep 20
  ( run1 s_a 1 $X --elx snh; run1 s_b 1 $X --elx snh; run1 zs 1 --gq_mass --elx snh ) &
  P1=$!; wait $P0 $P1
else
  run1 j_a 0 $X --elx jbar; run1 s_a 0 $X --elx snh; run1 j_b 0 $X --elx jbar; run1 s_b 0 $X --elx snh
  run1 zj 0 --gq_mass --elx jbar; run1 zs 0 --gq_mass --elx snh
fi
# 기준 w1a 덤프 -> 프레임별 입자 F det 최소·≤ 0 수 (끈 판 비교 열)
CUDA_VISIBLE_DEVICES=0 $PY - $C9/w1a_wm/pred.pt $B/base_det.json <<'PYX' > $B/base_det.log 2>&1
import json, sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); F = d["F"]
o = dict(p_min=[], p_le0=[])
for i in range(F.shape[0]):
    dt = torch.linalg.det(F[i].cuda().double()); fin = torch.isfinite(dt)
    o["p_min"].append(float(dt[fin].min()) if bool(fin.any()) else None); o["p_le0"].append(int((dt <= 0).sum()))
json.dump(dict(w1a=o), open(sys.argv[2], "w")); print("기준 w1a 입자 det", o["p_min"][-3:], o["p_le0"][-3:])
PYX
say "[기준 det] $(tail -1 $B/base_det.log)"
touch /root/wj/runs.done; kill $PT 2>/dev/null
table
# ---- 영상: 정답 | 판 (1) | 판 (2) -- 완주한 판 (a 먼저, 없으면 b) 이 하나라도 있으면. 발산한 판은 발산 전 마지막 프레임에서 멈춘 칸
pick() { local p=$1; for r in ${p}_a ${p}_b; do $PY -c "import json,sys; s=json.load(open('$B/$r/summary.json')); sys.exit(0 if s.get('diverged') is None and s.get('frames_done') == 100 else 1)" 2>/dev/null && { echo $r; return; }; done; echo ""; }
PJ=$(pick j); PS=$(pick s)
say "[완주 판] (1) ${PJ:-없음} (2) ${PS:-없음}"
if [ -n "$PJ$PS" ]; then
  [ -n "$PJ" ] || PJ=j_a; [ -n "$PS" ] || PS=s_a
  CFG=$W/wm2/gf/config.json; WMM=$W/gfmodel/model/watermelon; GTD=/dev/shm/dkta/ag_data/gt_wm; mkdir -p /dev/shm/dkta/ag_data
  [ -f $GTD/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt
  rend1() { local r=$1 cd=$2; local O=$B/$r; mkdir -p $O/vj
    # 발산한 판: 발산 전 마지막 유한 프레임까지 + 그 프레임을 100 프레임까지 되풀이 (멈춤 칸)
    $PY - $O/pred.pt $O/pred_v.pt <<'PYX' > $O/vj/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
ok = [i for i in range(P.shape[0]) if bool(torch.isfinite(P[i]).all()) and bool(torch.isfinite(F[i]).all())]
n = (max(ok) + 1) if ok else 0
while n > 0 and n - 1 not in ok: n -= 1
L = 100
if n < L:
    P = torch.cat([P[:n], P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F[:n], F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
    $PY -u exe/lb_roll_to_h5.py --dump $O/pred_v.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
    (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$cd $PY -u $D/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --out $O/vj/fem_watermelon.mp4 < /dev/null > $O/vj/r_fem.log 2>&1)
    rm -f $O/pred_v.pt
    say "  [렌더] $r $(cat $O/vj/pad.log | tail -1) $(ls -la $O/vj/fem_watermelon.mp4 2>&1 | cut -c1-160)"; }
  rend1 $PJ 0 & sleep 5; rend1 $PS $(( NG - 1 )) & wait
  lab() { $PY -c "import json; s=json.load(open('$B/$1/summary.json')); dv=s.get('diverged'); print('$2 (%s, 오차 %.2f %%%%)' % ('완주' if dv is None else '프레임 %d 발산 -- 그 뒤 멈춤' % dv, 100*s['err_mean_done']))"; }
  LJ=$(lab $PJ "(1) J 로그 장벽"); LS=$(lab $PS "(2) Stable Neo-Hookean")
  FF=$(command -v ffmpeg)
  H=$($PY -c "import imageio.v3 as iio; print(iio.improps('$TF/gt.mp4', plugin='pyav').shape[1])" 2>/dev/null || echo 720)
  if [ -f $B/$PJ/vj/fem_watermelon.mp4 ] && [ -f $B/$PS/vj/fem_watermelon.mp4 ]; then
    $FF -y -loglevel error -i $TF/gt.mp4 -i $B/$PJ/vj/fem_watermelon.mp4 -i $B/$PS/vj/fem_watermelon.mp4 -filter_complex \
      "[0:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='정답':x=20:y=20:fontsize=34:fontcolor=black[a];[1:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='$LJ':x=20:y=20:fontsize=30:fontcolor=black[b];[2:v]scale=-2:$H,drawtext=fontfile=$TF/font.otf:text='$LS':x=20:y=20:fontsize=30:fontcolor=black[c];[a][b][c]hstack=inputs=3:shortest=1" \
      -c:v libx264 -pix_fmt yuv420p -crf 20 $TF/fem_wmJ_cmp.mp4 > $TF/ff.log 2>&1
  fi
  if [ -s $TF/fem_wmJ_cmp.mp4 ]; then cp $TF/fem_wmJ_cmp.mp4 $B/; rclone copyto $TF/fem_wmJ_cmp.mp4 $R/v20261009/fem_wmJ_cmp.mp4 -q && say "[영상] result/anchorflow/v20261009/fem_wmJ_cmp.mp4 ($PJ | $PS)"
  else say "[영상 실패] $(for q in /dev/shm/dkta/ag_data/cv_wm_rend.log $B/$PJ/h5_fem.log $B/$PJ/vj/r_fem.log $B/$PS/h5_fem.log $B/$PS/vj/r_fem.log $TF/ff.log; do [ -f $q ] && tail -n 3 $q; done | tr '\n' ' ' | cut -c1-900)"; fi
else
  say "[영상 안 함] 완주한 판 없음"
fi
table
echo "[wmJ 끝] $(date '+%F %T')" > $M/wmJ.END; rclone copy $M/wmJ.END $MR -q
say "[wmJ 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8 --exclude "h5_fem/**"
if rclone check $B $RR --one-way --size-only -q --exclude "h5_fem/**" 2>/root/wj/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/wj/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/wmJ_CHECK_FAIL
fi
