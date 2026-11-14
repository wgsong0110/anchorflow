#!/bin/bash
# 절 CM158: 정답 MPM (PhysGaussian 모래, wolf_config.json) 의 격자만 0.02 m 로 바꿔 돌려 높이가 우리 판처럼 높아지는지 본다.
#   판: ref (n_grid 200 = dx 0.01, dt 2e-5 -- 원래 설정, 정답 재현 확인) · g100 (n_grid 100 = dx 0.02, dt 4e-5 -- 정답 CFL 규칙 dt ∝ dx) · g100s (n_grid 100, dt 2e-5 -- 서브스텝 같은 쪽 대조)
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT, CM_RUNS (기본 ref,g100,g100s). 자료: R2 _vastmirror (i-physgaussian, pgmodel/wolf_whitebg-trained, PhysGaussian/config/wolf_config.json, topo/pg_wolf_sand).
#   결과: R2 lbours/wolfcv/cm158 (cmd·log·cfg·h5 통째로, 60 초마다 동기화, 아무것도 지우지 않음). 끝 표식 vast.END -- 로컬에서 보고 destroy.
set -u
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/cm158; MI=$R/_vastmirror
mkdir -p /root/sv /workspace/cm158; LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/vast.log -q
export HOME=/root
W=/workspace/cm158; mkdir -p $W/pgmodel $W/topo $W/cfg
apt-get install -y -qq libx11-6 libxext6 libxrender1 libgl1 libglib2.0-0 libsm6 libxi6 libxxf86vm1 >> $LOG 2>&1   # taichi 가 libX11 을 요구 (첫 띄움 실패 몫)
if [ ! -f /root/sv/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" PyMCubes h5py scikit-image cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, taichi, warp, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1 | tee -a $LOG && touch /root/sv/pip.done
fi &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q --exclude ".git/**" &
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/wolf_config.json -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copyto $R/lbours/wolfcv/cm158/code/cm158_an.py $W/cm158_an.py -q &
wait
say "[자료] pip $(ls /root/sv/pip.done 2>&1) i-pg $(ls $W/i-physgaussian | wc -l) 모델 $(ls $W/pgmodel/wolf_whitebg-trained | tr '\n' ' ') GT h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) cfg $(stat -c %s $W/wolf_config.json 2>&1)"
[ -f /root/sv/pip.done ] && [ -s $W/wolf_config.json ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -s $W/cm158_an.py ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast.END; exit 1; }
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
# 설정 = wolf_config.json 에서 material n_grid · substep_dt 만 바꿈 (particle_filling.n_grid 는 그대로 -> 입자 채움 같음, 채움은 판끼리 캐시 공유)
python - <<PY
import json
b = json.load(open("$W/wolf_config.json"))
for n, ng, dt in (("ref", 200, 2e-5), ("g100", 100, 4e-5), ("g100s", 100, 2e-5)):
    c = json.loads(json.dumps(b)); c["n_grid"] = ng; c["substep_dt"] = dt
    assert c["particle_filling"]["n_grid"] == 100
    assert abs(c["frame_dt"] / dt - round(c["frame_dt"] / dt)) < 1e-9
    json.dump(c, open("$W/cfg/%s.json" % n, "w"), indent=4)
PY
for n in ref g100 g100s; do diff <(python -c "import json;print(json.dumps(json.load(open('$W/wolf_config.json')),indent=1,sort_keys=True))") <(python -c "import json;print(json.dumps(json.load(open('$W/cfg/$n.json')),indent=1,sort_keys=True))") | tr '\n' ' ' | tee -a $LOG; echo | tee -a $LOG; done
O=$W/out; mkdir -p $O; cp -r $W/cfg $O/cfg
( while sleep 60; do rclone copy $O $RX -q --transfers 8; done ) & WP=$!
run() { local n=$1; mkdir -p $O/$n
  local cmd="cd $W/i-physgaussian && AF_PGFILL_NPY=$O/fill.npy WARP_CACHE_PATH=$W/wpc_$n python -u gs_simulation.py --model_path $W/pgmodel/wolf_whitebg-trained --config $W/cfg/$n.json --output_path $O/$n --output_h5"
  echo "$cmd" > $O/$n/cmd.txt; say "$n 시작: $cmd"
  CUDA_VISIBLE_DEVICES=0 timeout 14400 bash -c "$cmd" < /dev/null > $O/$n/log.txt 2>&1
  say "$n rc=$? h5 $(ls $O/$n/simulation_ply 2>/dev/null | wc -l) 개 $(grep -a 'rror\|Traceback' $O/$n/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"
  rclone copy $O/$n $RX/$n -q --transfers 8; }
IFS=, read -ra RUNS <<< "${CM_RUNS:-ref,g100,g100s}"
run ${RUNS[0]}          # 첫 판이 채움 캐시를 만든다
PIDS=""; for n in "${RUNS[@]:1}"; do run $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS
kill $WP 2>/dev/null
python -I $W/cm158_an.py $W/topo/pg_wolf_sand/simulation_ply $O ${RUNS[@]} > $O/cm158_table.md 2>&1; say "an rc=$?"; cat $O/cm158_table.md | tee -a $LOG
rclone copy $O $RX -q --transfers 8
say "[끝] R2 대조: $(rclone check $O $RX --one-way -q 2>&1 | grep -c ERROR) 개 차이"
rclone copyto $LOG $RX/vast.log -q
echo "끝 $(date)" | rclone rcat $RX/vast.END
