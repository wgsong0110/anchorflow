#!/usr/bin/env bash
# FEM 실시간 판 float64 빠른 GPU (A100/H100) 측정 (doc/fem_realtime.md 절 I.6). 범용 pytorch 이미지 + cupy 휠만 (빌드 없음 -- CuPy NVRTC 는 런타임 컴파일).
# 인스턴스 시작 때 실행: env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT TAG.  결과는 처음부터 R2 lbours/fem_rt_lego/vast_<TAG>/ 로 동기화.
set -uo pipefail
export PYTHONUTF8=1 OMP_NUM_THREADS=4 LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
O=/workspace/out; mkdir -p $O /workspace/data
exec > >(tee -a $O/setup.log) 2>&1
R=r2:storage/result/anchorflow/lbours/fem_rt_lego/vast_${TAG}
echo "[시작] $(date -u +%FT%TZ)"
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
( while true; do sleep 60; rclone copy $O $R -q; done ) & SYNC=$!
nvidia-smi --query-gpu=name,power.limit,power.draw,clocks.current.sm,clocks.max.sm,utilization.gpu,memory.used --format=csv | tee $O/gpu_boot.txt
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; } >/dev/null 2>&1
pip install -q cupy-cuda12x 2>&1 | tail -1
python -c "import torch, cupy; print('torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)"
git clone -q https://github.com/wgsong0110/anchorflow.git /workspace/af && cd /workspace/af && git checkout -q $AF_COMMIT && git log --oneline -1
rclone copy r2:storage/result/anchorflow/vastpkg/L000.pt /workspace/data/ --transfers 8
D=/workspace/data/L000.pt
ls -la $D
# fp32/fp64 행렬곱 처리량 (전력 제한 확인용)
python - <<'PY' | tee $O/matmul.txt
import torch, time
for dt in (torch.float32, torch.float64):
    a = torch.randn(8192, 8192, device="cuda", dtype=dt); b = torch.randn_like(a)
    torch.backends.cuda.matmul.allow_tf32 = False
    a @ b; torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5): a @ b
    torch.cuda.synchronize(); print(dt, f"{5 * 2 * 8192**3 / (time.perf_counter() - t) / 1e12:.1f} TFLOPS")
PY
nvidia-smi --query-gpu=name,power.limit,power.draw,clocks.current.sm,clocks.max.sm --format=csv | tee -a $O/gpu_boot.txt
python -u exe/fem_rt_mix_bench.py --data $D --t 0 40 --micro --variants pcg_mix:2:0 < /dev/null > $O/bench.log 2>&1
P="--data $D --floor particle --fast --cuda --graph --lite --warp --near32 --sym --true_res --no_dump --dump_pred"
for run in "final_rt --pcg2 2" "mix --pcg2 4 --mix_delta 0.1" "t64_asm --pcg2 7 --asm_k" "tmix_asm --pcg2 6 --mix_delta 0.1 --asm_k" "final_rt_r2 --pcg2 2"; do
  set -- $run; tg=$1; shift
  mkdir -p $O/$tg
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/$tg/gpu_at_start.txt
  echo "[명령] 커밋 $AF_COMMIT python -u exe/fem_rt_sim.py --out $O/$tg $P $*" > $O/$tg/cmd.txt
  python -u exe/fem_rt_sim.py --out $O/$tg $P "$@" < /dev/null > $O/$tg/v_fem.log 2>&1
  grep -a "롤아웃\]\|구간 중앙" $O/$tg/v_fem.log | tail -2 | tee $O/$tg/err_fem.txt
  rclone copy $O $R -q
done
echo "[끝] $(date -u +%FT%TZ)"
kill $SYNC; rclone copy $O $R -q; touch $O/VAST_DONE; rclone copy $O $R -q
