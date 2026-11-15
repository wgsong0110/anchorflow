#!/bin/bash
# 절 CM189 (plastfix): 판 하나 실행 (카드 0, 혼자). 사용: bash exe/plastfix_run.sh <장면> <판 이름> "<감싸개 접두 (예: exe/ptan_wrap.py --ptan 1)>" <t: 단독 시간 판 (진단·덤프 뺌) | d: 덤프+진단> "<바꿀 인자 (예: --newton 12 --rho 0.5)>"
# 명령 = R2 lbours/nshape/ours/<장면>_K1/cmd.txt (기존 우리 판 K=1) 에서 --out 만 바꾸고 접두 감싸개를 python -u 바로 뒤에 넣고, 바꿀 인자는 같은 이름이 있으면 값만 바꿈 (없으면 덧붙임).
# 결과: /root/pf/<장면>/<판>/ (summary.json·fem.jsonl·nl.jsonl·run.log·cmd.txt·plast.json), R2 lbours/plastfix/runs/<장면>/<판>/ (pred.pt 제외)
set -u
SC=$1; NM=$2; WR=$3; MODE=$4; EX=${5:-}
R=r2:storage/result/anchorflow; MK=$R/lbours/plastfix
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pf XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0 WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_0 WANDB_MODE=disabled
export PYTHONPATH=/root/gfs/GaussianFluent:/root/gfs/GaussianFluent/gaussian-splatting
mkdir -p $CUPY_CACHE_DIR $WARP_CACHE_PATH
D=/root/pf/$SC/$NM; mkdir -p $D; cd /root/af
python - /root/pr/$SC.cmd "$D" "$WR" "$MODE" "$EX" > $D/cmd.txt <<'P'
import re, sys
src, out, wr, mode, ex = sys.argv[1:]
c = " ".join(open(src).read().split())
c = re.sub(r"--out \S+", f"--out {out}", c)
if wr.strip():
    c = c.replace("python -u ", "python -u " + wr.strip() + " ", 1)
if mode == "t":
    for f in ("--pdiag", "--qjd", "--fhash", "--hgt", "--bar_cnt", "--dump", "--dump_F", "--dump_part"):
        c = re.sub(r"(^| )" + re.escape(f) + r"(?= |$)", " ", c)
toks = ex.split(); i = 0
while i < len(toks):
    k = toks[i]; v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else None
    pat = r"(?<= )" + re.escape(k) + (r" \S+" if v is not None else "") + r"(?= |$)"
    if re.search(pat, c):
        c = re.sub(pat, k + (" " + v if v is not None else ""), c)
    else:
        c = c.replace(" --out ", f" {k}{' ' + v if v is not None else ''} --out ", 1)
    i += 2 if v is not None else 1
print(" ".join(c.split()))
P
t0=$(date +%s)
bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/run.log 2>&1; rc=$?
echo "[$SC/$NM] rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -a '롤아웃\]' $D/run.log | tail -1 | cut -c1-220) $(grep -a 'rror' $D/run.log | tail -1 | cut -c1-200)" | tee -a /root/pf/runs.log
if [ "$MODE" = d ] && [ -f $D/pred.pt ]; then
  python -u exe/nshape_plast_an.py --h5 /root/gfs/gt/$SC/simulation_ply --pred $D/pred.pt --config /root/gfs/hf/config/$SC.json --out $D/plast.json > $D/plast.log 2>&1
fi
rclone copy $D $MK/runs/$SC/$NM -q --exclude "pred.pt" --exclude "h5*/**" --exclude "*png/**"
rclone copyto /root/pf/runs.log $MK/runs_${CONTAINER_ID:-x}.log -q
