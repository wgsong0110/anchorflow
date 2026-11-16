#!/bin/bash
# [final_fix_mpmbl] 이어 돌리기 확인: bread i-PG N10 4 프레임 한 번에 vs 2 프레임 체크포인트 -> 죽임 -> 이어서 4 프레임. 오차 열 비교.
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb CUDA_VISIBLE_DEVICES=0 AF_IPG_FUSED=0 AF_IPG_GPU=1
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/workspace/mb/rtest; rm -rf $O; mkdir -p $O
C="exe/pareto_mpmbl.py --method ipg --nsub 10 --gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $AG/bread.pt --frames 4 --pg_cfg --dump"
python -u $C --out $O/straight > $O/straight.log 2>&1
python -u $C --out $O/rs --ckpt_every 2 > $O/rs1.log 2>&1 &
P=$!; until grep -q "체크포인트\] 프레임 2" $O/rs1.log; do sleep 2; kill -0 $P 2>/dev/null || break; done; kill $P; wait $P 2>/dev/null
python -u $C --out $O/rs --ckpt_every 2 --resume > $O/rs2.log 2>&1
python - <<PY
import json,torch
a=json.load(open("$O/straight/err.json"));b=json.load(open("$O/rs/err.json"))
print("한 번에", a["err"]); print("이어서 ", b["err"])
pa=torch.load("$O/straight/pred.pt",weights_only=False)["pred"]; pb=torch.load("$O/rs/pred.pt",weights_only=False)["pred"]
print("pred 최대 차", float((pa-pb).abs().max()), "프레임별", [float((pa[i]-pb[i]).abs().max()) for i in range(pa.shape[0])])
PY
