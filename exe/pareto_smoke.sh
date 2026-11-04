#!/bin/bash
# 파레토 묶음 시험 (짧게, 빈 카드 하나, 비측정): VR-GS lego·ficus 몇 프레임, MPM 3 프레임 시간, 우리 실행기 무관.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pareto XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python; D=$(pwd); O=/dev/shm/dkta/lbours/pareto/smoke; mkdir -p $O
CL=/dev/shm/dkta/anerr_claim
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
G=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r i u; do [ $i = 0 ] && continue; echo "$apps" | grep -q $u && continue; if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi; echo $i; break; done)
echo "카드 $G"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
[ -z "$G" ] && { echo 빈 카드 없음; exit 1; }
SD=/dev/shm/dkta/anerr_data; L0=$SD/L000.pt; FI=$SD/F_pg_v0.pt; ls -la $L0 $FI
export CUDA_VISIBLE_DEVICES=$G
timeout 600 $PY -u exe/pareto_vrgs.py --mode fit --nfit 2 --out $O/fl --data $L0 --tag L000 --frames 15 --floor vel --set 16x5 167x11 2>&1 | tail -15
timeout 600 $PY -u exe/pareto_vrgs.py --mode run --E 2e6 --out $O/fr --data $FI --tag ficus --frames 10 --floor none --cfg_bc --region_density --pg_vdamp --set 50x5 2>&1 | tail -8
(cd /home/dkta/work/i-physgaussian && timeout 300 $PY -u $D/exe/pareto_mpm.py --out $O/m --frames 3 --no_dump --rpic_cfg --sub_chk --time --nsub 104 2>&1 | tail -4; python -c "import json;d=json.load(open('$O/m/err.json'));print(d['ms'],d['err_mean'])")
rclone copy $O r2:storage/result/anchorflow/lbours/pareto/smoke -q --exclude "*.pt"
