#!/bin/bash
# [final_fix_mpmbl] 새 정상 호스트 A6000 에서 i-PG wolf N25 수정판 단독 시간 판 (MB_POST 훅): FD 1e-5 + AF_LS_REJECT + AF_IPG_FAST, 판 frt2.
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
nvidia-smi --query-gpu=name,power.limit,power.default_limit,clocks.max.sm --format=csv | tee /root/mb/gpucheck.txt
python exe/patch_ipg_noisefloor.py --ipg /home/dkta/work/i-physgaussian
export AF_LS_REJECT=1
bash exe/mpmbl_ipgwolf_timewait.sh frt2 25
