#!/bin/bash
# [final_fix_mpmbl] 파레토 i-PG wolf 2·3 회차 (맨 뒤 반복): 1 회차 시간 판 frt3 와 같은 코드·설정 (N 25, AF_FD_EPS 1e-5 + patch_ipg_noisefloor, AF_LS_REJECT 1, AF_IPG_FAST 1,
#   AF_IPG_FUSED 0, AF_IPG_GPU 1, 50 프레임, --time --dump) 을 카드 단독으로. 판 이름 = env MB_REP (frt4·frt5). MB_POST 훅 (exe/mpmbl_boot.sh).
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
nvidia-smi --query-gpu=name,power.limit,power.default_limit,clocks.max.sm --format=csv | tee /root/mb/gpucheck.txt
python exe/patch_ipg_noisefloor.py --ipg /home/dkta/work/i-physgaussian
export AF_LS_REJECT=1
touch /root/mb/ipgw_fast
bash exe/mpmbl_ipgwolf_timewait.sh ${MB_REP:?} 25
