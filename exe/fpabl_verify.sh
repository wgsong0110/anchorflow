#!/bin/bash
# [final_pa_bl] 시간 구간 cuboid 풀림 짧은 확인 (pillow2sofa 60 프레임 = 1.2 s: 0.5 s·1.0 s 두 상자 모두 풀림) -- VR-GS 실행, PhysTwin 연기 맞춤 (CMA 1 세대·학습 1 반복) 뒤 실행.
# 인스턴스 안 코드 폴더에서: source /root/pr/bl_env.sh; bash exe/fpabl_verify.sh
W=/home/dkta/work; O=/dev/shm/dkta/fbv; D=$W/newsc/pillow2sofa.pt; mkdir -p $O
DA="--data $D --tag pillow2sofa --frames 60 --floor none --cfg_bc --pg_vdamp"
export CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_v
python -u exe/pareto_vrgs.py --mode run --out $O/vrgs --E 10000 --set 200x11 $DA --dump_F > $O/vrgs.log 2>&1; tail -3 $O/vrgs.log
PYTHONPATH=$ST_PT_PYPATH python -u exe/stab_phystwin.py --mode fit --phystwin $ST_PT_REPO --out $O/pt $DA --cma_iter 1 --train_iters 1 > $O/ptfit.log 2>&1; tail -3 $O/ptfit.log
PYTHONPATH=$ST_PT_PYPATH python -u exe/stab_phystwin.py --mode run --phystwin $ST_PT_REPO --fit $O/pt --out $O/ptr $DA --dump > $O/ptrun.log 2>&1; tail -3 $O/ptrun.log
grep -h "구간 고정\|고정 정점" $O/*.log
python exe/fpabl_check_release.py --data $D --pred $O/vrgs/pred.pt $O/ptr/pred.pt
