#!/bin/bash
# [blx] 연기 시험 (짧은 프레임): VR-GS 실행 (덤프·물리 지표), PhysTwin 맞춤 (CMA 1 세대·학습 1 반복) -> 실행, Spring-Gaus 다시점·연기 학습 -> 실행. wolf·bread.
#  인스턴스 코드 폴더에서: source /root/pr/bl_env.sh; bash exe/blx_smoke.sh [장면들] [프레임 wolf] [프레임 bread]
SC=${1:-"wolf bread"}; FW=${2:-12}; FB=${3:-30}
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; O=/dev/shm/dkta/blxs; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1
for t in $SC; do
  if [ $t = wolf ]; then FR=$FW; DA="--data $AG/wolf.pt --tag wolf --frames $FR --floor vel --cfg_bc"; VS=400x11; E=5e7; CF=$W/PhysGaussian/config/wolf_config.json
  else FR=$FB; DA="--data $AG/bread.pt --tag bread --frames $FR --floor none --cfg_bc --pg_vdamp"; VS=100x11; E=2e3; CF=$W/topo/bread_plastic_config.json; fi
  WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_s python -u exe/pareto_vrgs.py --mode run --out $O/vrgs_$t --E $E --set $VS $DA --dump_F > $O/vrgs_$t.log 2>&1; tail -2 $O/vrgs_$t.log
  python -u exe/blx_phys_metrics.py --pred $O/vrgs_$t/pred.pt --data $AG/$t.pt --out $O/vrgs_$t/phys.json > $O/vrgs_${t}_phys.log 2>&1; tail -1 $O/vrgs_${t}_phys.log | cut -c1-400
  WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_s PYTHONPATH=$ST_PT_PYPATH python -u exe/stab_phystwin.py --mode fit --phystwin $ST_PT_REPO --out $O/pt_$t $DA --cma_iter 1 --train_iters 1 > $O/ptfit_$t.log 2>&1; tail -2 $O/ptfit_$t.log
  WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_s PYTHONPATH=$ST_PT_PYPATH python -u exe/stab_phystwin.py --mode run --phystwin $ST_PT_REPO --fit $O/pt_$t --out $O/ptr_$t $DA --dump --dump_F > $O/ptrun_$t.log 2>&1; tail -2 $O/ptrun_$t.log
  python -u exe/stab_mv_render.py --sim $AG/gt_$t --config $CF --shape $t --frames $FR --out $O/sg_$t/mv > $O/sgmv_$t.log 2>&1; tail -1 $O/sgmv_$t.log
  PYTHONPATH=$ST_SG_PYPATH python -u exe/stab_sg.py --mode prep --sg $ST_SG_REPO --fit $O/sg_$t --mv $O/sg_$t/mv --out $O/sg_$t $DA --smoke 1 > $O/sgprep_$t.log 2>&1; tail -1 $O/sgprep_$t.log
  PYTHONPATH=$ST_SG_PYPATH python -u exe/stab_sg.py --mode train --sg $ST_SG_REPO --fit $O/sg_$t --out $O/sg_$t $DA > $O/sgtrain_$t.log 2>&1; tail -1 $O/sgtrain_$t.log
  PYTHONPATH=$ST_SG_PYPATH python -u exe/stab_sg.py --mode run --sg $ST_SG_REPO --fit $O/sg_$t --out $O/sgr_$t $DA --dump > $O/sgrun_$t.log 2>&1; tail -2 $O/sgrun_$t.log
done
grep -h "구간\|재질\|구동\|소성" $O/*.log | sort | uniq | head -30
rclone copy $O r2:storage/result/anchorflow/lbours/final_pa/blx/smoke --exclude "*.pt" --exclude "*/mv/**" --exclude "*/data/**" --exclude "*/checkpoints/**" --exclude "*/exp/**" -q
