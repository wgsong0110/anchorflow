#!/bin/bash
# [final_fix_phys] bread 정답 렌더 폴더 고침: 부팅이 vast_fpa.sh 처럼 --out_dir_n 1 (0 프레임만) 로 만들어 정답 MPM 물리·렌더가 0 프레임뿐이었음.
#   전 프레임 폴더 ag_data/gt_bread_full 을 만들고 (pg_h5_to_pt 같은 인자, --out_dir_n 뺌) -> mpm_bread 물리·렌더·시각 다시 -> bread 의 다른 작업 시각 지표 (정답 렌더 대비) 다시.
#   실행기 (exe/ffx_phys_drv.sh) 가 끝난 뒤 (drv.log 의 DRV_DONE) 에 돌림. 코드 폴더에서: bash exe/ffx_phys_fixbread.sh
set -u
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; X=/workspace/fxp; RP=r2:storage/result/anchorflow/lbours/final_fix/phys; PY=/home/dkta/.conda/envs/af/bin/python
export CUDA_VISIBLE_DEVICES=${G:-0} PYTHONUTF8=1
G2=$AG/gt_bread_full
[ -s $G2/config.json ] || $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_full_unused.pt --out_dir $G2 > /root/pr/gt_bread_full.log 2>&1
echo "[정답 폴더] $(ls $G2/sim_*.h5 | wc -l) 프레임"
until grep -q DRV_DONE /root/pr/drv.log 2>/dev/null; do sleep 60; done
O=$X/mpm_bread; mkdir -p $O/old; mv $O/phys.json $O/vis.json $O/png $O/*.mp4 $O/DONE $O/old/ 2>/dev/null
N=$($PY -c "import json;c=json.load(open('$G2/config.json'));print(int(c['frame_dt']/c['substep_dt']))")
$PY -u exe/ffx_phys.py --method mpm --scene bread --N $N --data $AG/bread.pt --gt_h5 $G2 --floor none --frames 300 --out $O/phys.json \
  --note "정답 MPM (PhysGaussian 공식 config 서브스텝 int(frame_dt/substep_dt))" > $O/phys.log 2>&1; tail -n 1 $O/phys.log | cut -c1-300
$PY -u exe/lb_render.py --sim $G2 --config $G2/config.json --shape bread --out $O/mpm_bread.mp4 --png $O/png --frames 300 < /dev/null > $O/render.log 2>&1
$PY -u exe/ffx_vis.py --png $O/png --out $O/vis.json > $O/vis.log 2>&1 && touch $O/DONE; tail -n 1 $O/vis.log | cut -c1-300
rclone copy $O $RP/mpm_bread -q --transfers 8 --exclude "old/**"
for O in $X/*_bread_*/; do
  O=${O%/}; J=$(basename $O); [ -d $O/png ] || continue
  mv $O/vis.json $O/vis_ref1frame.json 2>/dev/null
  $PY -u exe/ffx_vis.py --png $O/png --ref $X/mpm_bread/png --out $O/vis.json > $O/vis.log 2>&1
  echo "[시각] $J $(tail -n 1 $O/vis.log | cut -c1-200)"
  rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt"
done
echo FIXBREAD_DONE | tee /root/pr/fixbread.done
rclone copyto /root/pr/fixbread.done r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_phys.fixbread.done -q
