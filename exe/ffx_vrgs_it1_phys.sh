#!/bin/bash
# 절 CM192: VR-GS 대표 설정 (반복 1, 마찰 없음 충돌, 커밋 32e95ea) 덤프 판 -> 물리·렌더·시각 (ffx_phys_drv.sh 의 베이스라인 경로와 같은 명령)
set -u
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; NW=$W/newsc; X=/workspace/fxp; RP=r2:storage/result/anchorflow/lbours/final_fix/phys
PY=/home/dkta/.conda/envs/af/bin/python
C=/root/pr/code_vr; if [ ! -d $C ]; then cp -r /root/pr/code $C; rclone cat r2:storage/result/anchorflow/lbours/final_pa/bl/code/fpabl_32e95ea.tar | tar -x -C $C exe/pareto_vrgs.py lib/anchorflow/vrgs_xpbd.py; fi
cd $C
[ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
for spec in "L000 167 264617.8308203485" "pillow2sofa 200 7736.487354333759"; do
  set -- $spec; T=$1; N=$2; E=$3; J=vrgs_${T}_n${N}it1; O=$X/$J; mkdir -p $O
  [ -f $O/DONE ] && continue
  if [ $T = L000 ]; then DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel; DA="--data $DATA --tag L000 --frames 80 --floor vel"
  else DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none; DA="--data $DATA --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp"; fi
  echo 32e95ea > $O/commit.txt
  [ -s $O/pred.pt ] || $PY -u exe/pareto_vrgs.py --mode run --out $O --E $E $DA --set ${N}x1 --dump_F > $O/run.log 2>&1
  echo "[실행] $J $(tail -1 $O/run.log | cut -c1-200)"
  [ -s $O/phys.json ] || $PY -u exe/ffx_phys.py --method vrgs --scene $T --N $N --data $DATA --pred $O/pred.pt --floor $FLOOR --frames $FR --out $O/phys.json \
      --note "VR-GS 대표 설정 (반복 1, 마찰 없음 충돌, E $E, 커밋 32e95ea) 덤프 판" > $O/phys.log 2>&1
  echo "[물리] $J $(tail -1 $O/phys.log | cut -c1-200)"
  if [ ! -f $O/png/$(printf %04d $FR).png ]; then
    [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
    $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $SHAPE --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
  fi
  while [ ! -f $X/mpm_$T/DONE ]; do sleep 30; done
  [ -s $O/vis.json ] || $PY -u exe/ffx_vis.py --png $O/png --ref $X/mpm_$T/png --out $O/vis.json > $O/vis.log 2>&1
  echo "[시각] $J $(tail -1 $O/vis.log | cut -c1-200)"
  rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt"; rclone copyto $O/pred.pt $RP/$J/pred.pt -q
  [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RP/$J/DONE -q
  rm -rf $O/h5
done
echo VR_IT1_DONE
