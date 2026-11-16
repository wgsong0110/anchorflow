#!/bin/bash
# [final_fix_mpmbl] AF_LS_REJECT 판 후처리: 롤아웃 -> 렌더 -> 물리 -> 시각 -> R2.  사용: bash exe/mpmbl_lsrej_post.sh <L000|pillow2sofa|bread> <판>
set -u
S=$1; REP=$2
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUDA_VISIBLE_DEVICES=0
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm
case $S in
  L000) N=42; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SH=lego; FR=80; FL=vel; DATA=$SD/L000.pt;;
  pillow2sofa) N=10; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SH=pillow2sofa; FR=300; FL=none; DATA=$NW/pillow2sofa.pt;;
  bread) N=10; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SH=bread; FR=300; FL=none; DATA=$AG/bread.pt;;
esac
O=/workspace/mb/ipg_${S}; T=N${N}_ls$REP; P=$O/post_$T; mkdir -p $P
[ -s $O/$T/pred.pt ] || rclone copy $RB/ipg_${S}/$T $O/$T -q
[ -d /root/mb/ref/$S ] && [ "$(ls /root/mb/ref/$S | wc -l)" -gt 10 ] || rclone copy $RB/../phys/mpm_${S}/png /root/mb/ref/$S --transfers 16 -q
python -u exe/lb_roll_to_h5.py --dump $O/$T/pred.pt --traj $GTD --out $P/h5 > $P/roll_h5.log 2>&1
python -u exe/lb_render.py --sim $P/h5 --config $CF --shape $SH --out $P/ipg_${S}.mp4 --png $P/png --frames $FR < /dev/null > $P/render.log 2>&1; rm -rf $P/h5
python -u exe/ffx_phys.py --method ipg --scene $S --N $N --data $DATA --pred $O/$T/pred.pt --floor $FL --frames $FR --out $P/phys.json --note "AF_LS_REJECT=1 판 $T" > $P/phys.log 2>&1
python -u exe/ffx_vis.py --png $P/png --ref /root/mb/ref/$S --out $P/vis.json > $P/vis.log 2>&1
rclone copy $P $RB/ipg_${S}/post_$T -q --transfers 8
[ -s $P/phys.json ] && [ -s $P/vis.json ] && touch $P/DONE && rclone copyto $P/DONE $RB/ipg_${S}/post_$T/DONE -q
echo "[후처리 $S $T] $(tail -1 $P/vis.log | cut -c1-120)"
