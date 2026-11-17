#!/bin/bash
# sg_post_p2s 렌더 실패 (lb_render MODEL 에 pillow2sofa 없음 -- pr2_patch_models.py 빠짐) 고쳐 h5·렌더·시각 지표만 다시, 올리고 DONE
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
RP=r2:storage/result/anchorflow/lbours/final_fix/phys; J=springgaus_pillow2sofa_nsg; X=/workspace/fxp/$J; NW=/home/dkta/work/newsc
L=/root/sgpost_fix.log; say() { echo "[fix $(date -u '+%F %T')] $*" >> $L; rclone copyto $L r2:storage/result/anchorflow/lbours/final_pa/bl/P2/marks_mv/sgpost_fix.log -q; }
cd /root/sgpost_code; python exe/pr2_patch_models.py >> $L 2>&1; grep -c '"pillow2sofa": "pillow2sofa_whitebg-trained"' exe/lb_render.py >> $L
python -u exe/lb_roll_to_h5.py --dump $X/pred.pt --traj $NW/gt_pillow2sofa --out $X/h5 > $X/roll_h5.log 2>&1; say "h5 rc=$?"
python -u exe/lb_render.py --sim $X/h5 --config $NW/gt_pillow2sofa/config.json --shape pillow2sofa --out $X/$J.mp4 --png $X/png --frames 300 < /dev/null > $X/render.log 2>&1
say "렌더 rc=$? $(ls $X/png 2>/dev/null | wc -l) 장 $(tail -c 200 $X/render.log | tr '\n' ' ')"
python -u exe/ffx_vis.py --png $X/png --ref /root/ref/pillow2sofa --out $X/vis.json > $X/vis.log 2>&1
say "시각 rc=$? $(head -c 300 $X/vis.json 2>/dev/null)"
rm -rf $X/h5
rclone copy $X $RP/$J -q --transfers 8 --exclude "/pred.pt" && [ -s $X/phys.json ] && [ -s $X/vis.json ] && touch $X/DONE && rclone copyto $X/DONE $RP/$J/DONE -q
say "끝 DONE $(rclone lsf $RP/$J/DONE 2>/dev/null)"
