#!/bin/bash
# [final_fix_mpmbl] bread 인스턴스 마무리: i-PG bread N10 판 a (덤프, 이 카드) 끝 기다림 -> 판 b·c (다른 두 인스턴스) 가 R2 에 오길 기다림 -> 렌더·물리·시각 -> R2 대조 -> destroy
set -u
cd /root/mb/code; export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUDA_VISIBLE_DEVICES=0
APID=$1; RB=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm; X=/workspace/mb; O=$X/ipg_bread; AG=/dev/shm/dkta/ag_data
LOG=/root/mb/bf.log; say() { echo "[bf $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RB/marks/breadfin.log -q; }
( while sleep 300; do rclone copy $X $RB -q --transfers 8 --exclude "*/h5/**" --exclude "*/pred.pt"; done ) & SY=$!
while kill -0 $APID 2>/dev/null; do sleep 60; done
say "[판 a] $(tail -1 $O/N10_a.log | cut -c1-200)"
[ -s $O/N10_a/pred.pt ] && rclone copyto $O/N10_a/pred.pt $RB/ipg_bread/N10_a/pred.pt -q
echo " 5:0 10:$(python -c "import json;d=json.load(open('$O/N10_a/err.json'));print(1 if d['div'] is None and d['frames_done']==d['frames_total'] else 0)")" > $O/tried.txt
echo "10 5 0" > $O/chosen.txt
for rep in b c; do
  until [ -n "$(rclone lsf $RB/ipg_bread/N10_$rep/err.json 2>/dev/null)" ]; do sleep 300; done
  rclone copy $RB/ipg_bread/N10_$rep $O/N10_$rep -q; rclone copyto $RB/ipg_bread/N10_$rep.log $O/N10_$rep.log -q; say "[판 $rep] $(tail -1 $O/N10_$rep.log | cut -c1-200)"
done
GTD=$AG/gt_bread_full
python -u exe/lb_roll_to_h5.py --dump $O/N10_a/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
python -u exe/lb_render.py --sim $O/h5 --config $GTD/config.json --shape bread --out $O/ipg_bread.mp4 --png $O/png --frames 300 < /dev/null > $O/render.log 2>&1; rm -rf $O/h5
python -u exe/ffx_phys.py --method ipg --scene bread --N 10 --data $AG/bread.pt --pred $O/N10_a/pred.pt --floor none --frames 300 --out $O/phys.json --note "대표 설정 덤프 판 (ipg N 10; k=20 N 5 는 프레임 166 발산)" > $O/phys.log 2>&1
python -u exe/ffx_vis.py --png $O/png --ref /root/mb/ref/bread --out $O/vis.json > $O/vis.log 2>&1
[ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE
kill $SY; rclone copy $X $RB -q --transfers 8 --exclude "*/h5/**"
say "[물리·시각] $(tail -1 $O/phys.log | cut -c1-100) / $(tail -1 $O/vis.log | cut -c1-100)"
if rclone check $X $RB --one-way --size-only -q --exclude "*/h5/**" 2>/root/mb/check4.txt; then say "[R2 확인 차이 0] destroy"; vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY; else say "[R2 확인 차이] $(head -2 /root/mb/check4.txt)"; fi
