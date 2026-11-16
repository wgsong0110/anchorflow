#!/bin/bash
# 절 CM192: 새 기본 판 (--wrx_el 1) 레고·pillow2sofa 우리 판 + 노이즈용 반복 덤프 -> 물리·렌더·시각 (ffx_phys_drv.sh 의 ours 경로와 같은 명령)
set -u
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; X=/workspace/fxp
RR=r2:storage/result/anchorflow/lbours/final_fix/runs; RP=r2:storage/result/anchorflow/lbours/final_fix/phys
PY=/home/dkta/.conda/envs/af/bin/python; cd /root/pr/code
scene() { case $1 in
  L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel;;
  pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none;;
  bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300; FLOOR=none;;
  wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel;; esac; }
JL=/root/nd2_jobs.txt   # 한 줄 = 장면 원본판폴더 작업이름
DL=$(( $(date +%s) + 8*3600 ))
while :; do
  left=0
  while read -r T SRC J; do
    O=$X/$J; [ -f $O/DONE ] && continue; left=1
    rclone lsf $RR/$SRC/pred.pt 2>/dev/null | grep -q . || continue
    scene $T; mkdir -p $O; echo "[nd2 $(date '+%T')] 시작 $J ($SRC)"
    [ -s $O/phys.json ] || { rclone copyto $RR/$SRC/pred.pt $O/pred.pt -q --multi-thread-streams 8; rclone copyto $RR/$SRC/cmd.txt $O/cmd.txt -q; rclone copyto $RR/$SRC/commit.txt $O/commit.txt -q
      $PY -u exe/ffx_phys.py --method ours --scene $T --N "$SRC" --data $DATA --pred $O/pred.pt --floor $FLOOR --frames $FR --out $O/phys.json --note "우리 새 기본 판 (절 CM192, --wrx_el 1) runs/$SRC" > $O/phys.log 2>&1; }
    if [ ! -f $O/png/$(printf %04d $FR).png ]; then
      [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
      $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $SHAPE --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
    fi
    while [ ! -f $X/mpm_$T/DONE ]; do sleep 30; done
    [ -s $O/vis.json ] || $PY -u exe/ffx_vis.py --png $O/png --ref $X/mpm_$T/png --out $O/vis.json > $O/vis.log 2>&1
    echo "[nd2] $J 물리 $(tail -c 120 $O/phys.log | tr '\n' ' ') 시각 $(tail -c 100 $O/vis.log | tr '\n' ' ')"
    rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt"
    [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RP/$J/DONE -q
    rm -rf $O/h5 $O/pred.pt
  done < $JL
  [ $left = 0 ] && break; [ $(date +%s) -gt $DL ] && break; sleep 60
done
echo ND2_DONE
