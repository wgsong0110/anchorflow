#!/bin/bash
# 절 CM192: 지표 대기열 일꾼 (exe/ffx_phys_boot.sh FX_MODE=q 가 일꾼 여럿을 띄움; 같은 카드 공유 -- 작업 대부분 CPU).
#   대기열 = R2 lbours/final_fix/queue/<라벨>.txt, 한 줄 = "<종류> <장면> <원본> <작업이름>"
#     종류 ours  : 원본 = R2 lbours/final_fix/<runs 폴더>/<판폴더> (pred.pt 가 올라와야 시작)
#     종류 sg    : Spring-Gaus 대표 설정 덤프 판 (원본 = 준비 판 이름, R2 lbours/final_pa/bl/P2/p/<원본>/sg_train.json 이 있어야 시작)
#   작업마다: 잠금 R2 phys/<작업>/LOCK (먼저 쓴 일꾼만), 물리 (ffx_phys.py) -> 렌더 (lb_roll_to_h5 + lb_render) -> 시각 (ffx_vis.py, 정답 렌더 = R2 phys/mpm_<장면>/png)
#   올리기는 뒤에서 (다음 작업을 바로 시작), DONE 은 올린 뒤에. 대기열 모든 작업이 R2 DONE 이면 끝 (부팅이 확인 뒤 스스로 destroy).
set -u
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; X=/workspace/fxp; mkdir -p $X
FX=r2:storage/result/anchorflow/lbours/final_fix; RP=$FX/phys; OX=r2:storage/result/anchorflow/lbours/final_pa/bl; Q=$FX/queue/${FX_LABEL:?}.txt
PO=/dev/shm/dkta/lbours/paper_remeasure2/stab/p
PY=/home/dkta/.conda/envs/af/bin/python; WID=${WID:-0}
scene() { case $1 in
  L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel; DA="--data $DATA --tag L000 --frames 80 --floor vel";;
  pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none; DA="--data $DATA --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp";;
  bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300; FLOOR=none; DA="";;
  wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel; DA="";; esac; }
say() { echo "[mq w$WID $(date '+%T')] $*"; }
up() {   # 뒤에서 올리고 DONE
  local O=$1 J=$2
  ( rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt" && [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RP/$J/DONE -q && rm -rf $O/h5 $O/pred.pt; echo "[up] $J rc=$?" ) >> $X/up$WID.log 2>&1 &
}
while :; do
  rclone copyto $Q /root/mq_$WID.txt -q 2>/dev/null
  left=0; got=0
  while read -r K T SRC J; do
    [ -z "${J:-}" ] && continue
    rclone lsf $RP/$J/DONE 2>/dev/null | grep -q . && continue
    left=1
    [ -d $X/$J ] && continue                                                         # 이 인스턴스의 다른 일꾼이 잡음
    rclone lsf $RP/$J/LOCK 2>/dev/null | grep -q . && continue                               # 다른 인스턴스가 잡음
    case $K in
      ours) rclone lsf $FX/$SRC/pred.pt 2>/dev/null | grep -q . || continue;;
      sg) rclone lsf $OX/P2/p/$SRC/sg_train.json 2>/dev/null | grep -q . && [ -f /root/pr/bl.done ] || continue;;
    esac
    mkdir $X/$J 2>/dev/null || continue
    echo "$(hostname) w$WID $(date)" | rclone rcat $RP/$J/LOCK -q
    O=$X/$J; scene $T; say "시작 $J ($K $SRC)"; got=1
    if [ $K = ours ]; then
      rclone copyto $FX/$SRC/pred.pt $O/pred.pt -q --multi-thread-streams 8; rclone copyto $FX/$SRC/cmd.txt $O/cmd.txt -q; rclone copyto $FX/$SRC/commit.txt $O/commit.txt -q
      $PY -u exe/ffx_phys.py --method ours --scene $T --N "$SRC" --data $DATA --pred $O/pred.pt --floor $FLOOR --frames $FR --out $O/phys.json --note "우리 판 $SRC" > $O/phys.log 2>&1
    else
      source /root/pr/bl_env.sh
      [ -d $PO/$SRC ] || rclone copy $OX/P2/p/$SRC $PO/$SRC -q --transfers 8
      N=$($PY -c "import json;print(json.load(open('$PO/$SRC/sg_train.json'))['n_step'])")
      PYTHONPATH=${ST_SG_PYPATH:-} $PY -u exe/stab_sg.py --mode run --sg ${ST_SG_REPO:-$W/Spring-Gaus} --fit $PO/$SRC --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1
      $PY -u exe/ffx_phys.py --method springgaus --scene $T --N $N --data $DATA --pred $O/pred.pt --no_F --floor $FLOOR --frames $FR --out $O/phys.json --note "Spring-Gaus 대표 설정 덤프 판 (N $N)" > $O/phys.log 2>&1
    fi
    $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
    $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $SHAPE --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
    [ -d /root/ref/$T ] || { mkdir -p /root/ref/$T.tmp$WID && rclone copy $RP/mpm_$T/png /root/ref/$T.tmp$WID -q --transfers 16 && { mv /root/ref/$T.tmp$WID /root/ref/$T 2>/dev/null || true; }; }
    $PY -u exe/ffx_vis.py --png $O/png --ref /root/ref/$T --out $O/vis.json > $O/vis.log 2>&1
    say "끝 $J 물리 $(tail -c 100 $O/phys.log | tr '\n' ' ') 시각 $(tail -c 80 $O/vis.log | tr '\n' ' ')"
    up $O $J
    break
  done < /root/mq_$WID.txt
  [ $left = 0 ] && break
  [ $got = 0 ] && sleep 60
done
wait
say "일꾼 끝"
