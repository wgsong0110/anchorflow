#!/bin/bash
# [CM192] Lego·pillow2sofa 비교 영상 다시 합성 (장면 이름 영문 Lego; 칸 영상은 R2 그대로). 대기열 작업 (인스턴스 안).
RX=r2:storage/result/anchorflow/lbours/stabfinal; V=/tmp/vv; mkdir -p $V/exe
rclone cat $RX/code/sf_5cc56c53.tar | tar -x -C $V exe/stabfinal_vid.py exe/stab4_vid.py
for q in lego pillow2sofa; do
  rclone copy $RX/res/$q $V/root/$q -q --include "plan_*.json" --include "vis/*_N*.mp4"
  python -u $V/exe/stabfinal_vid.py --root $V/root --scene $q --font /home/dkta/work/font.otf --tag _f5 --out $V/root/$q/vis/${q}_stab_cmp.mp4 || exit 1
  rclone copyto $V/root/$q/vis/${q}_stab_cmp.mp4 $RX/res/$q/vis/${q}_stab_cmp.mp4 || exit 1
done
