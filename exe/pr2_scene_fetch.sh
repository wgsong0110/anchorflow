#!/bin/bash
# [paper_remeasure2] 새 PhysGaussian 장면 자료를 인스턴스로 (vast_pr2_boot.sh 가 PR_SC=vasedeck,plane,.. 일 때 source): 3DGS 모델 · 데이터 .pt · 정답 h5 · 렌더용 정답 폴더 · config
W=/home/dkta/work; NW=$W/newsc; mkdir -p $NW
for k in ${PR_SC//,/ }; do
  case $k in vasedeck) mdl=vasedeck_whitebg-trained;; plane) mdl=plane-trained;; pillow2sofa) mdl=pillow2sofa_whitebg-trained;; esac
  [ -d $W/pgmodel/$mdl ] || { rclone copyto $RX/assets/$mdl.zip /root/pr/$mdl.zip -q && unzip -q -o /root/pr/$mdl.zip -d $W/pgmodel; }
  S=$RX/scenes/$k
  rclone copyto $S/$k.pt $NW/$k.pt -q --transfers 8 &
  rclone copy $S/h5 $NW/${k}_h5 -q --transfers 16 &
  rclone copy $S/gt_$k $NW/gt_$k -q --transfers 8 &
  rclone copyto $RX/assets/${k}_config.json $NW/${k}_config.json -q &
  wait
  mkdir -p $W/gtnew; [ -e $W/gtnew/$k ] || ln -s $NW/${k}_h5 $W/gtnew/$k    # 정답 궤적 meta.json 이 정답 생성 때 경로 (gtnew/<씬>/simulation_ply) 를 적어 둠
  say "[장면] $k pt $(stat -c %s $NW/$k.pt 2>&1) h5 $(ls $NW/${k}_h5/simulation_ply 2>/dev/null | wc -l) gt $(ls $NW/gt_$k 2>/dev/null | wc -l)"
done
python exe/pr2_patch_models.py
