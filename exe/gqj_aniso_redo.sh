#!/bin/bash
# 절 CM194: 이미 끝난 덤프 판의 이방성 다시 (lb_render --aniso_only, f64 특이값) -- R2 pred.pt -> h5 -> aniso.json -> gq.json 다시 요약. 인자: 판 이름들
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1
RG=r2:storage/result/anchorflow/lbours/gq_justify/runs; W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc
for n in "$@"; do
  s=${n%%_*}; case $s in
    L000) GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80;;
    pillow2sofa) GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300;;
    bread) GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300;;
    wolf) GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50;; esac
  D=/workspace/gq/an_$n; mkdir -p $D
  rclone copy $RG/$n $D -q --exclude "png.tar" --exclude "*.mp4" --multi-thread-streams 8
  python -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $GTD --out $D/h5 > $D/roll_h5b.log 2>&1
  python -u exe/lb_render.py --sim $D/h5 --config $CF --shape $SHAPE --out $D/x.mp4 --frames $FR --aniso_json $D/aniso.json --aniso_only 1 < /dev/null > $D/aniso.log 2>&1
  IFS=_ read sc v c m r <<< "$n"; rc=$(python -c "import json;print(json.load(open('$D/gq.json'))['rc'])")
  python exe/gqj_post.py $D $sc $v $c $m ${r#r} $rc > $D/post.log 2>&1
  rclone copyto $D/aniso.json $RG/$n/aniso.json -q; rclone copyto $D/gq.json $RG/$n/gq.json -q
  echo "$n $(cat $D/post.log | tail -1 | cut -c1-80) $(python -c "import json;o=json.load(open('$D/aniso.json'));print(o['rel_p99'],o['rel_max'],o['p99'])")"
  rm -rf "/workspace/gq/an_${n:?}"
done
echo ANISO_DONE
