#!/bin/bash
# 절 CM194: 끝난 덤프 판을 다시 렌더 (PNG·이방성·시각; lb_render 고친 뒤) -- R2 pred.pt -> h5 -> 렌더 -> 시각 (기본 조건은 정답 렌더 대비) -> gq.json 다시 요약. 인자: 판 이름들
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1
RG=r2:storage/result/anchorflow/lbours/gq_justify/runs; RP=r2:storage/result/anchorflow/lbours/final_fix/phys; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; NW=$W/newsc
python exe/pr2_patch_models.py > /dev/null 2>&1
for n in "$@"; do
  s=${n%%_*}; case $s in
    L000) GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80;;
    pillow2sofa) GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300;;
    bread) GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300;;
    wolf) GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50;; esac
  while [ "$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)" -lt ${GQ_MINDISK:-30} ]; do sleep 60; done
  D=/workspace/gq/rr_$n; mkdir -p $D
  rclone copy $RG/$n $D -q --exclude "png.tar" --exclude "*.mp4" --multi-thread-streams 8
  python -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $GTD --out $D/h5 > $D/roll_h5.log 2>&1
  rm -f $D/pred.pt
  python -u exe/lb_render.py --sim $D/h5 --config $CF --shape $SHAPE --out $D/$n.mp4 --png $D/png --frames $FR --aniso_json $D/aniso.json < /dev/null > $D/render.log 2>&1
  rm -rf "$D/h5"
  IFS=_ read sc v c m r <<< "$n"; REF=""
  if [ $c = base ]; then [ -d /root/ref/$s ] || { mkdir -p /root/ref/$s.tmp && rclone copy $RP/mpm_$s/png /root/ref/$s.tmp -q --transfers 16 && mv /root/ref/$s.tmp /root/ref/$s; }; REF="--ref /root/ref/$s"; fi
  python -u exe/ffx_vis.py --png $D/png $REF --out $D/vis.json > $D/vis.log 2>&1
  rc=$(python -c "import json;print(json.load(open('$D/gq.json'))['rc'])")
  python exe/gqj_post.py $D $sc $v $c $m ${r#r} $rc > $D/post.log 2>&1
  for f in aniso.json vis.json gq.json render.log vis.log $n.mp4; do [ -f $D/$f ] && rclone copyto $D/$f $RG/$n/$f -q; done
  [ $c = base ] && (cd $D && tar -cf png.tar png && rclone copyto png.tar $RG/$n/png.tar -q)
  echo "$n $(grep 대응 $D/render.log | cut -c1-80) $(python -c "import json;o=json.load(open('$D/gq.json'));print((o.get('vis') or {}).get('lpips'), (o.get('aniso') or {}).get('rel_p99'))")"
  rm -rf "/workspace/gq/rr_${n:?}"
done
echo RERENDER_DONE
