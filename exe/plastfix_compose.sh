#!/bin/bash
# 절 CM189: 렌더된 정답·우리 png 두 칸 영상 (칸 제목 오차·ms) -> 가로 ≤2400·yuv420p·faststart, 끝까지 디코드 확인, R2 v20261009/<이름> 에 올림.
#   plastfix_compose.sh D CFG "정답 라벨" ERR MS VNAME   (D 에 gt_png·ours_png; AF 코드 폴더에서 실행)
set -u
D=$1; CFG=$2; GTL=$3; ERR=$4; MS=$5; VN=$6; W=/home/dkta/work; VD=r2:storage/result/anchorflow/v20261009
FPS=$(python -c "import json;print(round(1/json.load(open('$CFG'))['frame_dt']))")
python -u exe/gfs_post.py vid --gt_label "$GTL" --our_label "우리 K=1  오차 ${ERR}%  ${MS} ms" --gt_png $D/gt_png --our_png $D/ours_png --out_json $D/visual.json --out_mp4 $D/cmp_raw.mp4 --font $W/font.otf --fps $FPS > $D/vid.log 2>&1 || { echo "[합성 실패] $(tail -1 $D/vid.log)"; exit 1; }
V=$D/$VN
ffmpeg -y -loglevel error -i $D/cmp_raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $V
DEC=$(ffmpeg -v error -i $V -f null - 2>&1 | wc -l)
echo "[영상] $V $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $V 2>&1) fps $FPS 디코드 오류 줄 $DEC"
[ "$DEC" = 0 ] && rclone copyto $V $VD/$VN -q && echo "[R2] $VD/$VN"
