#!/bin/bash
# 절 AQ 영상 합성 (로컬): 정답 (절 AG 같은 렌더러 판) | 우리 판 을 나란히, 위에 글자.
# 사용: bash exe/fem_aq_compose.sh <씬 bread|wm> <R2 우리 영상 경로 (lbours/fem_ao/...)> <출력 이름> "<우리 판 글자>"
set -e
S=$1; OURS=$2; NM=$3; LAB=$4
SHAPE=$([ $S = wm ] && echo watermelon || echo $S)
T=/tmp/fem_aq_cmp/$NM; mkdir -p $T
R=r2:storage/result/anchorflow/lbours
rclone copyto $R/fem_ag_$S/gt/gt_$SHAPE.mp4 $T/gt.mp4 -q
rclone copyto $R/$OURS $T/ours.mp4 -q
FT=$HOME/.fonts/NotoSansCJKkr-Regular.otf
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $T/gt.mp4)
ffmpeg -y -loglevel error -i $T/gt.mp4 -i $T/ours.mp4 -filter_complex \
 "[0:v]scale=-2:$H,drawtext=fontfile=$FT:text='정답':x=20:y=20:fontsize=36:fontcolor=black[a];[1:v]scale=-2:$H,drawtext=fontfile=$FT:text='$LAB':x=20:y=20:fontsize=36:fontcolor=black[b];[a][b]hstack=inputs=2:shortest=1" \
 -c:v libx264 -pix_fmt yuv420p -crf 20 $T/out.mp4
mkdir -p ~/workspace/result/anchorflow/v20261009
cp $T/out.mp4 ~/workspace/result/anchorflow/v20261009/$NM.mp4
echo "[합성] ~/workspace/result/anchorflow/v20261009/$NM.mp4"
