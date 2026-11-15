#!/bin/bash
# rep4 (11-15): 장면마다 정답 + 다섯 방법 비교 영상 (같은 카메라·같은 렌더 경로로 이미 만든 영상을 2x3 으로 붙임, 글자 표시).
#   사용: bash exe/make_rep4_vid.sh <장면> <작업 폴더>   -> <작업 폴더>/rep4_<장면>_cmp.mp4 (가로 2400 이하, yuv420p)
#   PG (wolf·bread): r10 렌더 경로 (목표·PhysTwin·GS-Verse = archive_20261009/r10, VR-GS = repflow_vrgs, Spring-Gaus = repflow1115/sg, 우리 = rep4 fix)
#   GF (kiwi·lollipop): GF 공식 렌더 경로 (정답 gfs_render, 방법 rep_render_gf) -- repflow1115/gf 와 rep4.  PhysTwin 은 k=1.
set -e
S=$1; W=$2; mkdir -p $W; cd $W
RA=r2:storage/result/anchorflow; RL=$RA/lbours
FONT=$HOME/.fonts/NotoSansCJKkr-Bold.otf
get() { [ -s $2 ] || rclone copyto $1 $2 -q; [ -s $2 ] || { echo "없음: $1"; exit 1; }; }
case $S in
wolf|bread)
  get $RA/archive_20261009/r10/${S}_target.mp4 0.mp4
  get $RL/rep4/$S/fix/hex_$S.mp4 1.mp4
  get $RA/archive_20261009/r10/${S}_phystwin_k1.mp4 2.mp4
  get $RA/archive_20261009/r10/${S}_vrgs.mp4 3.mp4
  get $RL/repflow_vrgs/main/${S}_vrgsx.mp4 4.mp4
  get $RL/repflow1115/sg/main/$S/sg_$S.mp4 5.mp4
  TS=800;;
*)
  G=$RL/repflow1115/gf/$S
  get $G/render/gt.mp4 0.mp4
  get $RL/rep4/$S/fix/render/hex_$S.mp4 1.mp4
  get $G/main/render/pt1_$S.mp4 2.mp4
  get $G/main/render/gsverse_$S.mp4 3.mp4
  if [ $S = lollipop ]; then get $RL/rep4/$S/ext_vrgsx/render/vrgsx_$S.mp4 4.mp4; else get $G/main/render/vrgsx_$S.mp4 4.mp4; fi
  get $G/main/render/sg_$S.mp4 5.mp4
  TS=608;;
esac
LAB=("정답" "우리 (hex)" "PhysTwin (k=1)" "GS-Verse" "VR-GS" "Spring-Gaus")
NF=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 0.mp4)
F=""; for i in 0 1 2 3 4 5; do
  F="$F[$i:v]scale=${TS}:${TS},setsar=1,tpad=stop_mode=clone:stop=-1,trim=end_frame=$NF,drawtext=fontfile=$FONT:text='${LAB[$i]}':x=12:y=10:fontsize=30:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw=6[v$i];"
done
F="$F[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[o]"
ffmpeg -y -v error -i 0.mp4 -i 1.mp4 -i 2.mp4 -i 3.mp4 -i 4.mp4 -i 5.mp4 -filter_complex "$F" -map "[o]" -frames:v $NF \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart rep4_${S}_cmp.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 rep4_${S}_cmp.mp4
ffmpeg -v error -i rep4_${S}_cmp.mp4 -f null - && echo "[재생 확인] rep4_${S}_cmp.mp4 디코드 오류 없음 ($NF 프레임)"
