#!/bin/bash
# rep4 (11-15): 장면마다 정답 + 방법 넷 (우리·PhysTwin k1·VR-GS·Spring-Gaus; GS-Verse 는 뺌) 비교 영상 (같은 카메라·같은 렌더 경로 영상을 한 줄 5 칸으로, 글자 표시).
#   사용: bash exe/make_rep4_vid.sh <장면> <작업 폴더>   -> <작업 폴더>/rep4_<장면>_cmp.mp4 (가로 2400 이하, yuv420p)
#   PG (wolf·bread): r10 렌더 경로 (목표·PhysTwin = archive_20261009/r10, VR-GS = repflow_vrgs, Spring-Gaus = repflow1115/sg, 우리 = rep4 fix)
#   새 형상 PG 장면: rep4/<장면> (정답 gfs_render, 방법 rep_render_gf).  PhysTwin 은 k=1.
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
  get $RL/repflow_vrgs/main/${S}_vrgsx.mp4 3.mp4
  get $RL/repflow1115/sg/main/$S/sg_$S.mp4 4.mp4;;
*)   # 새 형상 PG 장면: rep4 nshape 경로 (정답 gfs_render, 방법 rep_render_gf, 같은 카메라)
  G=$RL/rep4/$S
  get $G/gt.mp4 0.mp4
  get $G/fix/render/hex_$S.mp4 1.mp4
  get $G/pt1/render/pt1_$S.mp4 2.mp4
  get $G/vrgsx/render/vrgsx_$S.mp4 3.mp4
  get $G/sg/render/sg_$S.mp4 4.mp4;;
esac
LAB=("정답" "우리 (hex)" "PhysTwin (k=1)" "VR-GS" "Spring-Gaus"); TS=480   # 한 줄 5 칸 = 가로 2400
NF=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 0.mp4)
F=""; for i in 0 1 2 3 4; do
  F="$F[$i:v]scale=${TS}:${TS},setsar=1,tpad=stop_mode=clone:stop=-1,trim=end_frame=$NF,drawtext=fontfile=$FONT:text='${LAB[$i]}':x=12:y=10:fontsize=24:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw=6[v$i];"
done
F="$F[v0][v1][v2][v3][v4]hstack=inputs=5[o]"
ffmpeg -y -v error -i 0.mp4 -i 1.mp4 -i 2.mp4 -i 3.mp4 -i 4.mp4 -filter_complex "$F" -map "[o]" -frames:v $NF \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart rep4_${S}_cmp.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 rep4_${S}_cmp.mp4
ffmpeg -v error -i rep4_${S}_cmp.mp4 -f null - && echo "[재생 확인] rep4_${S}_cmp.mp4 디코드 오류 없음 ($NF 프레임)"
