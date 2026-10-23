#!/bin/bash
# 정답 | 출력만 최적화(L-BFGS 500) | 출력만 최적화 뉴턴 40걸음 3칸 영상 (로컬, ffmpeg). 짧은 칸은 마지막 프레임 유지.
# bash exe/ovnewton_vid.sh <T> <뉴턴 영상> <출력>
T=$1; NV=$2; OUT=$3
S=/home/wgsong/workspace/result/anchorflow/lbours/spstick
FONT=/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf
H=480
lab(){ echo "drawtext=fontfile=$FONT:text='$1':fontsize=26:fontcolor=black:box=1:boxcolor=white@0.7:x=(w-tw)/2:y=10"; }
ffmpeg -y -loglevel error -i $S/gt_$T.mp4 -i $S/ov_$T.mp4 -i $NV -filter_complex "
[0:v]scale=-2:$H,setsar=1,tpad=stop_mode=clone:stop_duration=30,$(lab '정답')[a];
[1:v]scale=-2:$H,setsar=1,tpad=stop_mode=clone:stop_duration=30,$(lab '출력만 최적화 (L-BFGS 500)')[b];
[2:v]scale=-2:$H,setsar=1,tpad=stop_mode=clone:stop_duration=30,$(lab '출력만 최적화 뉴턴 40걸음')[c];
[a][b][c]hstack=inputs=3,drawtext=fontfile=$FONT:text='프레임 %{frame_num}':start_number=0:fontsize=44:fontcolor=red:x=20:y=h-th-20[o]" \
  -map "[o]" -frames:v $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $S/gt_$T.mp4) \
  -c:v libx264 -pix_fmt yuv420p -crf 20 $OUT && echo "[저장] $OUT"
