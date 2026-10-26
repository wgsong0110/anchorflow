#!/bin/bash
# 로컬에서 4 칸 영상 붙이기 (정답 | ε 고정 | 투영 ADMM | ALM): 칸 렌더는 클러스터 (topo_pa_run.sh), drawtext 는 로컬 ffmpeg.
# 프레임 번호 크게 빨간색, 멈춘 프레임(topo_pa_cmp 의 json, 정답 이동의 10 % 미만 첫 프레임) 부터 그 칸에 '멈춤'.
#   bash exe/topo_pa_vid.sh wolf
set -eu
S=$1
R=/home/wgsong/workspace/result/anchorflow/topo/riem
E=$R/eps_cmp/vid; P=$R/proj_alm/vid; J=$R/proj_alm/${S}_tetmesh_proj_alm.json
FONT=/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf
st() { python3 -c "import json; print(json.load(open('$J'))['$1']['stall'])"; }
SF=$(st 'ε 고정'); SP=$(st '투영 ADMM'); SA=$(st 'ALM')
echo "[멈춤] ε 고정 $SF  투영 $SP  ALM $SA"
lab() { echo "drawtext=fontfile=$FONT:text='$1':x=20:y=20:fontsize=40:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw=8"; }
stop() { [ "$1" -gt 0 ] && echo ",drawtext=fontfile=$FONT:text='멈춤':x=(w-tw)/2:y=h-th-40:fontsize=72:fontcolor=red:box=1:boxcolor=white@0.7:boxborderw=10:enable='gte(n\,$1)'" || true; }
TMP=$(mktemp -d)
ffmpeg -loglevel error -y -i $E/${S}_gt.mp4 -i $E/${S}_fix.mp4 -i $P/${S}_proj.mp4 -i $P/${S}_alm.mp4 -filter_complex \
 "[0]$(lab '정답')[a];[1]$(lab 'ε 고정')$(stop $SF)[b];[2]$(lab '투영 ADMM')$(stop $SP)[c];[3]$(lab 'ALM')$(stop $SA)[d];[a][b][c][d]hstack=inputs=4,drawtext=fontfile=$FONT:text='%{n}':x=w-tw-30:y=20:fontsize=110:fontcolor=red[o]" \
 -map "[o]" -c:v libx264 -pix_fmt yuv420p $TMP/out.mp4
cp $TMP/out.mp4 $R/${S}_tetmesh_proj_alm.mp4 && rm -r $TMP
echo "[영상] $R/${S}_tetmesh_proj_alm.mp4"
