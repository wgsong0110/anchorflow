#!/bin/bash
# rep_final (11-15) 장면 하나의 비교 영상: 정답 (PG h5) 과 네 방법 (우리 hex · PhysTwin 최저 k · VR-GS · Spring-Gaus) 을 같은 lb_render 카메라로
#   (plastfix 레고 세팅 영상과 같은 카메라: --cam_back 1.6 --elev 40 --back2 1.1, 위 방향 = 세계 위 = 회전 R 셋째 행 -> 위에서 내려다보고 중력이 화면 아래).
#   go_rep4.sh 가 추적을 끝낸 뒤 같은 인스턴스에서 부른다 (O·IN·결과 폴더 규약이 go_rep4 와 같음).  사용: bash exe/rep_final_vid.sh <장면> <R2 결과 루트>
set -u
s=$1; R4=$2; PY=${PY:-python}; W=/home/dkta/work; O=/dev/shm/dkta/rep4; IN=$O/in_$s; V=/workspace/vid_$s; CFG=$IN/config.json
mkdir -p $V; L=$V/vid.log; say() { echo "[vid $(date '+%F %T')] $*" | tee -a $L; rclone copyto $L $R4/vid/vid_$s.log -q; }
[ -f $W/font.otf ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
[ -d $V/gt ] || $PY -u exe/pg_h5_to_pt.py --h5_dir $O/h5_$s --config $CFG --out_pt $V/gt.pt --out_dir $V/gt > $V/gt_cv.log 2>&1 || { say "정답 변환 실패 $(tail -1 $V/gt_cv.log)"; exit 1; }
UP=$($PY -c "
import json,math,numpy as np
c=json.load(open('$CFG')); R=np.eye(3)
for dg,ax in zip(c['rotation_degree'],c['rotation_axis']):
    cs,sn=math.cos(dg/180*3.1415926),math.sin(dg/180*3.1415926)
    R=np.array({0:[[1,0,0],[0,cs,-sn],[0,sn,cs]],1:[[cs,0,sn],[0,1,0],[-sn,0,cs]],2:[[cs,-sn,0],[sn,cs,0],[0,0,1]]}[ax])@R
print(','.join('%.6f'%q for q in R[2]))")
FPS=$($PY -c "import json;print(round(1/json.load(open('$CFG'))['frame_dt']))")
rm_of() { grep -a "\[요약\]" $O/$s/$1/$2_$s.log 2>/dev/null | tail -1 | sed -E 's/.*RMSE ([0-9.]+)%.*/\1/'; }
PT=pt1; PB=999; for k in pt1 pt10 pt100 pt1000; do r=$(rm_of $k $k); [ -n "$r" ] && python3 -c "import sys; sys.exit(0 if $r < $PB else 1)" && { PT=$k; PB=$r; }; done
say "장면 $s 위 방향 $UP fps $FPS PhysTwin 영상 판 $PT ($PB %)"
NG=$(nvidia-smi -L | wc -l); i=0; P=""
rend() { local src=$1 nm=$2 g=$3
  [ -s $V/$nm.mp4 ] || CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $src --config $CFG --shape $s --out $V/$nm.mp4 --cam_back 1.6 --elev 40 --up $UP --back2 1.1 --png $V/${nm}_png --fps $FPS < /dev/null > $V/r_$nm.log 2>&1
  say "렌더 $nm rc=$? $(ls $V/${nm}_png 2>/dev/null | wc -l) 장 $(grep -a '대응\|rror' $V/r_$nm.log | tail -2 | tr '\n' ' ' | cut -c1-200)"; }
rend $V/gt gt $((i % NG)) & P="$P $!"; i=$((i + 1))
for vn in hex:hex $PT:$PT vrgsx:vrgsx sg:sg; do v=${vn%%:*}; n=${vn#*:}_$s
  [ -d $V/h5_$v ] || $PY -u exe/rep_final_vid.py to_h5 --res $O/$s/$v/$n.npz --aux $IN/aux_$s.npz --gt_dir $V/gt --out $V/h5_$v > $V/cv_$v.log 2>&1 || { say "변환 실패 $v $(tail -2 $V/cv_$v.log | tr '\n' ' ')"; continue; }
  rend $V/h5_$v $v $((i % NG)) & P="$P $!"; i=$((i + 1))
done
wait $P
declare -A NAME=([bread_foam_drop_lego]="bread (foam, 낙하)" [drums_honey_lego]="drums (foam)" [chair_paste_lego]="chair (plasticine)")
$PY -u exe/rep_final_vid.py grid --png $V/gt_png $V/hex_png $V/${PT}_png $V/vrgsx_png $V/sg_png \
  --label "정답 (PhysGaussian MPM)" "우리 (육면체)  $(rm_of hex hex)%" "PhysTwin (k=${PT#pt})  $PB%" "VR-GS  $(rm_of vrgsx vrgsx)%" "Spring-Gaus  $(rm_of sg sg)%" \
  --title "${NAME[$s]:-$s}|L2 오차 (RMSE/L)|같은 카메라" --font $W/font.otf --fps $FPS --out $V/raw.mp4 > $V/grid.log 2>&1
OUT=$V/rep_final_${s}_cmp.mp4
ffmpeg -y -loglevel error -i $V/raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $OUT
DEC=$(ffmpeg -v error -i $OUT -f null - 2>&1 | wc -l)
say "영상 $(tail -1 $V/grid.log | cut -c1-120) $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $OUT 2>&1) 디코드 오류 줄 $DEC"
rclone copyto $OUT $R4/vid/$(basename $OUT) -q
rclone copy $V $R4/vid/$s -q --include "*.mp4" --include "*.log" --exclude "raw.mp4"
say "끝 $R4/vid/$(basename $OUT)"
