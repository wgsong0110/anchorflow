#!/bin/bash
# 절 CM20 영상 합성 (로컬; 렌더는 잡 안 exe/stab_fint.sh): R2 lbours/fint 의 렌더와 정답 렌더를 받아 exe/fem_vres_compose.py (가로 ≤ 2400, yuv420p, faststart)
# -> R2 result/anchorflow/v20261009/fem_fint_{lego_A,lego_BK2,wm}_cmp.mp4 -> ~/vidcache/ 로 받아 ffprobe 확인. 사용: bash exe/fint_compose_local.sh <작업 폴더>
set -u
WD=${1:?}; mkdir -p $WD; cd "$(dirname "$0")/.." || exit 1
R=r2:storage/result/anchorflow; F=$R/lbours/fint
g() { rclone copyto "$1" "$2" -q; }
g $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $WD/fb.otf; g $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $WD/fr.otf
g $R/lbours/fem2_lego/gt_lego.mp4 $WD/gt_lego.mp4; g $R/lbours/fem_ag_wm/gt/gt_watermelon.mp4 $WD/gt_wm.mp4
mkdir -p ~/vidcache
get() { local n=$1 v=$2; mkdir -p $WD/$n/vj; g $F/$n/summary.json $WD/$n/summary.json; g $F/$n/fem.jsonl $WD/$n/fem.jsonl; g $F/$n/vj/$v $WD/$n/vj/$v
  local vv=$WD/$n/vj/$v
  if [ -s $vv ] && tail -1 $WD/$n/fem.jsonl | grep -q '"err": NaN'; then local k=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $vv)
    ffmpeg -v error -y -i $vv -frames:v $((k - 1)) -c:v libx264 -crf 18 -pix_fmt yuv420p $vv.cut.mp4 && mv $vv.cut.mp4 $vv && echo "[마지막 비유한 프레임 뺌] $n $k -> $((k - 1))"; fi; }
put() { local N=$1
  if [ -s $WD/$N ]; then rclone copyto $WD/$N $R/v20261009/$N -q && rclone copyto $R/v20261009/$N ~/vidcache/$N -q \
      && echo "[올림·받음] $N $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv ~/vidcache/$N | tr '\n' ' ') moov 앞 $(python3 -c "d=open('$HOME/vidcache/$N','rb').read(1<<20); print(d.find(b'moov') < d.find(b'mdat'))")"
  else echo "[합성 실패] $N"; fi; }
CAP="레고 · HHT ρ 0.8, 뉴턴 2, εN 1e-2 · εP 1e-1, 결정적 · 끔 = 입자 F 를 G2P 에서 누적 · 켬 = 입자 F 를 프레임 끝 가우스 점 F 의 칸 안 삼선형 보간으로 교체 (--fp_interp)"
for c in "A fa (A) 원래 고착 (K 2, PCG 6)" "BK2 K2 (B) 장벽 K 2 (채택 판, PCG 4)"; do set -- $c; k=$1 r=$2; shift 2; t="$*"
  get ${r}_off fem_lego.mp4; get ${r}_on fem_lego.mp4
  N=fem_fint_lego_${k}_cmp.mp4
  CMP_MAXW=2400 python3 exe/fem_vres_compose.py $WD/$N "$t · 80 프레임 · $CAP" $WD/fb.otf $WD/fr.otf $WD/tmp_$k \
    "정답 (MPM)" $WD/gt_lego.mp4 - "끔" $WD/${r}_off/vj/fem_lego.mp4 $WD/${r}_off/summary.json "켬 (F 교체)" $WD/${r}_on/vj/fem_lego.mp4 $WD/${r}_on/summary.json > $WD/compose_$k.log 2>&1
  put $N; done
g $F/wm_vid_off.txt $WD/wm_vid_off.txt; g $F/wm_vid_on.txt $WD/wm_vid_on.txt
WO=$(cat $WD/wm_vid_off.txt 2>/dev/null); WN=$(cat $WD/wm_vid_on.txt 2>/dev/null)
if [ -n "$WO" ] && [ -n "$WN" ]; then get $WO fem_watermelon.mp4; get $WN fem_watermelon.mp4
  N=fem_fint_wm_cmp.mp4
  CMP_MAXW=2400 python3 exe/fem_vres_compose.py $WD/$N "수박 · SNH 판 (절 CM13 s_a 명령), 100 프레임, 비결정 · 끔 = 입자 F·logJp 를 입자 RM 으로 누적 · 켬 = 마지막 걸음도 가우스 점 RM 뒤 F·logJp 를 칸 안 삼선형 보간으로 교체" $WD/fb.otf $WD/fr.otf $WD/tmp_wm \
    "정답 (GF)" $WD/gt_wm.mp4 - "끔 ($WO)" $WD/$WO/vj/fem_watermelon.mp4 $WD/$WO/summary.json "켬 ($WN)" $WD/$WN/vj/fem_watermelon.mp4 $WD/$WN/summary.json > $WD/compose_wm.log 2>&1
  put $N; else echo "[수박 영상 칸 없음]"; fi
echo "[끝]"
