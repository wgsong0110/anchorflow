#!/bin/bash
# 절 CM23 ficus 영상 합성 (로컬; 렌더는 잡 안 exe/stab_fk1vid.sh): R2 lbours/k1new/fk1vid 의 K 1·K 2 렌더 + 정답 렌더 (lbours/fem3_ficus/gt_ficus.mp4, 절 CM4 와 같은 카메라)
# -> exe/fem_vres_compose.py (CMP_PERIOD=1: 칸마다 오차·주기 비·진폭 비, 가로 ≤ 2400, yuv420p, faststart) -> R2 result/anchorflow/v20261009/fem_ficus_k1_cmp.mp4 -> ~/vidcache/ ffprobe. 사용: bash exe/fk1vid_compose_local.sh <작업 폴더>
set -u
WD=${1:?}; mkdir -p $WD; cd "$(dirname "$0")/.." || exit 1
R=r2:storage/result/anchorflow; F=$R/lbours/k1new/fk1vid
g() { rclone copyto "$1" "$2" -q; }
g $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $WD/fb.otf; g $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $WD/fr.otf
g $R/lbours/fem3_ficus/gt_ficus.mp4 $WD/gt_ficus.mp4
for n in F_K1 F_K2; do mkdir -p $WD/$n/vj; g $F/$n/summary.json $WD/$n/summary.json; g $F/$n/period.txt $WD/$n/period.txt; g $F/$n/vj/fem_ficus.mp4 $WD/$n/vj/fem_ficus.mp4; done
N=fem_ficus_k1_cmp.mp4
CAP="ficus F_pg_v0 125 프레임 · 기본 판 (절 CM23: 보간 교체 --fp_interp, 바닥 없음, HHT ρ 0.8, εN 1e-2 · εP 1e-1, 상한 뉴턴 2 x PCG 24) · 결정적 · 주기 비·진폭 비 = 잎 끝 (z 상위 5 %) 정답 대비"
CMP_PERIOD=1 CMP_MAXW=2400 python3 exe/fem_vres_compose.py $WD/$N "$CAP" $WD/fb.otf $WD/fr.otf $WD/tmp \
  "정답 (PG)" $WD/gt_ficus.mp4 - "기본 판 (프레임당 1 걸음)" $WD/F_K1/vj/fem_ficus.mp4 $WD/F_K1/summary.json "비교 (프레임당 2 걸음)" $WD/F_K2/vj/fem_ficus.mp4 $WD/F_K2/summary.json > $WD/compose.log 2>&1
cat $WD/compose.log
mkdir -p ~/vidcache
if [ -s $WD/$N ]; then rclone copyto $WD/$N $R/v20261009/$N -q && rclone copyto $R/v20261009/$N ~/vidcache/$N -q \
    && echo "[올림·받음] $N $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv ~/vidcache/$N | tr '\n' ' ') moov 앞 $(python3 -c "d=open('$HOME/vidcache/$N','rb').read(1<<20); print(d.find(b'moov') < d.find(b'mdat'))")"
else echo "[합성 실패] $N"; fi
