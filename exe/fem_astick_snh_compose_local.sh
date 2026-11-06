#!/bin/bash
# 절 CM18 영상 합성 (로컬; 렌더는 인스턴스 exe/vast_astick_snh.sh): R2 marks/astick_snh.END 를 기다린 뒤 정답 렌더와 sa·su1·su2 렌더를 받아
# exe/fem_vres_compose.py 그대로 (가로 2400 이하, CMP_MAXW) -> R2 result/anchorflow/v20261009/fem_astick_snh_L000_cmp.mp4 -> ~/vidcache/ 로 받아 ffprobe 확인
# 사용: bash exe/fem_astick_snh_compose_local.sh <작업 폴더>
set -u
WD=${1:?}; mkdir -p $WD; cd "$(dirname "$0")/.." || exit 1
R=r2:storage/result/anchorflow; F=$R/lbours/fem_cm
until [ -n "$(rclone lsf $F/marks/astick_snh.END 2>/dev/null)" ]; do sleep 60; done
g() { rclone copyto "$1" "$2" -q; }
g $F/code/NotoSansCJKkr-Bold.otf $WD/fb.otf; g $F/code/NotoSansCJKkr-Regular.otf $WD/fr.otf
CAP="HHT ρ 0.8, K 2, 뉴턴 2, PCG 6, εN 1e-2 · εP 1e-1, 결정적 · 탄성 Stable Neo-Hookean (μ_s = μ, λ_s = λ + μ) · (A) = 가우스 점 고착 · (i)·(ii) = 고착 몫 법선 한쪽 구속, (i) 접선 고착, (ii) 접선 자유"
mkdir -p ~/vidcache
t=L000
g $R/lbours/fem2_lego/gt_lego.mp4 $WD/gt_$t.mp4
for v in sa su1 su2; do d=$WD/${v}_$t; mkdir -p $d/vj; g $F/astick_snh/${v}_$t/summary.json $d/summary.json; g $F/astick_snh/${v}_$t/fem.jsonl $d/fem.jsonl; g $F/astick_snh/${v}_$t/vj/fem_lego.mp4 $d/vj/fem_lego.mp4
  vv=$d/vj/fem_lego.mp4
  if [ -s $vv ] && tail -1 $d/fem.jsonl | grep -q '"err": NaN'; then n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $vv)   # 마지막 비유한 프레임 (빈 렌더) 빼기 -- 보이는 칸만
    ffmpeg -v error -y -i $vv -frames:v $((n - 1)) -c:v libx264 -crf 18 -pix_fmt yuv420p $vv.cut.mp4 && mv $vv.cut.mp4 $vv && echo "[마지막 비유한 프레임 뺌] ${v}_$t $n -> $((n - 1))"; fi; done
N=fem_astick_snh_${t}_cmp.mp4
CMP_MAXW=2400 python3 exe/fem_vres_compose.py $WD/$N "lego $t 80 프레임 · $CAP" $WD/fb.otf $WD/fr.otf $WD/tmp_$t \
  "정답 (MPM)" $WD/gt_$t.mp4 - "(A) + SNH" $WD/sa_$t/vj/fem_lego.mp4 $WD/sa_$t/summary.json \
  "(i) + SNH" $WD/su1_$t/vj/fem_lego.mp4 $WD/su1_$t/summary.json "(ii) + SNH" $WD/su2_$t/vj/fem_lego.mp4 $WD/su2_$t/summary.json > $WD/compose_$t.log 2>&1
if [ -s $WD/$N ]; then rclone copyto $WD/$N $R/v20261009/$N -q && rclone copyto $R/v20261009/$N ~/vidcache/$N -q \
    && echo "[올림·받음] $N $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv ~/vidcache/$N | tr '\n' ' ') moov 앞 $(python3 -c "d=open('$HOME/vidcache/$N','rb').read(1<<20); print(d.find(b'moov') < d.find(b'mdat'))")"
else echo "[합성 실패] $N $(tail -3 $WD/compose_$t.log | tr '\n' ' ')"; fi
echo "[끝]"
