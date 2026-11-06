#!/bin/bash
# 절 CM17 영상 합성 (로컬; 렌더는 인스턴스 exe/vast_astick.sh, 합성은 여기서 -- 절 CM15.3 인스턴스 합성 멈춤 때문): R2 marks/astick.END 를 기다린 뒤
# 정답 렌더와 astick a·u1·u2 렌더를 받아 exe/fem_vres_compose.py 그대로 (가로 2400 이하, CMP_MAXW) -> R2 result/anchorflow/v20261009/fem_astick_up_<씬>_cmp.mp4 -> ~/vidcache/ 로 받아 ffprobe 확인
# 사용: bash exe/fem_astick_compose_local.sh <작업 폴더>
set -u
WD=${1:?}; mkdir -p $WD; cd "$(dirname "$0")/.." || exit 1
R=r2:storage/result/anchorflow; F=$R/lbours/fem_cm
until [ -n "$(rclone lsf $F/marks/astick.END 2>/dev/null)" ]; do sleep 60; done
g() { rclone copyto "$1" "$2" -q; }
g $F/code/NotoSansCJKkr-Bold.otf $WD/fb.otf; g $F/code/NotoSansCJKkr-Regular.otf $WD/fr.otf
CAP="HHT ρ 0.8, K 2, 뉴턴 2, PCG 6, εN 1e-2 · εP 1e-1, 결정적 · (A) = 가우스 점 고착 (절 CM8 a6) · (i)·(ii) = 고착 몫 법선을 한쪽 구속 (걸음 변위 법선 < 0 일 때만, 활성 집합), (i) 접선 고착, (ii) 접선 자유"
mkdir -p ~/vidcache
for t in L000 0000 0013 0018; do
  [ $t = L000 ] && g $R/lbours/fem2_lego/gt_lego.mp4 $WD/gt_$t.mp4 || g $R/lbours/fem_af_lego/gt_$t/gt_lego.mp4 $WD/gt_$t.mp4
  for v in a u1 u2; do d=$WD/${v}_$t; mkdir -p $d/vj; g $F/astick/${v}_$t/summary.json $d/summary.json; g $F/astick/${v}_$t/fem.jsonl $d/fem.jsonl; g $F/astick/${v}_$t/vj/fem_lego.mp4 $d/vj/fem_lego.mp4
    vv=$d/vj/fem_lego.mp4
    if [ -s $vv ] && tail -1 $d/fem.jsonl | grep -q '"err": NaN'; then n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $vv)   # 마지막 비유한 프레임 (빈 렌더) 빼기 -- 보이는 칸만
      ffmpeg -v error -y -i $vv -frames:v $((n - 1)) -c:v libx264 -crf 18 -pix_fmt yuv420p $vv.cut.mp4 && mv $vv.cut.mp4 $vv && echo "[마지막 비유한 프레임 뺌] ${v}_$t $n -> $((n - 1))"; fi; done
  N=fem_astick_up_${t}_cmp.mp4
  CMP_MAXW=2400 python3 exe/fem_vres_compose.py $WD/$N "lego $t 80 프레임 · $CAP" $WD/fb.otf $WD/fr.otf $WD/tmp_$t \
    "정답 (MPM)" $WD/gt_$t.mp4 - "(A) 원래" $WD/a_$t/vj/fem_lego.mp4 $WD/a_$t/summary.json \
    "(i) 법선 한쪽 + 접선 고착" $WD/u1_$t/vj/fem_lego.mp4 $WD/u1_$t/summary.json "(ii) 법선 한쪽 + 접선 자유" $WD/u2_$t/vj/fem_lego.mp4 $WD/u2_$t/summary.json > $WD/compose_$t.log 2>&1
  if [ -s $WD/$N ]; then rclone copyto $WD/$N $R/v20261009/$N -q && rclone copyto $R/v20261009/$N ~/vidcache/$N -q \
      && echo "[올림·받음] $N $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv ~/vidcache/$N | tr '\n' ' ') moov 앞 $(python3 -c "d=open('$HOME/vidcache/$N','rb').read(1<<20); print(d.find(b'moov') < d.find(b'mdat'))")"
  else echo "[합성 실패] $N $(cat $WD/compose_$t.log | tail -3 | tr '\n' ' ')"; fi
done
echo "[끝]"
