#!/bin/bash
# ficus B-스플라인 FFD 비교 영상 2×3 (로컬 ffmpeg 합성): 정답 | 선형 | 수정 Gregory / 회전판 | A | B, 프레임 번호 크게 빨간색.
#   3DGS 알파블렌딩 렌더 -> v20261009/ficus_bspline.mp4
#   탄성 에너지 색 (psi_color 공통 고정 범위 log10(ψ+1e-3) -3.000 ~ 4.933, 칸별 정규화 없음) -> v20261009/ficus_bspline_psi.mp4
set -eu
B=/home/wgsong/workspace/result/anchorflow/lbours; V=/home/wgsong/workspace/result/anchorflow/v20261009
FB=/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf
mk(){ # 출력 제목 입력 6 개
  local out=$1 ttl=$2; shift 2
  local labs=("정답 (PG)" "선형 출력만 최적화" "수정 Gregory" "셀 안 버블 회전" "A 2차 B-스플라인+det 장벽" "B 3차 B-스플라인+차 상한")
  local f="" i=0
  for i in 0 1 2 3 4 5; do
    f="$f[$i:v]scale=640:640,setsar=1,drawtext=fontfile=$FB:text='${labs[$i]}':fontsize=32:fontcolor=black:x=(w-tw)/2:y=14[p$i];"
  done
  f="${f}[p0][p1][p2][p3][p4][p5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0,pad=iw:ih+80:0:80:white,drawtext=fontfile=$FB:text='$ttl':fontsize=28:fontcolor=black:x=20:y=24,"
  f="${f}drawtext=fontfile=$FB:text='프레임 %{eif\\:n\\:d}':fontsize=72:fontcolor=red:x=w-tw-24:y=2[o]"
  ffmpeg -y -loglevel error -i "$1" -i "$2" -i "$3" -i "$4" -i "$5" -i "$6" -filter_complex "$f" -map "[o]" -c:v libx264 -pix_fmt yuv420p -crf 20 -r 30 "$out"
  echo "[영상] $out $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,width,height -of csv=p=0 "$out")"
}
mk /tmp/ficus_bspline.mp4 "ficus 균일 초기 속도 · 출력만 최적화 · 3DGS 렌더" \
   $B/ficus_v0/gt_ficus.mp4 $B/ficus_v0/ov_ficus.mp4 $B/ficus_v0_greg_fix/run/ov_ficus.mp4 $B/ficus_v0_localrot/run/ov_ficus.mp4 \
   $B/ficus_v0_bspline/A/ov_ficus.mp4 $B/ficus_v0_bspline/B/ov_ficus.mp4
mk /tmp/ficus_bspline_psi.mp4 "탄성 에너지 밀도 색 log10(ψ+0.001), 공통 고정 범위 -3.00 ~ 4.93 (turbo)" \
   $B/ficus_v0/psi/psi_gt_ficus.mp4 $B/ficus_v0/psi/psi_ov_ficus.mp4 $B/ficus_v0_greg_fix/run/psi/psi_ov_ficus.mp4 $B/ficus_v0_localrot/run/psi/psi_ov_ficus.mp4 \
   $B/ficus_v0_bspline/A/psi/psi_ov_ficus.mp4 $B/ficus_v0_bspline/B/psi/psi_ov_ficus.mp4
cp /tmp/ficus_bspline.mp4 /tmp/ficus_bspline_psi.mp4 $V/
echo FICUS_BSPLINE_VID_DONE
