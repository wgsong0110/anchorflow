#!/bin/bash
# 절 CM16 영상 합성 (로컬; 절 CM15.3 에서 인스턴스 합성이 멈춘 일 때문에 렌더는 인스턴스, 합성은 여기서): R2 marks/nclamp.END 를 기다린 뒤
# 정답·비교 칸 (절 CM15 vres o_L000·vs_<씬>, 절 CM10 barx x_wolf, 절 CM12 barstick a_wolf) 과 nclamp nc_<씬> 렌더를 받아 exe/fem_vres_compose.py 그대로 -> R2 result/anchorflow/v20261009/fem_nclamp_{lego_L000,wolf}_cmp.mp4
# 사용: bash exe/fem_nclamp_compose_local.sh <작업 폴더>
set -u
WD=${1:?}; mkdir -p $WD; cd "$(dirname "$0")/.." || exit 1
R=r2:storage/result/anchorflow; F=$R/lbours/fem_cm
until rclone lsf $F/marks/nclamp.END >/dev/null 2>&1 && [ -n "$(rclone lsf $F/marks/nclamp.END 2>/dev/null)" ]; do sleep 60; done
g() { rclone copyto "$1" "$2" -q; }
g $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $WD/gt_wolf.mp4; g $R/lbours/fem2_lego/gt_lego.mp4 $WD/gt_L000.mp4
g $F/code/NotoSansCJKkr-Bold.otf $WD/fb.otf; g $F/code/NotoSansCJKkr-Regular.otf $WD/fr.otf
for p in vres/o_L000 vres/vs_L000 vres/vs_wolf barx/x_wolf barstick/a_wolf nclamp/nc_L000 nclamp/nc_wolf; do
  d=$WD/$(echo $p | tr / _); mkdir -p $d; g $F/$p/summary.json $d/summary.json; rclone copy $F/$p/vj $d/vj -q --filter "- fem_barx*" --filter "+ fem_*.mp4" --filter "- *"; done
# 클램프 판 마지막 프레임이 비유한 (발산 프레임 덤프 = 빈 렌더) 이면 그 프레임을 빼서 정지 화면이 마지막 유한 프레임이 되게 (풀이 결과는 그대로, 보이는 칸만)
for t in L000 wolf; do d=$WD/nclamp_nc_$t; g $F/nclamp/nc_$t/fem.jsonl $d/fem.jsonl; v=$(ls $d/vj/fem_*.mp4)
  if tail -1 $d/fem.jsonl | grep -q '"err": NaN'; then n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $v)
    ffmpeg -v error -y -i $v -frames:v $((n - 1)) -c:v libx264 -crf 18 -pix_fmt yuv420p $v.cut.mp4 && mv $v.cut.mp4 $v && echo "[마지막 비유한 프레임 뺌] nc_$t $n -> $((n - 1))"; fi; done
CAP="HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 결정적 · (B) = 장벽 + 띠 고착 + --y0eb · 저항 강 = 절 CM15 −z 저항 k 9 · 노드 법선 클램프 = 사용자 지시 하드 클램프: 걸음 시작 노드 위치가 평면 이하인 노드의 걸음 변위 법선 성분 ≥ 0 (활성 집합, 접선 자유, 마찰 없음)"
python3 exe/fem_vres_compose.py $WD/fem_nclamp_lego_L000_cmp.mp4 "lego L000 80 프레임 · PCG 상한 4 · $CAP" $WD/fb.otf $WD/fr.otf $WD/tmp_L000 \
  "정답 (MPM)" $WD/gt_L000.mp4 - "(B) 장벽 (표본점)" $WD/vres_o_L000/vj/fem_lego.mp4 $WD/vres_o_L000/summary.json \
  "−z 저항 강 (k 9)" $WD/vres_vs_L000/vj/fem_lego.mp4 $WD/vres_vs_L000/summary.json "노드 법선 클램프" $WD/nclamp_nc_L000/vj/fem_lego.mp4 $WD/nclamp_nc_L000/summary.json > $WD/compose_L000.log 2>&1
python3 exe/fem_vres_compose.py $WD/fem_nclamp_wolf_cmp.mp4 "wolf 50 프레임 · PCG 상한 5 ((A) 3) · $CAP" $WD/fb.otf $WD/fr.otf $WD/tmp_wolf \
  "정답 (PG 모래)" $WD/gt_wolf.mp4 - "(A) 가우스 점 고착" $WD/barstick_a_wolf/vj/fem_wolf.mp4 $WD/barstick_a_wolf/summary.json "(B) 장벽 (표본점)" $WD/barx_x_wolf/vj/fem_wolf.mp4 $WD/barx_x_wolf/summary.json \
  "−z 저항 강 (k 9)" $WD/vres_vs_wolf/vj/fem_wolf.mp4 $WD/vres_vs_wolf/summary.json "노드 법선 클램프" $WD/nclamp_nc_wolf/vj/fem_wolf.mp4 $WD/nclamp_nc_wolf/summary.json > $WD/compose_wolf.log 2>&1
for n in fem_nclamp_lego_L000_cmp.mp4 fem_nclamp_wolf_cmp.mp4; do
  if [ -s $WD/$n ]; then rclone copyto $WD/$n $R/v20261009/$n -q && rclone check $WD $R/v20261009 --include $n --one-way -q && echo "[올림 확인] $n $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv $WD/$n | tr '\n' ' ')" || echo "[올림 확인 실패] $n"
  else echo "[합성 실패] $n"; fi; done
echo "[끝] $(cat $WD/compose_*.log | tr '\n' ' ')"
