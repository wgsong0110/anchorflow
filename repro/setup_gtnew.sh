#!/bin/bash
# 기록된 실행의 부팅 스크립트가 하던 경로 준비 (exe/vast_ffx.sh 등과 같음): 새 PG 장면 정답 meta.json 이 적어 둔
# $W/gtnew/<장면> (정답 생성 때 경로) 를 $W/newsc 의 h5 로 이어 준다. 렌더 훅 (lib/anchorflow/rtrender1.py) 이 읽는다.
# 사용: W=<작업 폴더> bash repro/setup_gtnew.sh
set -eu
W=${W:?W=<work dir>}
mkdir -p $W/gtnew
for k in vasedeck plane pillow2sofa; do
  [ -d $W/newsc/${k}_h5 ] && [ ! -e $W/gtnew/$k ] && ln -s $W/newsc/${k}_h5 $W/gtnew/$k
  [ -d $W/newsc/gt_$k ] && [ ! -e $W/gtnew/gt_$k ] && ln -s $W/newsc/gt_$k $W/gtnew/gt_$k
done
echo "gtnew 준비: $(ls $W/gtnew | tr '\n' ' ')"
