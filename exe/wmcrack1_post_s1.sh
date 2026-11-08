#!/bin/bash
# 절 CM59 s1 뒤: 첫 접촉 층별·입자별 진단 (exe/wmcrack1_prof.py, 읽기만). 사용: bash exe/wmcrack1_post_s1.sh <출력> <R2> <gt_jp> <자료 디렉토리> <GF h5>
O=$1; RO=$2; JP=$3; WMD=$4; GH5=$5
PY=/home/dkta/.conda/envs/af/bin/python
NS=$(grep -v '^#' exe/wmcrack1_spec.txt | awk '$1=="s1" && $4=="r" {print $2}' | while read n; do [ -s $O/$n/wsh/fs_004.npz ] && echo $n; done | tr '\n' ',')
$PY exe/wmcrack1_prof.py $O $JP $WMD/wm.pt $GH5 $NS > $O/wmcrack1_prof_s1.md
