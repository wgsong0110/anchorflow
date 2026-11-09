#!/bin/bash
# 절 CM80 s1 뒤: 첫 접촉 층별·입자별 진단 (exe/wmcrack1_prof.py 그대로, 읽기만). 사용: bash exe/wmK_post_s1.sh <출력> <R2> <gt_jp> <자료 디렉토리> <GF h5>
O=$1; RO=$2; JP=$3; WMD=$4; GH5=$5
PY=/home/dkta/.conda/envs/af/bin/python
NS=$(for g in H5 K16 K32 K64 K128; do for i in 1 2 3 4; do n=${g}_$i; [ -s $O/$n/wsh/fs_004.npz ] && echo $n; done; done | tr '\n' ',')
$PY exe/wmcrack1_prof.py $O $JP $WMD/wm.pt $GH5 $NS > $O/wmK_prof_s1.md
