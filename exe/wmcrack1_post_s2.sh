#!/bin/bash
# 절 CM59 s2 뒤: 첫 접촉 층별·입자별 진단 (exe/wmcrack1_prof.py, 읽기만). 사용: bash exe/wmcrack1_post_s1.sh <출력> <R2> <gt_jp> <자료 디렉토리> <GF h5>
O=$1; RO=$2; JP=$3; WMD=$4; GH5=$5
PY=/home/dkta/.conda/envs/af/bin/python
NS=$(for n in dP_1 dR_1 dK16w_1 C_1 V2_1 N_1 N_2 NP_1 NV_1 NS_1; do [ -s $O/$n/wsh/fs_004.npz ] && echo $n; done | tr '\n' ',')
$PY exe/wmcrack1_prof.py $O $JP $WMD/wm.pt $GH5 $NS > $O/wmcrack1_prof_s2.md
