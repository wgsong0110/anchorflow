#!/bin/bash
# 절 CM55 s1 앞 단계: 수박 (b) 칸 덤프 (s0k dM_b) 그림 -- 빼는 면 (3x3x3 열림) 위치. 덤프는 R2 에서 /dev/shm 로 받아 그린 뒤 지움.
O=$1; RO=$2; PY=/home/dkta/.conda/envs/af/bin/python
D=/dev/shm/dkta/ghost5_fig; mkdir -p $D $O/fig
for f in 10 14 20 50 100; do rclone copyto $RO/dM_b/g4cells_f$f.npz $D/g4cells_f$f.npz -q; done
$PY exe/ghost5_wm_plot.py $O/fig/ghost5_wm_b_removed.png $D/g4cells_f10.npz $D/g4cells_f14.npz $D/g4cells_f20.npz $D/g4cells_f50.npz $D/g4cells_f100.npz
rclone copy $O/fig $RO/fig -q
rm -rf $D
