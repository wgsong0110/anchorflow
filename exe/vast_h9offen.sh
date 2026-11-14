#!/bin/bash
# wolf H9 / H9 − 확장·S_구속·C 두 판의 입자별 탄성 에너지 색 영상 (exe/vast_cvben.sh 방식; 같은 인스턴스에서 exe/vast_h9offvid.sh 뒤에 돎).
#   색 범위 log10 psi: -3 ~ log10(1.4e9) 고정. 덤프 F 가 없으면 [F 없음] 으로 멈춤. 결과 R2 lbours/wolfcv/cm139/h9offen, 끝 표식 cm139/h9offen.END
set -u
H=3932a75; R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/cm139; W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; GTW=$AG/gt_wolf; V=$W/h9offvid
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python; cd /root/sv/code/$H || exit 1
O=$V/en; mkdir -p $O/psi $O/vid
rclone copyto $R/lbours/wolfcv/cm131/h9/pred.pt $V/h9.pred.pt -q
python -c "
import torch
for n in ['h9','Kacdhnz','Kacdhnz_2']:
    D=torch.load('$V/'+n+'.pred.pt',map_location='cpu',weights_only=False,mmap=True); print(n,'F' in D, D['F'].shape if 'F' in D else list(D.keys()))
" 2>&1 | tee $O/fcheck.log
grep -q "False" $O/fcheck.log && { echo "F 없음 $(date)" | rclone rcat $RX/h9offen.END; rclone copy $O $RX/h9offen -q; exit 1; }
[ -d $V/h5_h9 ] || $PY -u exe/lb_roll_to_h5.py --dump $V/h9.pred.pt --traj $GTW --out $V/h5_h9 > $V/h9.h5.log 2>&1
$PY -u exe/cvben_psi.py $GTW $AG/wolf.pt $O/psi -3 9.146128 h9=$V/h9.pred.pt a=$V/Kacdhnz.pred.pt b=$V/Kacdhnz_2.pred.pt > $O/psi.log 2>&1
grep -q CVBEN_PSI_DONE $O/psi.log || { echo "psi 실패 $(date)" | rclone rcat $RX/h9offen.END; rclone copy $O $RX/h9offen -q; exit 1; }
RC="exe/lb_render.py --config $GTW/config.json --shape wolf"
$PY -u $RC --sim $GTW --frames 50 --colors $O/psi/col_gt.npy --out $O/vid/gt_en.mp4 < /dev/null > $O/vid/r_gt.log 2>&1
for nm in h9:h9 a:Kacdhnz b:Kacdhnz_2; do k=${nm%%:*}; d=${nm##*:}; $PY -u $RC --sim $V/h5_$d --frames 50 --colors $O/psi/col_$k.npy --out $O/vid/${k}_en.mp4 < /dev/null > $O/vid/r_$k.log 2>&1; done
rclone copy $O $RX/h9offen -q --include "vid/**" --include "*.log" --include "psi/en.json"
echo "끝 $(date)" | rclone rcat $RX/h9offen.END
