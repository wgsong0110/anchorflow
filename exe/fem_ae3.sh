#!/bin/bash
# 절 AE 3 차: lego V_p·m_p 를 정답 float32 위치로 (--vp_h5) -- 연기 시험 뒤 스윕 (AE_SET=4, 10 회, 카드 1~6 x 일꾼 3), 비측정.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OD=/tmp/dkta/lbours; OL=$OD/fem_ae_lego
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke_h5 1 --mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2 --N 256 --lim_sr 5 --frames 3 --vp_h5 /home/dkta/work/lbdata/test_low/L000/sim_0000000000.h5 < /dev/null > /dev/null 2>&1
grep -a "절 AE\|롤아웃\]\|Error\|assert" $OL/smoke_h5/v_fem.log | cut -c1-250
[ -f $OL/smoke_h5/EX_DONE ] || { echo "[연기 시험 실패]"; tail -5 $OL/smoke_h5/v_fem.log; exit 1; }
for G in 1 2 3 4 5 6; do for k in 0 1 2; do
  AE_SET=4 AE_R=10 bash exe/fem_ae_gpu.sh $G $(( (G - 1) * 3 + k )) 18 exe/fem_ae_jobs.py < /dev/null > $OL/w4_${G}_$k.log 2>&1 &
done; done
wait
python exe/fem_ab_table.py $OL/ae_l_h5* > $OL/table4.md; cat $OL/table4.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ae_lego -q
echo AE3_DONE $(date +%T)
