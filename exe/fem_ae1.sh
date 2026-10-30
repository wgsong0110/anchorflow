#!/bin/bash
# 절 AE 1 차: (0) V_p 확인 (fem_ae_vp.py, 카드 6) (1) vpmesh 연기 시험 (lego 5 프레임, 카드 1) -> 통과하면 (2) 스윕 (fem_ae_jobs.py, 카드 1~6 x 일꾼 3) -> 표·주기.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OD=/tmp/dkta/lbours; OL=$OD/fem_ae_lego; OF=$OD/fem_ae_ficus; mkdir -p $OL $OF
RL=r2:storage/result/anchorflow/lbours/fem_ae_lego; RF=r2:storage/result/anchorflow/lbours/fem_ae_ficus
echo "[커밋] $(git log -1 --oneline) $(date +%T)"; df -h /home /tmp | tail -2; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
( CUDA_VISIBLE_DEVICES=6 python -u exe/fem_ae_vp.py --data $W/one/L000/L000.pt --gt_h5 $W/lbdata/test_low/L000/sim_0000000000.h5 --out $OL/vp_lego.json > $OL/vp_lego.log 2>&1
  CUDA_VISIBLE_DEVICES=6 python -u exe/fem_ae_vp.py --data $FD --region_density --gt_h5 $W/lbours/ficus_v0/gt/sim_0000000000.h5 --out $OF/vp_ficus.json > $OF/vp_ficus.log 2>&1
  rclone copy $OL $RL -q --include "vp_*"; rclone copy $OF $RF -q --include "vp_*"; echo "[V_p 확인 끝] $(date +%T)" ) &
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke_vpm 1 --mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2 --N 256 --sp vpmesh --lim_sr 5 --frames 5 < /dev/null > /dev/null 2>&1
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke_nvpm 1 --mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2 --N 256 --sp vpmesh --fupd navg --fupd_b 1 --lim_sr 3 --frames 5 < /dev/null > /dev/null 2>&1
grep -a "절 AE vpmesh\|롤아웃\]\|Error" $OL/smoke_vpm/v_fem.log $OL/smoke_nvpm/v_fem.log | cut -c1-300
[ -f $OL/smoke_vpm/EX_DONE ] && [ -f $OL/smoke_nvpm/EX_DONE ] || { echo "[연기 시험 실패] 스윕 안 띄움"; tail -20 $OL/smoke_vpm/v_fem.log; exit 1; }
echo "[연기 시험 통과] 스윕 시작 $(date +%T)"
for G in 1 2 3 4 5 6; do
  for k in 0 1 2; do
    q=$(( (G - 1) * 3 + k ))
    bash exe/fem_ae_gpu.sh $G $q 18 exe/fem_ae_jobs.py < /dev/null > $OL/w_${G}_$k.log 2>&1 &
  done
done
wait
echo "[스윕 끝] $(date +%T)"
python exe/fem_ab_table.py $OL/ae_l_* > $OL/table1.md; python exe/fem_ab_table.py $OF/ae_f_* > $OF/table1.md
cat $OL/table1.md $OF/table1.md
rclone copy $OL $RL -q; rclone copy $OF $RF -q
rclone check $OF $RF --one-way --size-only -q --include "*/pred.pt" && { rm -f $OF/ae_f_*/pred.pt; echo "[정리] R2 확인 뒤 /tmp ficus pred.pt 지움"; }
echo AE1_DONE $(date +%T)
