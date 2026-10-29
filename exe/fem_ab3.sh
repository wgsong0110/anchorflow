#!/bin/bash
# 절 AB 3 차: 단독 측정 (GPU0, 실행마다 /home/dkta/work/gpu0.lock flock -- 다른 갈래 측정과 직렬화, GPU0 이 빌 때; 카드 1·2 에는 이 갈래 비측정 스윕이 돌 수 있다 -- gpu_at_start.txt)
# 절 AC 빠른 경로 (--cfuse --cfuse_mode 2) + 클램프 5: lego N256 x 3 회, ficus N640 x 2 회 -- 기존 F 갱신 (같은 세션 기준) 과 후보 셋.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OD=/tmp/dkta/lbours
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --lim_sr 5 --cfuse --cfuse_mode 2"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --lim_sr 5 --cfuse --cfuse_mode 2"
declare -A V=([T]="" [navg1]="--fupd navg --fupd_b 1" [nab1b0p05]="--fupd navgbar --fupd_b 1 --fupd_b2 0.05" [nbar0p05]="--fupd nbar --fupd_b 0.05")
for r in 1 2 3; do
  for k in T nab1b0p05 navg1 nbar0p05; do
    EX_TIMED=1 EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh t_l_${k}_n256_r$r 0 $LB ${V[$k]} --N 256 < /dev/null > /dev/null 2>&1
    echo "$(date +%T) t_l_${k}_n256_r$r $(grep -a '롤아웃\]' $OD/fem_ab_lego/t_l_${k}_n256_r$r/v_fem.log | cut -c1-170)"
    if [ $r -le 2 ]; then
      EX_TIMED=1 EX_ROOT=fem_ab_ficus bash exe/fem_ab_run.sh t_f_${k}_n640_r$r 0 $FB ${V[$k]} --N 640 < /dev/null > /dev/null 2>&1
      echo "$(date +%T) t_f_${k}_n640_r$r $(grep -a '롤아웃\]' $OD/fem_ab_ficus/t_f_${k}_n640_r$r/v_fem.log | cut -c1-170)"
    fi
  done
done
python exe/fem_p_period.py --data $FD --pred $OD/fem_ab_ficus/t_f_*/pred.pt > $OD/fem_ab_ficus/period3.txt 2>&1
python exe/fem_ab_table.py $OD/fem_ab_lego/t_l_* > $OD/fem_ab_lego/table3t.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/t_f_* > $OD/fem_ab_ficus/table3t.md
cat $OD/fem_ab_lego/table3t.md $OD/fem_ab_ficus/table3t.md
for d in $OD/fem_ab_*/t_*; do echo "$(basename $d) 시작 $(tr '\n' ' ' < $d/gpu_at_start.txt) 끝 $(tr '\n' ' ' < $d/gpu_at_end.txt)"; done > $OD/fem_ab_lego/t_gpu.txt
rclone copy $OD/fem_ab_lego r2:storage/result/anchorflow/lbours/fem_ab_lego -q
rclone copy $OD/fem_ab_ficus r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo AB3_DONE $(date +%T)
