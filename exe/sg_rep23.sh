#!/bin/bash
# final_fix_sg: Spring-Gaus pillow2sofa 단독 시간 r2·r3 (가장 낮은 우선순위) -- sg_tail (bread 시간·지표)·sg_p2s_mq 가 끝나고 카드 0 이 10 분 연속 비면
# 같은 pr2_grid 를 같은 인자로 다시 (준비 판 res.json·r1 res.json 이 있어 건너뛰고 r2·r3 만), 끝나면 R2 final_pa/bl/P2 로.
source /root/fbenv.sh; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
L=/root/sg_rep23.log; say() { echo "[rep23 $(date '+%F %T')] $*" >> $L; rclone copyto $L r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_sg_rep23.log -q; }
say "대기: sg_tail·sg_p2s_mq·rep1_cut 끝 + 카드 0 이 10 분 연속 빔"
free=0
while [ $free -lt 10 ]; do
  if pgrep -f "sg_tail.sh|sg_p2s_mq.sh|sg_rep1_cut.py|exe/pr2_grid.py" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then free=0; else free=$((free+1)); fi
  sleep 60
done
O=/dev/shm/dkta/lbours/paper_remeasure2/stab
D=$(ls -d /root/pr/code/cf55da5); cd $D
[ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
say "시작 (r1 $(ls $O/g | tr '\n' ' '))"
PR2_PREP_TMO_H=1 python -u exe/pr2_grid.py --out $O --jobs exe/fpabl_jobs_T2.json --reps 3 --scenes L000 \
  --until $(( $(date +%s) + 6 * 3600 )) --hard $(( $(date +%s) + 7 * 3600 )) >> $O/grid_main.log 2>&1
say "pr2_grid 끝 rc=$? $(tail -2 $O/grid_main.log | tr '\n' ' ')"
rclone copy $O r2:storage/result/anchorflow/lbours/final_pa/bl/P2 -q --transfers 8 --exclude "*/h5/**" --exclude "p/**"
rclone copy $O/p/springgaus_pillow2sofa_prep r2:storage/result/anchorflow/lbours/final_pa/bl/P2/p/springgaus_pillow2sofa_prep -q --max-depth 1
say "R2 올림 $(python -c "import json;print([(r['reps'],r['ok'],r['err'],r['ms'],r['ms_all']) for r in json.load(open('$O/grid.json'))])")"
touch /root/sg_rep23.done
