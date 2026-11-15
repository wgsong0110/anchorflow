#!/bin/bash
# [paper_remeasure2] 프레임 시간 분해: R2 에 남은 시간 판 명령 (card.txt) 을 가져와 prof_wrap 을 앞에 붙이고 프레임을 줄여 한 번 돎. (vast_pr2_boot.sh 가 PR_MODE=prof 일 때 source; cwd = 코드 폴더)
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; O=/dev/shm/dkta/prof; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 WARP_CACHE_PATH=/dev/shm/dkta/warp_cache_0
for spec in "vasedeck stab_D2/g/ours_vasedeck_n1_r1 16" "ficus stab_A/g/ours_ficus_n1_r1 16" "vasedeck_K2 stab_D2/g/ours_vasedeck_n2_r1 16"; do set -- $spec; nm=$1; src=$2; fr=$3
  rclone cat $RX/$src/card.txt | sed -n '1p' | sed 's/^시작 때 다른 프로세스 [0-9]* 개 [0-9-]* [0-9:]* //' > $O/$nm.cmd
  python - "$O/$nm.cmd" "$O/$nm" "$fr" <<'P'
import re, sys
c = open(sys.argv[1]).read().strip()
fr = sys.argv[3]
c = re.sub(r"AF_SP_FR=1,\d+", f"AF_SP_FR=1,{fr}", c) if "AF_SP_FR" in c else c
c = re.sub(r"--frames \d+", f"--frames {fr}", c)
c = re.sub(r"--out \S+$", f"--out {sys.argv[2]}", c)
c = c.replace("exe/startpt_wrap.py", "exe/prof_wrap.py --prof_from 8 --prof_n 6 exe/startpt_wrap.py", 1) if "rotbc_wrap" not in c else c.replace("exe/rotbc_wrap.py --rot_bc 1 exe/startpt_wrap.py", "exe/rotbc_wrap.py --rot_bc 1 exe/prof_wrap.py --prof_from 8 --prof_n 6 exe/startpt_wrap.py", 1)
open(sys.argv[1], "w").write(c)
P
  say "[프로파일] $nm: $(cut -c1-200 $O/$nm.cmd)"
  bash -c "$(cat $O/$nm.cmd)" > $O/$nm.log 2>&1; say "[프로파일] $nm rc=$? $(tail -1 $O/$nm.log | cut -c1-160)"
  rclone copy $O/$nm $RX/prof/$nm -q; rclone copyto $O/$nm.log $RX/prof/$nm.log -q
done
echo "$(date)" | rclone rcat $MK/PROF_END
