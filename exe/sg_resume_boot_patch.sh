#!/bin/bash
# onstart 가 부팅 스크립트를 받은 직후 부름 (컨테이너 재시작 대비, 사용자 11-16 이어 돌리기 지시):
#  코드 tar 를 푼 바로 뒤에 Spring-Gaus 이어 돌리기 덧판 (R2 <OX>/code/sgresume_overlay_<커밋>.tar = exe/sg_resume.py + 훅 넣은 exe/stab_sg.py) 을 덮고
#  SG_RESUME_R2_MAP 을 내보냄 (저장점 R2 바로 동기화). $1 = 1 이면 pr2_grid --reps 3 -> 1 (final_fix_sg: 2·3 회는 /root/sg_rep23.sh 가 맨 뒤에).
B=/root/vast_fpabl_boot.sh
grep -q sgresume_overlay $B && exit 0
python - $B "${1:-0}" <<'P'
import re, sys
f, r1 = sys.argv[1], sys.argv[2] == "1"; s = open(f).read()
m = re.search(r'^(D=/root/pr/code/\$H; mkdir -p \$D; rclone cat \$OX/code/\S+_\$H\.tar \| tar -x -C \$D)$', s, re.M)
assert m, "코드 풀기 줄 못 찾음"
s = s.replace(m.group(1), m.group(1) + '\nrclone cat $OX/code/sgresume_overlay_$H.tar | tar -x -C $D && echo "[sg_resume] 덧판 $H $(date)" >> /root/sg_resume_boot.log'
              '\nexport SG_RESUME_R2_MAP=/dev/shm/dkta/lbours/paper_remeasure2/stab=$OX/$T', 1)
if r1:
    assert s.count("--reps 3") == 1; s = s.replace("--reps 3", "--reps 1")
open(f, "w").write(s); print("부팅 고침", f)
P
