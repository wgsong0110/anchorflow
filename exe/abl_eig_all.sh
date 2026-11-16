#!/bin/bash
# 절 CM195: ablation 영 에너지 모드 수 (exe/q27_eig.py, 첫 프레임 탄성 헤시안, 강체 = 6 x 연결 성분 수 뺀 값도 기록) 를 네 장면 x 10 판 전부.
#   입력 R2 lbours/final_fix/runs_nd/<장면>_<판>_r1/{pred.pt,q27solver.pt} (기존 def·q1·q27 값과 같은 덤프), 출력 R2 lbours/final_fix/abl_eig/<장면>_<판>/{eig.json,eig.log}
#   사용 (인스턴스 tmux): bash abl_eig_all.sh <q27_eig.py 경로> <작업 폴더>
PYF=$1; W=$2; RN=r2:storage/result/anchorflow/lbours/final_fix/runs_nd; RO=r2:storage/result/anchorflow/lbours/final_fix/abl_eig
PY=${PY:-python}
mkdir -p $W
for s in wolf L000 bread pillow2sofa; do
  for v in def q1 q27 q64 be mid nofpi c075 c15 c20; do
    n=${s}_${v}; O=$W/$n
    rclone lsf $RO/$n/eig.json 2>/dev/null | grep -q . && { echo "[skip] $n"; continue; }
    mkdir -p $O
    q=$(case $v in q1) echo 1;; q27) echo 27;; q64) echo 64;; *) echo 8;; esac)
    kb=$([ $s = bread ] && echo 800 || echo 2400)
    echo "[$(date +%T)] $n q$q kb$kb 내려받기"
    rclone copy $RN/${n}_r1 $O --include pred.pt --include q27solver.pt -q --transfers 4 --multi-thread-streams 8
    echo "[$(date +%T)] $n 계산"
    $PY -u $PYF --pred $O/pred.pt --solver $O/q27solver.pt --frames 0 --q $q --kb $kb --out $O/eig.json > $O/eig.log 2>&1
    echo "[$(date +%T)] $n rc=$? $(grep '프레임' $O/eig.log | tail -1 | cut -c1-300)"
    rm -f $O/pred.pt $O/q27solver.pt
    rclone copy $O $RO/$n -q
  done
done
echo "[$(date +%T)] 끝"; touch $W/ALLDONE; rclone copyto $W/ALLDONE $RO/ALLDONE -q
