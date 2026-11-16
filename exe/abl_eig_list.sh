#!/bin/bash
# 절 CM195: 지정한 영 에너지 모드 칸만 GPU 일꾼 하나로 차례로 (expandable_segments), 칸마다 시작 전 GPU 메모리 여유 >= MINFREE MiB (기본 20480) 를 기다림.
#   사용: bash abl_eig_list.sh <q27_eig.py> <작업 폴더> <장면_판> ...   출력 R2 lbours/final_fix/abl_eig/<장면_판>
PYF=$1; W=$2; shift 2; MINFREE=${MINFREE:-20480}
RN=r2:storage/result/anchorflow/lbours/final_fix/runs_nd; RO=r2:storage/result/anchorflow/lbours/final_fix/abl_eig
PY=${PY:-python}; mkdir -p $W
for n in "$@"; do
  rclone lsf $RO/$n/eig.json 2>/dev/null | grep -q . && { echo "[skip] $n"; continue; }
  O=$W/$n; mkdir -p $O; s=${n%%_*}; v=${n#*_}
  q=$(case $v in q1) echo 1;; q27) echo 27;; q64) echo 64;; *) echo 8;; esac); kb=$([ $s = bread ] && echo 800 || echo 2400)
  echo "[$(date +%T)] $n q$q kb$kb 내려받기"
  rclone copy $RN/${n}_r1 $O --include pred.pt --include q27solver.pt -q --transfers 4
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge $MINFREE ]; do echo "[$(date +%T)] 메모리 여유 부족, 기다림"; sleep 60; done
  echo "[$(date +%T)] $n 계산 (여유 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1))"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $PYF --pred $O/pred.pt --solver $O/q27solver.pt --frames 0 --q $q --kb $kb --out $O/eig.json > $O/eig.log 2>&1
  rc=$?; rm -f $O/pred.pt $O/q27solver.pt
  [ -s $O/eig.json ] && rclone copy $O $RO/$n -q
  echo "[$(date +%T)] $n rc=$rc $(grep '프레임' $O/eig.log | tail -1 | cut -c1-300)"
done
echo "[$(date +%T)] 끝"; touch $W/ALLDONE
