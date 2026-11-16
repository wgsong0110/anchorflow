#!/bin/bash
# 절 CM195: ablation E (서브스텝 N) 영 에너지 모드 -- 표 E 와 같은 덤프 (Lego·pillow2sofa = final_fix/runs/<장면>_ssN.dump.wel_r1,
#   bread = runs_wolfN/bread_ssN.dump.bs2_r1, wolf = runs_wolfN/wolf_ss1.dump.bs2_r1 · wolf_ssN.dump.bs2.gran_r1 (N>1, 절 CM196)) 의 pred.pt 0 번 프레임.
#   E 덤프에는 q27solver.pt 가 없어 같은 장면 기본 판 (runs_nd/<장면>_def_r1/q27solver.pt: 칸 행렬·입자 부피·μλ, N 과 무관) 을 씀.
#   칸마다 시작 전 GPU 메모리 여유 >= MINFREE MiB. 출력 R2 lbours/final_fix/abl_eig/<장면>_ssN.  사용: bash abl_eig_E.sh <q27_eig.py> <작업 폴더>
PYF=$1; W=$2; MINFREE=${MINFREE:-20480}; PY=${PY:-python}
F=r2:storage/result/anchorflow/lbours/final_fix; RO=$F/abl_eig; mkdir -p $W
for s in wolf L000 bread pillow2sofa; do for N in 1 2 4 8 16 32 64 128; do
  n=${s}_ss$N
  case $s in L000|pillow2sofa) src=$F/runs/${s}_ss$N.dump.wel_r1;; bread) src=$F/runs_wolfN/bread_ss$N.dump.bs2_r1;;
    wolf) [ $N = 1 ] && src=$F/runs_wolfN/wolf_ss1.dump.bs2_r1 || src=$F/runs_wolfN/wolf_ss$N.dump.bs2.gran_r1;; esac
  rclone lsf $RO/$n/eig.json 2>/dev/null | grep -q . && { echo "[skip] $n"; continue; }
  O=$W/$n; mkdir -p $O; kb=$([ $s = bread ] && echo 800 || echo 2400)
  echo "[$(date +%T)] $n <- $src 내려받기"
  rclone copy $src $O --include pred.pt -q --transfers 4; rclone copy $F/runs_nd/${s}_def_r1 $O --include q27solver.pt -q
  echo "$src" > $O/src.txt
  until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge $MINFREE ]; do echo "[$(date +%T)] 메모리 여유 부족, 기다림"; sleep 60; done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u $PYF --pred $O/pred.pt --solver $O/q27solver.pt --frames 0 --q 8 --kb $kb --out $O/eig.json > $O/eig.log 2>&1
  rc=$?; rm -f $O/pred.pt $O/q27solver.pt
  [ -s $O/eig.json ] && rclone copy $O $RO/$n -q
  echo "[$(date +%T)] $n rc=$rc $(grep '프레임' $O/eig.log | tail -1 | cut -c1-300)"
done; done
echo "[$(date +%T)] 끝"; touch $W/ALLDONE
