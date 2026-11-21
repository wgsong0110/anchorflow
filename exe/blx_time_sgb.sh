#!/bin/bash
# [blx] Spring-Gaus bread 단독 시간 3 회 (RTX A6000, 다른 에이전트 인스턴스의 대기열에서 한 작업으로; 새 인스턴스 금지 11-16).
#   준비 판 = R2 lbours/final_pa/blx/SGB/p/springgaus_bread_prep (sg_train.json 의 절대 경로 때문에 같은 자리 /dev/shm/dkta/lbours/paper_remeasure2/stab/p 에 받음;
#   다른 이름 폴더라 그 인스턴스의 다른 준비 판과 안 겹침). 측정 = pr2_grid one_run 과 같은 규칙 (시작 때 카드 0 의 다른 계산 프로세스 0 을 기다리고, 실행 중 5 초마다 기록),
#   결과 R2 lbours/final_pa/blx/TB/g/springgaus_bread_n<N>_r<k>/{summary.json,log.txt,card.txt}, 끝 표식 marks_TB/TIME_DONE.
#   사용: bash blx_time_sgb.sh <커밋> (환경: stab_bl_setup 이 끝난 인스턴스 -- /root/pr/bl_env.sh 또는 ST_SG_REPO·ST_SG_PYPATH)
set -u
H=${1:?커밋}; R=r2:storage/result/anchorflow; OX=$R/lbours/final_pa/blx
if [ -z "$(rclone lsf $OX/SGB/p/springgaus_bread_prep/res.json 2>/dev/null)" ]; then echo "[blx] 준비 판 아직 없음 (SGB 학습 중) -- 할 일 없이 끝 (rc 3)"; exit 3; fi
rclone lsf $OX/marks_TB/TIME_DONE >/dev/null 2>&1 && [ -n "$(rclone lsf $OX/marks_TB/TIME_DONE 2>/dev/null)" ] && { echo "[blx] 이미 끝"; exit 0; }
PY=/home/dkta/.conda/envs/af/bin/python; [ -x $PY ] || PY=python
[ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
D=/workspace/blx_code_$H; mkdir -p $D; rclone cat $OX/code/blx_$H.tar | tar -x -C $D
PO=/dev/shm/dkta/lbours/paper_remeasure2/stab/p; F=$PO/springgaus_bread_prep; mkdir -p $PO
rclone copy $OX/SGB/p/springgaus_bread_prep $F -q --transfers 16 --exclude "mv/**"
AG=/dev/shm/dkta/ag_data; mkdir -p $AG; [ -s $AG/bread.pt ] || rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q
N=$($PY -c "import json;print(json.load(open('$F/sg_train.json'))['n_step'])")
X=/workspace/blxt; mkdir -p $X
u0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | head -1 | cut -d, -f2 | tr -d ' ')
others() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$u0 '$1==u{print $2}'; }
cd $D
for k in 1 2 3; do
  O=$X/springgaus_bread_n${N}_r$k; mkdir -p $O
  [ -s $O/summary.json ] && continue
  while [ -n "$(others)" ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others | wc -l) 개 $(date -u '+%F %T')" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 PYTHONPATH=${ST_SG_PYPATH:-} $PY -u exe/stab_sg.py --mode run --sg ${ST_SG_REPO:-/dev/shm/dkta/ext/Spring-Gaus} --fit $F --N $N --out $O \
    --data $AG/bread.pt --tag bread --frames 300 --floor none --cfg_bc --pg_vdamp > $O/log.txt 2>&1 &
  P=$!; seen=""
  while kill -0 $P 2>/dev/null; do seen="$seen $(others | grep -v "^$P$")"; sleep 5; done
  echo "끝 $(date -u '+%F %T') 실행 중 카드 0 의 다른 계산 프로세스: $(echo $seen | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')" >> $O/card.txt
  rclone copy $O $OX/TB/g/$(basename $O) -q
done
rclone check $X $OX/TB/g --one-way --size-only -q && echo "$(date -u)" | rclone rcat $OX/marks_TB/TIME_DONE
