#!/bin/bash
# 절 AI: 클램프 끈 발산 판에 절 AJ 발산 셀 진단 (--lagdiag --divtrack, 계산 불변) + F 덤프 -> exe/fem_dc_cells.py (발산 셀 입자 σ1·σ3·이웃 퍼짐 vs 정답).
# 비측정, 카드 1~6 중 가장 덜 쓰는 카드. 출력 /tmp/dkta/lbours/fem_ai_dc -> R2 lbours/fem_ai_dc.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
O=/tmp/dkta/lbours/fem_ai_dc; RR=r2:storage/result/anchorflow/lbours/fem_ai_dc; mkdir -p $O
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $1<=6' | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "[커밋] $(git log -1 --oneline) 카드 $G $(date '+%F %T')"
while read -r ROOT TG ARGS; do
  case $TG in L000_a0b0_v_free_r1|L000_a2b2_v_free_r1|0019_a0b1_n_free_r1|wm_a1b1_v_free_r1) ;; *) continue;; esac
  T2=dc_$TG
  ARGS=$(echo "$ARGS" | sed 's/--no_dump//; s/--ai_stat [0-9]*//; s/--ai_stat_frames [0-9,]*//; s/--fspread [0-9]*//')
  EX_ROOT=fem_ai_dc bash exe/fem_ab_run.sh $T2 $G $ARGS --lagdiag --divtrack --dump_f32 --dump_div < /dev/null > /dev/null 2>&1
  D=$(echo "$ARGS" | sed 's/.*--data \([^ ]*\).*/\1/')
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_dc_cells.py --run $O/$T2 --data $D < /dev/null > $O/$T2/dc.log 2>&1
  echo "== $T2 $(grep -a '롤아웃\]' $O/$T2/v_fem.log | cut -c1-140)"; tail -12 $O/$T2/dc.log
  rclone copy $O/$T2 $RR/$T2 -q
done < <(python exe/fem_ai_jobs.py s1; python exe/fem_ai_jobs.py wm)
echo "[끝] $(date +%T)"
