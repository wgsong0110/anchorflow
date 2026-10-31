#!/bin/bash
# 절 AG.7 (조정자 10-31): 절 AG 씬 명령 (fem_ag_scene.sh 그대로) + 관성 질량 하한 --ah_mlow (절 AH.18), wolf 는 node_h 를 CFL 을 만족하게 키운 판 (메시 해상도 변경 = 정의 변경).
# AHAG_RUNS="씬:N:꼬리:추가인자(쉼표=공백) ..." 를 카드 AHAG_G (기본 1..6) 에 나눠 동시에 (비측정). AHAG_DUMP=1 이면 --dump_f32 --dump_div.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
GS=(${AHAG_G:-1 2 3 4 5 6}); i=0
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"
one() { local S=$1 N=$2 SF=$3 X=$4 G=$5; ag_scene $S
  case " $X " in *" --drop_plast "*) SARGS=${SARGS/ --plast/}; X=${X/--drop_plast/} ;; esac   # 절 AG.18 대조: 재질 없이 탄성으로 (씬 인자의 --plast 를 뺀다)
  local TG=ag_${S}_n${N}${SF}_r1 D="--no_dump"; [ -n "${AHAG_DUMP:-}" ] && D="--dump_f32 --dump_div"
  # 씬 인자의 node_h 를 바꾸는 판: X 안에 --node_h 가 있으면 그것이 뒤에 와서 우선 (argparse 마지막 값)
  EX_ROOT=fem_ag_$S bash exe/fem_ab_run.sh $TG $G $SARGS $REC $D --N $N $X < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]\|CFL\]' /tmp/dkta/lbours/fem_ag_$S/$TG/v_fem.log | cut -c1-230 | tr '\n' ' ')"
}
for R in ${AHAG_RUNS:?}; do
  IFS=: read -r S N SF X <<< "$R"; X=${X//,/ }
  G=${GS[$((i % ${#GS[@]}))]}
  one $S $N $SF "$X" $G > /tmp/dkta/ahag_$i.log 2>&1 &
  i=$((i + 1)); [ $((i % ${#GS[@]})) -eq 0 ] && wait
done
wait
cat /tmp/dkta/ahag_*.log; rm -f /tmp/dkta/ahag_*.log
echo "AHAG_DONE ${AHAG_ID:-} $(date +%T)"; mkdir -p /tmp/dkta/lbours/fem_ag_sum; touch /tmp/dkta/lbours/fem_ag_sum/AHAG_${AHAG_ID:-x}_DONE
rclone copy /tmp/dkta/lbours/fem_ag_sum/AHAG_${AHAG_ID:-x}_DONE r2:storage/result/anchorflow/lbours/fem_ag_sum -q
