#!/bin/bash
# 절 AG 진단 실행 (비측정, 카드마다 동시): 목록 파일 줄 = <카드> <태그> <씬> <N> <프레임> [추가 인자...] -- "--noplast" 이면 --plast 를 뺀다 (같은 씬 탄성 FCR).
# 출력 /tmp/dkta/lbours/fem_ag_diag/<태그> (R2 같은 경로). 사용: bash exe/fem_ag_diag.sh <목록 파일>
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
OS=/tmp/dkta/lbours/fem_ag_sum; mkdir -p $OS
echo "[진단] $(git log -1 --oneline) $(date '+%F %T')"
one() {
  local G=$1 TG=$2 S=$3 N=$4 FRM=$5; shift 5
  ag_scene $S; local A="$SARGS"
  if [ "$1" = "--noplast" ]; then A=${A/--plast/}; shift; fi
  EX_ROOT=fem_ag_diag bash exe/fem_ab_run.sh $TG $G $A $REC --no_dump --N $N --frames $FRM "$@" < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ag_diag/$TG/v_fem.log | cut -c1-190)"
}
declare -A Q
while read -r G rest; do [ -z "$G" ] && continue; Q[$G]+="$rest"$'\n'; done < $1
for G in "${!Q[@]}"; do
  ( while read -r line; do [ -z "$line" ] && continue; one $G $line; done <<< "${Q[$G]}" ) &
done
wait
rclone copy /tmp/dkta/lbours/fem_ag_diag r2:storage/result/anchorflow/lbours/fem_ag_diag -q
echo "AG_DIAG_DONE $(date +%T)"
