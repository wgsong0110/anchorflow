#!/bin/bash
# 절 AH 비측정 스윕: 연기 시험 (3 프레임) 뒤 작업 목록 (fem_ah_jobs.py, AH_SET/AH_V/AH_N/AH_R 환경) 을 카드 AH_G (기본 "1 2 3 4 5 6") x 일꾼 AH_K (기본 3) 로 동시에.
# 출력 /tmp/dkta/lbours/<EX_ROOT>/<태그> (R2 lbours/<EX_ROOT>/<태그>, 실행마다 2 분 간격 + 끝 동기화). 끝 표식 /tmp/dkta/lbours/fem_ah_lego/SWEEP_<AH_ID>_DONE (+ R2).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
ID=${AH_ID:?AH_ID}; GS=${AH_G:-"1 2 3 4 5 6"}; KW=${AH_K:-3}
OD=/tmp/dkta/lbours/fem_ah_lego; mkdir -p $OD
export AF_DATA=/tmp/dkta/af_data; [ -f $AF_DATA/0019.pt ] || export AF_DATA=/home/dkta/work/lbpt/test
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 스윕 $ID 판 ${AH_SET:-1}/${AH_V:-전부} N배 ${AH_N:-1} 카드 $GS x $KW, 작업 $(python exe/fem_ah_jobs.py | wc -l)"
if [ -n "${AH_SMOKE:-}" ]; then
  G0=$(echo $GS | cut -d' ' -f1)
  EX_ROOT=fem_ah_lego bash exe/fem_ab_run.sh smoke_$ID $G0 $(python exe/fem_ah_jobs.py | head -1 | cut -d' ' -f3- | sed 's/--frames 80/--frames 14/') < /dev/null
  [ -f $OD/smoke_$ID/EX_DONE ] || { echo "[연기 시험 실패]"; tail -20 $OD/smoke_$ID/v_fem.log; exit 1; }
  tail -3 $OD/smoke_$ID/fem.jsonl | cut -c1-400
fi
NG=$(echo $GS | wc -w); K=$((NG * KW)); i=0
for G in $GS; do for k in $(seq 0 $((KW - 1))); do bash exe/fem_af_gpu.sh $G $i $K exe/fem_ah_jobs.py < /dev/null > $OD/w_${ID}_${G}_$k.log 2>&1 & i=$((i + 1)); done; done
wait
cat $OD/w_${ID}_*.log | grep -v "^\[일꾼" | sort | cut -c1-200
echo "SWEEP_${ID}_DONE $(date +%T)"; touch $OD/SWEEP_${ID}_DONE; rclone copy $OD/SWEEP_${ID}_DONE r2:storage/result/anchorflow/lbours/fem_ah_lego -q
