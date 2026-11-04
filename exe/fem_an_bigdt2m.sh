#!/bin/bash
# 절 BD 보충 (bigdt2m; 커밋 $1, 잡 130567): 묶음 bigdt2 의 정답 MPM wolf·bread 가 PG 원본 config 에 grid_lim 이 없어 KeyError 로 멈춘 것 (exe/gt_sens.py 에 PG decode_param 기본값을 넣어 고침) 을 다시.
# 출력은 따로 /dev/shm/dkta/lbours/fem_an/bigdt2m (R2 bigdt2m/) 에 돌리고, bigdt2 풀이 마지막 줄 (mpmK8_bread) 까지 띄운 뒤 결과를 bigdt2/ 로 덮어씀 (bigdt2 풀이 같은 이름을 다시 덮지 않게),
# 그다음 표 bigdt2_table.md 를 다시 내고 끝 표식 bigdt2m.END. (전체 끝 = bigdt2.END 와 bigdt2m.END 둘 다)
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
R=/dev/shm/dkta/lbours/fem_an/bigdt2m; RR=r2:storage/result/anchorflow/lbours/fem_an/bigdt2m
R2=/dev/shm/dkta/lbours/fem_an/bigdt2; RR2=r2:storage/result/anchorflow/lbours/fem_an/bigdt2
mkdir -p $R /dev/shm/dkta/an_lists
L=/dev/shm/dkta/an_lists/bigdt2m.txt; : > $L
for t in wolf bread; do echo "mpmK0 $t --sub_chk --rpic_cfg" >> $L; done
for K in 1 2 4 8; do for t in wolf bread; do echo "mpmK$K $t --nsub $K --sub_chk --rpic_cfg" >> $L; done; done
echo "[bigdt2m 시작] $C $(date '+%F %T')" > $R/bigdt2m.log; rclone copy $R/bigdt2m.log $RR -q
AN_UNTIL=$(date -d '2026-11-05 12:09 KST' +%s) AN_CARDS="${AN_CARDS:-1 2 3 4 5 6}" AN_PAR=${AN_PAR:-3} AN_GAP=4 bash exe/fem_an_bigdt_pool.sh bigdt2m $L > $R/pool.log 2>&1 &
PP=$!
HARD=$(date -d '2026-11-05 12:20 KST' +%s)
while kill -0 $PP 2>/dev/null && [ "$(date +%s)" -lt "$HARD" ]; do sleep 60; done
# bigdt2 풀이 mpmK8_bread 를 끝낼 때까지 (그 뒤로 bigdt2 풀은 wolf·bread 정답 이름을 안 건드림)
while ! grep -q rc= $R2/mpmK8_bread/run_info.txt 2>/dev/null && [ "$(date +%s)" -lt "$HARD" ]; do sleep 60; done
for d in $R/mpmK*_wolf $R/mpmK*_bread; do
  [ -f $d/run_info.txt ] || continue; n=$(basename $d)
  rm -rf $R2/$n.first; [ -d $R2/$n ] && mv $R2/$n $R2/$n.first                        # 첫 시도 (KeyError) 는 .first 로 남김
  cp -r $d $R2/$n; rclone copy $R2/$n $RR2/$n -q
done
$PY exe/fem_an_bigdt2_table.py $R2 > $R2/bigdt2_table.md 2>&1; rclone copy $R2/bigdt2_table.md $RR2 -q
echo "[bigdt2m 끝] $C $(date '+%F %T') $(kill -0 $PP 2>/dev/null && echo '풀 아직 돎 (마감)')" | tee -a $R/bigdt2m.log > $R2/bigdt2m.END
rclone copy $R/bigdt2m.log $RR -q; rclone copy $R/pool.log $RR -q; rclone copy $R2/bigdt2m.END $RR2 -q
