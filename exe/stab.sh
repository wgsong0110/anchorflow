#!/bin/bash
# 안정 최소 서브스텝 표 묶음 (doc/pareto.md 절 S; 커밋 $1, 잡 안): R2 lbours/stab/ 의 판 결과를 내려받아 이어 감 -> exe/stab.py -> 5 분마다 R2.
# 마감: ST_UNTIL (epoch, 새 판 안 띄움) · ST_HARD (표·끝 표식) -- 안 주면 지금부터 60 h · 66 h (잡 72 h 하드 리밋 안). ST_ARGS 는 stab.py 에 그대로 (예 "--smoke 3 --methods mpm").
# 출력 /dev/shm/dkta/lbours/stab (ST_OUT 로 바꿈) -> R2 lbours/stab/ (h5 폴더 빼고 전부; pred.pt 덤프 포함), 표 stab_table.md, 끝 표식 stab.END.
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_stab XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext; mkdir -p $CUPY_CACHE_DIR
PY=/home/dkta/.conda/envs/af/bin/python
R=${ST_OUT:-/dev/shm/dkta/lbours/stab}; RR=r2:storage/result/anchorflow/lbours/${R##*/lbours/}
mkdir -p $R
UNTIL=${ST_UNTIL:-$(( $(date +%s) + 60 * 3600 ))}; HARD=${ST_HARD:-$(( $(date +%s) + 66 * 3600 ))}
echo "[stab.sh 시작] $C $(date '+%F %T') -> $R ($RR)" >> $R/stab.log
df -h /dev/shm /tmp | tail -2 >> $R/stab.log; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $R/stab.log
rclone copy $RR $R -q --include "p/*/res.json" --include "p/*/vis.json" --include "gt_*/done" --include "gt_*/png/**"          # 이어 돌리기 (판 결과만)
sync_r2() { rclone copy $R $RR -q --exclude "*/h5/**" --exclude "code/**"; }
( while true; do sleep 300; sync_r2; done ) & SY=$!
$PY -u exe/stab.py --out $R --until $UNTIL --hard $HARD ${ST_ARGS:-} >> $R/stab_main.log 2>&1
echo "[stab.sh 끝] rc=$? $(date '+%F %T')" >> $R/stab.log
kill $SY; sync_r2
