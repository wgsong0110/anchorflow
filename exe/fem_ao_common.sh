#!/bin/bash
# 절 AO 공통 설정 시험 (잡 안): 탄성 씬 (lego L000 + 0000..0019, ficus F_pg_v0) 을 실시간 판 (fem_ao_rt_sim.py, 모래 아니면 절 AN RTSolver 그대로) 으로
# 같은 걸음 간격 AO_DT (s) 에서 (K = ceil(frame_dt / AO_DT)), 셀 AO_CELL. 빈 카드 하나 (4·5·6 중, 실행 동안 그 카드 단독 여부 기록) 에서 순서대로.
# 사용: AO_DT=0.01 AO_CELL=1 bash exe/fem_ao_common.sh <이름> "<추가 인자>"
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/tmp/dkta/cache/cupy XDG_CACHE_HOME=/tmp/dkta/cache TORCH_EXTENSIONS_DIR=/tmp/dkta/cache/torch_ext
NM=$1; EXTRA=$2; R=/tmp/dkta/lbours/fem_ao/$NM; RR=r2:storage/result/anchorflow/lbours/fem_ao/$NM; mkdir -p $R
G=""
for w in $(seq 480); do
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > $R/smi.txt; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader > $R/apps.txt
  G=$(awk -F', ' 'FILENAME ~ /apps/ {b[$1]=1; next} {m=$3; sub(/ MiB/, "", m); if (!($2 in b) && ($1 == 4 || $1 == 5 || $1 == 6) && m + 0 < 2000) {print $1; exit}}' $R/apps.txt $R/smi.txt)
  [ -n "$G" ] && break; sleep 30
done
[ -z "$G" ] && { echo "[오류] 빈 카드 없음"; exit 1; }
UU=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
echo "카드 $G $(cat .done) $(date '+%F %T') dt ${AO_DT:-0.01} 셀 ${AO_CELL:-1} 인자 $EXTRA" > $R/info.txt
for t in L000 $(seq -f %04g 0 19) ficus; do
  O=$R/$t; mkdir -p $O
  if [ $t = ficus ]; then DA="--data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --floor none --cfg_bc --region_density --pg_vdamp --dump"
  elif [ $t = L000 ]; then DA="--data /home/dkta/work/one/L000/L000.pt --frames 80 --floor vel"
  else DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --floor vel"; fi
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_ao_rt_sim.py --out $O --tag $t $DA --cell ${AO_CELL:-1} --dt_step ${AO_DT:-0.01} --mlow 0 $EXTRA < /dev/null > $O/log.txt 2>&1 &
  PP=$!
  while kill -0 $PP 2>/dev/null; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$UU -v t=$(date +%s) '$1==u{print t, $2}' >> $O/apps_during.txt; sleep 5; done
  wait $PP
  NF=$(awk -v p=$PP '$2!=p{print $2}' $O/apps_during.txt | sort -u | wc -l)
  echo "$t 다른프로세스 $NF $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -1 | cut -c1-200)" | tee -a $R/run.out
done
[ -f $R/ficus/pred.pt ] && $PY -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred $R/ficus/pred.pt > $R/ficus/period.txt 2>&1
tail -3 $R/ficus/period.txt
rclone copy $R $RR -q
