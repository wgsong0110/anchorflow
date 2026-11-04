#!/bin/bash
# 절 AN.14 판정 j1 비교 영상용 덤프·렌더 (lego 0003·0016, ficus; 판 bej·hhj): j1 과 같은 명령에 --dump --dump_F 만 더해 다시 돈다 (--det 라 결과 같아야 함 -> x_sha1 대조).
# 빈 카드 (1~6 중 계산 프로세스 0, 남의 claim 없음) 하나에 하나씩 6 개 -> 같은 카드에서 h5 변환 + 3DGS 알파블렌딩 렌더 (exe/lb_render.py) -> R2 lbours/fem_an/j1vid/<판>_<궤적>, 끝 표식 j1vid/END.
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_imp XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
W=/home/dkta/work; R=/dev/shm/dkta/lbours/fem_an/j1vid; RR=r2:storage/result/anchorflow/lbours/fem_an/j1vid; CL=/dev/shm/dkta/anerr_claim
mkdir -p $R $CL $CUPY_CACHE_DIR
C=$(cat .done); echo "[j1vid] 커밋 $C $(date '+%F %T')"; df -h /tmp /dev/shm | tail -2
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse"; T="--tolN 1e-2 --tolP 1e-1"
FI=/dev/shm/dkta/anerr_data/F_pg_v0.pt; [ -s $FI ] || FI=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
job() {
  local G=$1 V=$2 t=$3 I=$4; local O=$R/${V}_$t; mkdir -p $O
  if [ $t = ficus ]; then DA="--data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp"; NC="--newton 2 --cg 10"; GTD=$W/lbours/ficus_v0/gt; CFG=$GTD/config.json; SH=ficus
  else DA="--data /tmp/dkta/af_data/$t.pt --frames 80 --floor vel"; NC="--newton 1 --cg 5"; GTD=$W/lbdata/test/$t; CFG=$W/lbdata/config.json; SH=lego; fi
  echo "카드 $G 커밋 $C $(date '+%F %T') 인자 $DA $B --det $I $T $NC --dump --dump_F" > $O/run_info.txt
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_an_rt_sim.py --out $O --tag $t $DA $B --det $I $T $NC --dump --dump_F < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  rclone cat r2:storage/result/anchorflow/lbours/fem_an/j1/${V}_$t/summary.json > $O/j1_summary.json 2>/dev/null
  echo "x_sha1 이번 $(grep -o '"x_sha1": "[^"]*"' $O/summary.json) j1 $(grep -o '"x_sha1": "[^"]*"' $O/j1_summary.json) / err_mean 이번 $(grep -o '"err_mean": [^,]*' $O/summary.json) j1 $(grep -o '"err_mean": [^,]*' $O/j1_summary.json)" > $O/sha_check.txt
  if [ -f $O/pred.pt ]; then
    $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/h5.log 2>&1
    mkdir -p $O/vj
    CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $O/h5 --config $CFG --shape $SH --out $O/vj/${V}_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  fi
  rclone copy $O $RR/${V}_$t -q --exclude "h5/**"
  echo "$V $t 카드 $G: $(tail -1 $O/run_info.txt) | $(cat $O/sha_check.txt) | 영상 $(ls $O/vj/*.mp4 2>/dev/null)" >> $R/done.txt
  rclone copy $R/done.txt $RR -q
  rm -rf $O/h5
}
for t in 0003 0016 ficus; do for vv in "bej:--integ be" "hhj:--ga hht --rho 0.7"; do V=${vv%%:*}; I=${vv#*:}
  while true; do G=$(free_card); [ -n "$G" ] && break; sleep 30; done
  job $G $V $t "$I" &
  echo $! > $CL/$G; echo "띄움 $V $t 카드 $G $(date +%T)"; sleep 15
done; done
wait
echo "[j1vid 끝] $(date '+%F %T')" > $R/END; rclone copyto $R/END $RR/END -q; rclone copyto $R/done.txt $RR/done.txt -q
