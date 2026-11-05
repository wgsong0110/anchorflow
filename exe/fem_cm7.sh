#!/bin/bash
# 절 CM7 (커밋 $1, 잡 130567 일꾼 w2): 재질 판 노드 상자 넘침 다시 할당 결함 -- (a) 넘친 프레임 MG 층 차원 불법 접근, (b) 다시 할당이 결과를 바꿈.
# 결정적 판에서 용량 2.0 (기본) · 4.0 · 1.0 (이른 다시 할당) · 강제 다시 할당 (AF_FORCE_REDO) 끝 해시 대조 (wolf K 1·K 8, bread K 1·K 8), wolf K 8 기본 용량 완주,
# memcheck: 옛 판 차원 (AF_MGDIMS_OLD=1, 불법 접근 커널 이름) 과 고친 판. 진단 환경 AF_REDO_KEEPW=0 (옛 ω 다시 잼), AF_REDO_COPY=1 (옛 버퍼 내용 옮김).
# 줄: <판> <씬> <환경 (;로 이음, 없으면 ->> <인자...>; 판 이름이 s 로 시작하면 compute-sanitizer memcheck 아래.
C=$1; D=/dev/shm/dkta/impwt/$C; RC=r2:storage/result/anchorflow/lbours/fem_cm/code
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
if [ ! -f $D/.done ]; then
  mkdir -p /dev/shm/dkta/impwt; rm -rf $D.tmp
  rclone copyto $RC/imp_$C.bundle $D.bundle -q && git clone -q $D.bundle $D.tmp && git -C $D.tmp checkout -q $C && mv $D.tmp $D && echo bundle
  echo $C > $D/.done
fi
cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm7 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
NM=cm7; W=/dev/shm/dkta/$NM; mkdir -p $W $CUPY_CACHE_DIR
MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; RR=r2:storage/result/anchorflow/lbours/fem_cm/$NM
R=/dev/shm/dkta/lbours/fem_cm/$NM; CL=/dev/shm/dkta/anerr_claim; mkdir -p $R $CL
UNTIL=$(date -d '2026-11-05 12:05 KST' +%s); HARD=$(date -d '2026-11-05 12:18 KST' +%s)
HW="--ga hht --rho 0.8 --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det"                 # 절 BD bigdt2 wolf 줄 그대로 (K 만 다름)
HB="--ga hht --rho 0.8 --newton 2 --cg 1 --tolN 1e-2 --tolP 1e-1 --det"                 # 절 BD bigdt2 bread 줄 그대로
L=$W/list.txt; cat > $L <<EOT
w8c2 wolf - $HW --K 8
s8old wolf AF_MGDIMS_OLD=1 $HW --K 8 --frames 41
w1c2 wolf - $HW --K 1
w1c4 wolf - $HW --K 1 --cap_mul 4.0
w1c1 wolf - $HW --K 1 --cap_mul 1.0
w1c1nw wolf AF_REDO_KEEPW=0 $HW --K 1 --cap_mul 1.0
w1c1cp wolf AF_REDO_COPY=1 $HW --K 1 --cap_mul 1.0
w1f wolf AF_FORCE_REDO=5,20,35 $HW --K 1
w1fcp wolf AF_FORCE_REDO=5,20,35;AF_REDO_COPY=1 $HW --K 1
w8c4 wolf - $HW --K 8 --cap_mul 4.0
w8c2cp wolf AF_REDO_COPY=1 $HW --K 8
w8f wolf AF_FORCE_REDO=5,20 $HW --K 8 --cap_mul 4.0
b1c2 bread - $HB --K 1
b1c4 bread - $HB --K 1 --cap_mul 4.0
b1c1 bread - $HB --K 1 --cap_mul 1.0
b1f bread AF_FORCE_REDO=50,150 $HB --K 1 --cap_mul 4.0
b1c2cp bread AF_REDO_COPY=1 $HB --K 1
b8c2 bread - $HB --K 8
b8c4 bread - $HB --K 8 --cap_mul 4.0
s8new wolf - $HW --K 8
EOT
rclone copy $L $RR -q
source exe/fem_cm3_scn.sh
free_card() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || return
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r i u; do
    [ "$i" = 0 ] && continue
    case " 1 2 3 4 5 6 " in *" $i "*) ;; *) continue;; esac
    echo "$apps" | grep -q "$u" && continue
    if [ -f $CL/$i ] && kill -0 "$(cat $CL/$i)" 2>/dev/null; then continue; fi
    echo $i; break
  done
}
CS=$(command -v compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer /home/dkta/.conda/envs/af/bin/compute-sanitizer 2>/dev/null | head -1)
run_one() {
  local G=$1 V=$2 t=$3 E=$4; shift 4
  local O=$R/${V}_$t; mkdir -p $O
  cm_scene $t "$@"
  local EV=(); [ "$E" != "-" ] && IFS=';' read -ra EV <<< "$E"
  echo "카드 $G $(cat .done) $(date '+%F %T') ENV $E $SIM $DA $XA" > $O/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/apps_at_start.txt
  case $V in
    s*) if [ -z "$CS" ]; then echo "compute-sanitizer 없음" > $O/log.txt; false
        else env "${EV[@]}" CUDA_VISIBLE_DEVICES=$G timeout 2400 $CS --tool memcheck --print-limit 30 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; fi;;
    *) env "${EV[@]}" CUDA_VISIBLE_DEVICES=$G timeout 1500 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1;;
  esac
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  echo "$V $t (카드 $G) $(grep -a '롤아웃\|rror\|발산\|ERROR SUMMARY' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')" >> $R/done.txt
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt; rclone copy $R/done.txt $RR -q
}
echo "[묶음 $NM] $C $(date '+%F %T') sanitizer=$CS"
while read -r V t E A; do
  [ -z "$V" ] && continue
  [ -f $R/${V}_$t/run_info.txt ] && grep -q rc= $R/${V}_$t/run_info.txt && continue
  [ "$(date +%s)" -ge "$UNTIL" ] && { echo "마감 -- 안 띄움 $V $t" | tee -a $R/skipped.txt; continue; }
  while true; do
    while [ $(jobs -rp | wc -l) -ge 6 ]; do sleep 10; done
    G=$(free_card); [ -n "$G" ] && break; sleep 15
  done
  run_one $G $V $t $E $A &
  echo $! > $CL/$G
  echo "띄움 $V $t 카드 $G $(date +%T)"
  sleep 6
done < $L
while [ $(jobs -rp | wc -l) -gt 0 ] && [ "$(date +%s)" -lt "$HARD" ]; do sleep 20; done
$PY exe/fem_cm7_sum.py $R > $W/cm7_table.md 2>&1
cat $R/done.txt >> $W/cm7_table.md
rclone copyto $W/cm7_table.md $MR/cm7_table.md -q
date '+%F %T' > $W/cm7.END; [ $(jobs -rp | wc -l) -gt 0 ] && echo "마감: 아직 도는 판 있음" >> $W/cm7.END
rclone copyto $W/cm7.END $MR/cm7.END -q
echo "[cm7 끝] $(date '+%F %T')"
