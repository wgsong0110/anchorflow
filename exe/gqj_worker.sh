#!/bin/bash
# 절 CM194 (gq_justify) 일꾼: R2 대기열 lbours/gq_justify/queue/<GQ_Q>.txt (한 줄 = 장면:판:조건:모드:반복) 를 여러 인스턴스가 함께 (R2 LOCK 으로 차지).
#   판마다: exe/gqj_cmd.py 명령 -> 실행 -> exe/gqj_post.py 요약 (gq.json); 모드 dump 면 물리 (ffx_phys) -> 렌더 (lb_roll_to_h5 + lb_render, 이방성) -> 시각 (ffx_vis).
#   결과 R2 lbours/gq_justify/runs/<장면>_<판>_<조건>_<모드>_r<반복>/ (+ DONE). 시간 모드 (time) 는 카드에 다른 계산 프로세스가 없을 때만 시작 (이 일꾼들의 다른 판도 끝날 때까지, flock 배타).
set -u
H=$1; GPU=${GQ_GPU:-0}; WID=${WID:-0}
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=4 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_gq XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
RG=r2:storage/result/anchorflow/lbours/gq_justify; RP=r2:storage/result/anchorflow/lbours/final_fix/phys; Q=$RG/queue/${GQ_Q:?}.txt
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; X=/workspace/gq; mkdir -p $X /dev/shm/dkta/tmp $CUPY_CACHE_DIR /root/gq
export TMPDIR=/dev/shm/dkta/tmp
PY=/home/dkta/.conda/envs/af/bin/python; CL=/root/gq/card$GPU.lock
say() { echo "[gq $(hostname | cut -c1-8) w$WID $(date '+%m-%d %T')] $*"; }
scene() { case $1 in
  L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel;;
  pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none;;
  bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300; FLOOR=none;;
  wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel;; esac; }
others() { nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | grep -c .; }
foreign() { local n=0; for p in $(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "/workspace/gq/" || n=$((n+1)); done; echo $n; }   # 다른 에이전트 계산 (겹치지 않게)
clean() { rm -rf "${X:?}/${1:?}" "${X:?}/${1:?}.claim"; }
idle=0
while :; do
  rclone copyto $Q /root/gq/q_$WID.txt -q 2>/dev/null
  got=0; left=0
  while IFS=: read -r s v c m r; do
    [ -z "${r:-}" ] && continue
    case ",${GQ_SC:-L000,pillow2sofa,bread,wolf}," in *",$s,"*) ;; *) continue;; esac
    [ "$m" = time ] && [ "${GQ_TIME:-0}" != 1 ] && continue                             # 시간 판은 A6000 단독 일꾼만
    [ "$m" != time ] && [ "${GQ_TONLY:-0}" = 1 ] && continue
    n=${s}_${v}_${c}_${m}_r$r
    rclone lsf $RG/runs/$n/DONE 2>/dev/null | grep -q . && continue
    left=1
    rclone lsf $RG/runs/$n/LOCK 2>/dev/null | grep -q . && continue
    mkdir $X/$n.claim 2>/dev/null || continue
    echo "$(hostname) w$WID $(date)" | rclone rcat $RG/runs/$n/LOCK -q
    got=1; D=$X/$n; clean $n; mkdir -p $D $X/$n.claim; scene $s
    $PY exe/gqj_cmd.py $s $v $c $m $D > $D/cmd.txt 2> $D/cmd.err || { say "$n 명령 생성 실패 $(tail -1 $D/cmd.err)"; rclone copy $D $RG/runs/$n -q; break; }
    echo "$H" > $D/commit.txt
    t0=$(date +%s)
    if [ "$m" = time ]; then
      exec 8>$CL; flock -x 8
      while [ "$(others)" != 0 ]; do sleep 30; done
      say "$n 시작 (시간, 카드 $GPU 단독)"
      CUDA_VISIBLE_DEVICES=$GPU timeout ${GQ_TMO:-5400} bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/log.txt 2>&1; rc=$?
      echo "after $(others)" > $D/card_after.txt
      flock -u 8; exec 8>&-
    else
      exec 8>$CL; flock -s 8
      while [ "$(foreign)" != 0 ]; do sleep 30; done
      say "$n 시작 (카드 $GPU, 다른 프로세스 $(others))"
      CUDA_VISIBLE_DEVICES=$GPU timeout ${GQ_TMO:-5400} bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/log.txt 2>&1; rc=$?
      flock -u 8; exec 8>&-
    fi
    $PY exe/gqj_post.py $D $s $v $c $m $r $rc > $D/post.log 2>&1
    say "$n rc=$rc $(( $(date +%s) - t0 )) 초 $(tail -1 $D/post.log | cut -c1-200)"
    if [ "$m" = dump ] && [ -f $D/pred.pt ]; then
      ( exec 8>$CL; flock -s 8
        GV=""; [ $c = v2 ] && GV="(충격 x2: 에너지·자유낙하 지표는 원래 중력 기준)"
        $PY -u exe/ffx_phys.py --method ours --scene $s --N "$v.$c" --data $DATA --pred $D/pred.pt --floor $FLOOR --frames $FR --out $D/phys.json --note "gq_justify $v $c $GV" > $D/phys.log 2>&1
        $PY -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $GTD --out $D/h5 > $D/roll_h5.log 2>&1
        CUDA_VISIBLE_DEVICES=$GPU $PY -u exe/lb_render.py --sim $D/h5 --config $CF --shape $SHAPE --out $D/$n.mp4 --png $D/png --frames $FR --aniso_json $D/aniso.json < /dev/null > $D/render.log 2>&1
        REF=""; if [ $c = base ]; then [ -d /root/ref/$s ] || { mkdir -p /root/ref/$s.tmp$WID && rclone copy $RP/mpm_$s/png /root/ref/$s.tmp$WID -q --transfers 16 && { mv /root/ref/$s.tmp$WID /root/ref/$s 2>/dev/null || true; }; }; REF="--ref /root/ref/$s"; fi
        CUDA_VISIBLE_DEVICES=$GPU $PY -u exe/ffx_vis.py --png $D/png $REF --out $D/vis.json > $D/vis.log 2>&1
        $PY exe/gqj_post.py $D $s $v $c $m $r $rc > $D/post.log 2>&1
        rclone copy $D $RG/runs/$n -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt" --exclude "png/**" --exclude "sp/**"
        rclone copyto $D/pred.pt $RG/runs/$n/pred.pt -q --s3-chunk-size 64M --s3-upload-concurrency 8
        [ $c = base ] && (cd $D && tar -cf png.tar png && rclone copyto png.tar $RG/runs/$n/png.tar -q)
        echo "$(date -u)" | rclone rcat $RG/runs/$n/DONE; say "$n 지표 끝 $(tail -1 $D/post.log | cut -c1-120)"; clean $n ) >> $X/bg$WID.log 2>&1 &
    else
      rclone copy $D $RG/runs/$n -q --transfers 8 --exclude "sp/**"
      echo "$(date -u)" | rclone rcat $RG/runs/$n/DONE; clean $n
    fi
    break
  done < /root/gq/q_$WID.txt
  if [ $left = 0 ]; then idle=$((idle + 60)); [ $idle -ge ${GQ_IDLE:-1800} ] && break; sleep 60; continue; fi
  idle=0; [ $got = 0 ] && sleep 60
done
wait
say "일꾼 끝"
