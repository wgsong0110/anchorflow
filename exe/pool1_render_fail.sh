#!/bin/bash
# pool1 풀 이력 중 렌더에 실패한 것(.fail)을 다시 렌더 (사용자 10-25). 실패 원인: 이어 돌리기로 이어받은 상태(trace_from>0)의
#   이력이 한 프레임뿐이면 lb_roll_to_h5 --first_from_dump 가 빈 텐서에서 죽었다 (이 커밋에서 고침). 원본 .pt·.fail 표시는 그대로 두고 mp4 만 만든다.
#   bash exe/pool1_render_fail.sh <GPU>     (잡 안에서만: AF_JOB=.. AF_NODE=.. AF_DETACH=1 bash exe/job_run.sh $W/pool1_render_fail_boot.sh <로그>)
export PYTHONUTF8=1 OMP_NUM_THREADS=2
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}; cd "$(dirname "$0")/.." || exit 1
exec 9>/tmp/pool1_render_fail.lock; flock -n 9 || { echo "[중단] 다른 재렌더가 돌고 있다"; exit 0; }
for m in gnn gunet; do TD=$W/lbours/pool1_$m/pool/pool_trace; V=$TD/vid; R=r2:storage/result/anchorflow/lbours/pool1_$m/pool/pool_trace/vid
  for fl in $(ls $V/*.fail 2>/dev/null); do n=$(basename $fl .fail); f=$TD/$n.pt
    [ -f $V/$n.mp4 ] && continue; [ -f $f ] || continue
    H=/tmp/pool1_h5r_${m}_$n; [ -e $H ] && mv $H ${H}_prev_$(date +%s)
    FF=$(python -c "import torch,json;m=torch.load('$f',map_location='cpu',weights_only=False);print('--first_from_dump' if m['meta']['trace_from']>0 else '')")
    python -u exe/lb_roll_to_h5.py --dump $f --traj $W/lbdata/test_low/L000 --out $H $FF > $V/$n.h5r.log 2>&1 \
      && CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $H --config $W/lbdata/config.json --shape lego --out $V/$n.mp4 > $V/$n.rr.log 2>&1 < /dev/null
    rm -rf $H      # 변환 h5 만 (원본 .pt 그대로)
    [ -f $V/$n.mp4 ] && { echo "[재렌더] $m $n"; rclone copy $V $R -q --include "$n.*"; } || echo "[재렌더 실패] $m $n"
  done
done
echo POOL1_RENDER_FAIL_DONE
