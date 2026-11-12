#!/bin/bash
# wolf H9(=H8, R2 lbours/wolfh12/h8_1) 칸 모은 영상 진입점. 사용: bash exe/go_wolffields.sh <커밋> <단계 (wf)>. 노드 / ·/tmp 안 씀.
# 덤프 /home/dkta/work/wolffields, 나머지 /dev/shm/dkta/wolffields. 단계 wf = 세 양 계산 -> 색 렌더 (정답 일반·탄성·운동, H9 일반·탄성·운동·칸 높이 비) -> R2. 합성은 wolffields_compose.py.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolffields
O=/dev/shm/dkta/wolffields; DD=/home/dkta/work/wolffields; mkdir -p $O $DD /dev/shm/dkta/tmp /dev/shm/dkta/cache
export TMPDIR=/dev/shm/dkta/tmp XDG_CACHE_HOME=/dev/shm/dkta/cache PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; ffmpeg $(command -v ffmpeg)"
GTW=/dev/shm/dkta/ag_data/gt_wolf
if [ ! -f $GTW/meta.json ] || [ ! -f $GTW/sim_0000000050.h5 ]; then
  GTW=/dev/shm/dkta/wolfcv_gt; mkdir -p $GTW
  [ -f $GTW/sim_0000000050.h5 ] || { CUDA_VISIBLE_DEVICES="" $PY -u exe/pg_h5_to_pt.py --h5_dir /home/dkta/work/topo/pg_wolf_sand --config /home/dkta/work/PhysGaussian/config/wolf_config.json --out_pt $GTW/tmp.pt --out_dir $GTW > $O/mkgt.log 2>&1; rm -f $GTW/tmp.pt; }
fi
WPT=/dev/shm/dkta/ag_data/wolf.pt
say "정답 $GTW ($(ls $GTW | wc -l) 파일) wolf.pt $(ls -la $WPT 2>&1 | cut -c1-60)"
[ -s $DD/pred.pt ] || { [ -s /home/dkta/work/wolfen/pred.pt ] && cp /home/dkta/work/wolfen/pred.pt $DD/pred.pt || rclone copyto r2:storage/result/anchorflow/lbours/wolfh12/h8_1/pred.pt $DD/pred.pt; }
ND=/home/dkta/work/cellh/c9_1
[ "$(ls $ND/nodes 2>/dev/null | wc -l)" = 50 ] || { ND=$DD/c9_1; mkdir -p $ND; rclone copy r2:storage/result/anchorflow/lbours/wolfcv/cellh/c9_1/nodes $ND/nodes -q --transfers 8; }
say "pred.pt $(du -sh $DD/pred.pt | cut -f1) 노드 $ND $(ls $ND/nodes | wc -l) 개"
$PY -u exe/wolffields_psi.py $DD/pred.pt $GTW $WPT $ND $O/psi > $O/psi.log 2>&1; say "psi rc=$?"
rclone copyto $O/psi.log $R/psi.log -q; rclone copyto $O/psi/fields.json $R/fields.json -q
grep -q WOLFFIELDS_DONE $O/psi.log || { say "psi 실패 $(tail -3 $O/psi.log | cut -c1-300)"; exit 1; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; NG=$(nvidia-smi -L | wc -l); G=""
while [ -z "$G" ]; do
  for g in $(seq 0 $((NG - 1))); do f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $$ > $f ) 2>/dev/null && { G=$g; break; }
  done; [ -z "$G" ] && sleep 30
done
say "카드 $G"
mkdir -p $O/h5p $O/vid
$PY -u exe/lb_roll_to_h5.py --dump $DD/pred.pt --traj $GTW --out $O/h5p > $O/h5.log 2>&1; say "h5 rc=$?"
export CUDA_VISIBLE_DEVICES=$G
RC="exe/lb_render.py --config $GTW/config.json --shape wolf"
$PY -u $RC --sim $GTW --frames 50 --out $O/vid/gt_plain.mp4 < /dev/null > $O/vid/r0.log 2>&1; say "gt_plain rc=$?"
for q in en ke; do
  $PY -u $RC --sim $GTW --frames 50 --colors $O/psi/col_gt_$q.npy --out $O/vid/gt_$q.mp4 < /dev/null > $O/vid/rg_$q.log 2>&1; say "gt_$q rc=$?"
done
for q in en ke r; do
  $PY -u $RC --sim $O/h5p --frames 50 --colors $O/psi/col_pred_$q.npy --out $O/vid/h9_$q.mp4 < /dev/null > $O/vid/rh_$q.log 2>&1; say "h9_$q rc=$?"
done
$PY -u $RC --sim $O/h5p --frames 50 --out $O/vid/h9_plain.mp4 < /dev/null > $O/vid/rh_p.log 2>&1; say "h9_plain rc=$?"
rm -f $CL/$G
rclone copy $O/vid $R/vid -q; rclone copy $O/psi $R/psi -q --include "*.json"
if command -v ffmpeg >/dev/null; then
  $PY -u exe/wolffields_compose.py $O/psi/fields.json $O/vid $O/fem_wolf_h9_fields.mp4 $O/tmpc > $O/compose.log 2>&1; say "compose rc=$?"
  $PY -u exe/wolffields_html.py $O/psi $O/vid $O/wolf_energy_scatter.html $O/tmph > $O/html.log 2>&1; say "html rc=$?"
  rclone copy $O/fem_wolf_h9_fields.mp4 $R -q; rclone copy $O/wolf_energy_scatter.html $R -q; rclone copy $O/compose.log $R -q; rclone copy $O/html.log $R -q
else say "ffmpeg 없음 -- 합성은 따로"; fi
say "끝"; echo done > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
