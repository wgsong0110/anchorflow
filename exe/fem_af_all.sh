#!/bin/bash
# 절 AF 전체 (잡 안, AF_DETACH 한 번): (0) 데이터 /tmp 사본 (1) N 스윕 비측정 카드 0~6 x 일꾼 3 (2) 단독 측정 GPU0 (다른 카드 빈 상태, gpu0.lock) -- 1 회차에 F 덤프
# (3) 3DGS 렌더 카드 0~6 동시 (우리 + 정답) (4) R2 확인 뒤 /tmp 큰 덤프만 지움. 출력 /tmp/dkta/lbours/fem_af_lego (R2 lbours/fem_af_lego).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; OD=/tmp/dkta/lbours/fem_af_lego; RR=r2:storage/result/anchorflow/lbours/fem_af_lego; mkdir -p $OD
DD=/tmp/dkta/af_data
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"; df -h /tmp | tail -1
# (0) 정답 .pt 를 노드 /tmp 로 (읽기만, NFS 반복 읽기 줄임) -- 자리가 모자라면 NFS 그대로
FREE=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if [ "$FREE" -gt 80 ]; then mkdir -p $DD; for i in $(seq -w 0 19); do t=00$i; [ -f $DD/$t.pt ] || cp $W/lbpt/test/$t.pt $DD/; done; export AF_DATA=$DD; else export AF_DATA=$W/lbpt/test; fi
echo "[데이터] $AF_DATA $(date +%T)"
# 연기 시험 (3 프레임)
if [ ! -f $OD/SWEEP_DONE ]; then
  EX_ROOT=fem_af_lego bash exe/fem_ab_run.sh smoke 0 --data $AF_DATA/0004.pt --tag T0004 --floor vel --node_h 0.028284 --frames 3 --no_dump \
     --mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5 --N 256 < /dev/null > /dev/null 2>&1
  head -3 $OD/smoke/v_fem.log | cut -c1-400; grep -a "롤아웃\]" $OD/smoke/v_fem.log
  [ -f $OD/smoke/EX_DONE ] || { echo "[연기 시험 실패]"; tail -5 $OD/smoke/v_fem.log; exit 1; }
  # (1) 스윕
  for G in 0 1 2 3 4 5 6; do for k in 0 1 2; do
    bash exe/fem_af_gpu.sh $G $(( G * 3 + k )) 21 exe/fem_af_jobs.py < /dev/null > $OD/w1_${G}_$k.log 2>&1 &
  done; done
  wait
  touch $OD/SWEEP_DONE
fi
python exe/fem_af_pick.py $OD $OD/table1.md $OD/timed_list.txt > /dev/null; cat $OD/table1.md
rclone copy $OD $RR -q --exclude "/af_*/**"
echo "[스윕 끝] $(date +%T)"
# (2) 단독 측정: GPU0, 카드 1~6 은 비어 있어야 (스윕 끝) -- 회차 1 에 F 덤프 (저장은 측정 밖)
REC="--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OD/gpu_before_timed.txt; cat $OD/gpu_before_timed.txt
for r in 1 2 3; do
  while read -r t N k; do
    TG=af_${t}_n${N}_t$r; D=""; [ $r = 1 ] && [ $k = star ] && D="--dump_f32" || D="--no_dump"
    EX_TIMED=1 EX_ROOT=fem_af_lego bash exe/fem_ab_run.sh $TG 0 --data $AF_DATA/$t.pt --tag T$t --floor vel --node_h 0.028284 --frames 80 $D $REC --N $N < /dev/null > /dev/null 2>&1
    echo "$(date +%T) $TG $(grep -a '롤아웃\]' $OD/$TG/v_fem.log | cut -c1-170) | 다른 카드 $(awk -F, '$1!=0{print $2}' $OD/$TG/gpu_at_start.txt | tr -d ' \n')"
  done < $OD/timed_list.txt
done
python exe/fem_af_table.py $OD > $OD/table2t.md; cat $OD/table2t.md
rclone copy $OD $RR -q --exclude "roll_fem.pt" --exclude "h5_fem/**"
echo "[측정 끝] $(date +%T)"
# (3) 렌더: 궤적마다 (덤프 -> h5 -> 3DGS 알파블렌딩) + 정답 렌더, 카드 0~6 에 나눠 동시
rend() {
  local G=$1 t=$2 N=$3; local O=$OD/af_${t}_n${N}_t1 GD=$OD/gt_$t
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $t"; return; }
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $W/lbdata/test/$t --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj $GD
  [ -f $O/vj/fem_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 \
     --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $GD/gt_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $W/lbdata/test/$t --config $W/lbdata/config.json --shape lego --out $GD/gt_lego.mp4 \
     --frame_label 0 --caption " " < /dev/null > $GD/r_gt.log 2>&1
  [ -f $O/vj/fem_lego.mp4 ] && touch $O/vj/RENDER_DONE
  rclone copy $O $RR/af_${t}_n${N}_t1 -q; rclone copy $GD $RR/gt_$t -q
  # R2 에 같은 크기로 올라갔는지 확인한 뒤 /tmp 쪽 큰 덤프만 지운다 (R2 원본 유지)
  if rclone check $O $RR/af_${t}_n${N}_t1 --one-way --size-only -q 2>/dev/null; then rm -rf $O/roll_fem.pt $O/h5_fem; echo "[R2 확인·/tmp 정리] $t"; else echo "[R2 확인 실패 -- /tmp 유지] $t"; fi
}
i=0
while read -r t N k; do
  [ $k = star ] || continue
  G=$((i % 7)); rend $G $t $N > $OD/rend_$t.log 2>&1 &
  i=$((i + 1)); [ $((i % 7)) -eq 0 ] && wait
done < $OD/timed_list.txt
wait
cat $OD/rend_*.log
rclone copy $OD $RR -q
echo "AF_DONE $(date +%T)"; touch $OD/AF_DONE; rclone copy $OD/AF_DONE $RR -q
