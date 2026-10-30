#!/bin/bash
# 절 AF (사용자 10-30 고침): 씬마다 N 을 찾지 않고 공통 설정 하나 -- 확정 플래그 + 서브스텝 간격 고정 h_s = 6.25e-5 s (N = round(frame_dt/h_s); lego 1/60 -> 267).
# (1) lego 홀드아웃 20 궤적 x 3 회 비측정 (카드 0~6 동시, 회차 1 은 F 덤프 f32) (2) 3DGS 렌더 카드 0~6 동시 (우리 + 정답), R2 확인 뒤 /tmp 큰 덤프만 지움
# (3) 단독 측정 GPU0 x 3 회 (gpu0.lock, 카드 1~6 이 빌 때까지 기다린 뒤; 덤프 없음). 출력 /tmp/dkta/lbours/fem_af_lego (R2 lbours/fem_af_lego).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; OD=/tmp/dkta/lbours/fem_af_lego; RR=r2:storage/result/anchorflow/lbours/fem_af_lego; mkdir -p $OD
DD=/tmp/dkta/af_data; [ -f $DD/0019.pt ] || DD=$W/lbpt/test
HS=6.25e-5; N=$(python -c "print(round((1/60)/$HS))")
REC="--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"
SC="--floor vel --node_h 0.028284 --frames 80"
TR=$(seq -f "%04g" 0 19)
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') h_s $HS -> N $N, 데이터 $DD"
idle() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v s="$1" '$1>=s && $2+0>10{b=1} END{exit b}'; }
# 앞 판 (N 스윕 렌더) 이 남아 있으면 끝날 때까지 (카드 0~6 이 빌 때까지)
until idle 0; do sleep 20; done
# (1) 비측정 3 회: 카드 0~6 x 일꾼 3
run1() { local G=$1 k=$2 i=0
  for r in 1 2 3; do for t in $TR; do
    if [ $((i % 21)) -eq $k ]; then
      D="--no_dump"; [ $r = 1 ] && D="--dump_f32"
      EX_ROOT=fem_af_lego bash exe/fem_ab_run.sh af_${t}_c${N}_r$r $G --data $DD/$t.pt --tag T$t $SC $D $REC --N $N < /dev/null > /dev/null 2>&1
      echo "$(date +%T) af_${t}_c${N}_r$r $(grep -a '롤아웃\]' $OD/af_${t}_c${N}_r$r/v_fem.log | cut -c1-160)"
    fi; i=$((i + 1)); done; done; }
for G in 0 1 2 3 4 5 6; do for k in 0 1 2; do run1 $G $((G * 3 + k)) > $OD/wc_${G}_$k.log 2>&1 & done; done
wait
cat $OD/wc_*.log | sort
echo "[비측정 끝] $(date +%T)"
# (2) 렌더
rend() {
  local G=$1 t=$2; local O=$OD/af_${t}_c${N}_r1 GD=$OD/gt_$t
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $t"; return; }
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $W/lbdata/test/$t --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj $GD
  [ -f $O/vj/fem_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 \
     --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $GD/gt_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $W/lbdata/test/$t --config $W/lbdata/config.json --shape lego --out $GD/gt_lego.mp4 \
     --frame_label 0 --caption " " < /dev/null > $GD/r_gt.log 2>&1
  [ -f $O/vj/fem_lego.mp4 ] && touch $O/vj/RENDER_DONE
  rclone copy $O $RR/af_${t}_c${N}_r1 -q; rclone copy $GD $RR/gt_$t -q
  if rclone check $O $RR/af_${t}_c${N}_r1 --one-way --size-only -q 2>/dev/null; then rm -rf $O/roll_fem.pt $O/h5_fem; echo "[R2 확인·/tmp 정리] $t"; else echo "[R2 확인 실패 -- /tmp 유지] $t"; fi
}
i=0
for t in $TR; do rend $((i % 7)) $t > $OD/rendc_$t.log 2>&1 & i=$((i + 1)); [ $((i % 7)) -eq 0 ] && wait; done
wait
cat $OD/rendc_*.log
echo "[렌더 끝] $(date +%T)"
# (3) 단독 측정: GPU0, 카드 1~6 이 빈 것을 회마다 확인 (남의 실행이 있으면 빌 때까지 기다린다; 시작·끝 상태는 gpu_at_start/end.txt)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OD/gpu_before_c.txt; cat $OD/gpu_before_c.txt
for r in 1 2 3; do for t in $TR; do
  TG=af_${t}_c${N}_t$r
  w=0; until idle 1; do sleep 15; w=$((w + 15)); done; [ $w -gt 0 ] && echo "[대기] $TG 카드 1~6 빌 때까지 $w s"
  EX_TIMED=1 EX_ROOT=fem_af_lego bash exe/fem_ab_run.sh $TG 0 --data $DD/$t.pt --tag T$t $SC --no_dump $REC --N $N < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]' $OD/$TG/v_fem.log | cut -c1-160) | 카드 1~6 시작 $(awk -F, '$1!=0{print $2}' $OD/$TG/gpu_at_start.txt | tr -d ' \n') 끝 $(awk -F, '$1!=0{print $2}' $OD/$TG/gpu_at_end.txt | tr -d ' \n')"
done; done
python exe/fem_af_ctable.py $OD $N > $OD/table_c.md; cat $OD/table_c.md
rclone copy $OD $RR -q
echo "AFC_DONE $(date +%T)"; touch $OD/AFC_DONE; rclone copy $OD/AFC_DONE $RR -q
