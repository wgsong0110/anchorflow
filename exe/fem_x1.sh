#!/bin/bash
# 절 X: (1) 연기 시험 (3 프레임 + 진단 덤프 + 적분점 검산) -> (2) GPU 0~6 스윕 (fem_x_jobs.py, 비측정, 카드마다 비었을 때만 -- 같은 잡의 절 W 와 카드 잠금 공유)
# -> (3) 발산 진단 (fem_dv_stats.py, 절 U 그대로)·적분점 검산 (fem_x_qpt.py)·표·ficus 주기 -> (4) 모든 반복 완주한 판이 있으면 단독 재측정 3 회 + F 덤프 최종·3DGS 렌더
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
XL=$W/lbours/fem_x_lego; XF=$W/lbours/fem_x_ficus
mkdir -p $XL $XF $W/gpulock_w; rm -f $XL/GPU?_DONE_*
sync_r2() { rclone copy $XL r2:storage/result/anchorflow/lbours/fem_x_lego -q; rclone copy $XF r2:storage/result/anchorflow/lbours/fem_x_ficus -q; }
( while true; do sleep 180; sync_r2; done ) & SY=$!
trap 'kill $SY 2>/dev/null' EXIT
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
smoke() {  # smoke <G> <루트> <태그> <인자...>
  G=$1; RT=$2; TG=$3; shift 3
  until mkdir $W/gpulock_w/$G 2>/dev/null; do sleep 20; done
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 30; done
  EX_ROOT=$RT EX_NOWAIT=1 bash exe/fem_ex_run.sh $TG $G "$@" > /dev/null 2>&1
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_x_qpt.py $W/lbours/$RT/$TG > $W/lbours/$RT/$TG/qpt.log 2>&1
  rmdir $W/gpulock_w/$G
}
smoke 0 fem_x_lego x_smoke_l --mode lag --opt --mass bary --no_dump --floor vel --rstat --lagdiag --lagdump 1 --frames 3 --N 192 &
smoke 1 fem_x_ficus x_smoke_f --mode lag --opt --mass bary --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --lagdiag --lagdump 1 --N 640 &
smoke 2 fem_x_lego x_smoke_lg --mode lag --opt --mass bary --no_dump --floor vel --rstat --frames 3 --N 192 &
wait
bad=0
for d in $XL/x_smoke_l $XF/x_smoke_f $XL/x_smoke_lg; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d)"; else echo "[연기 실패] $(basename $d)"; tail -20 $d/v_fem.log; bad=1; fi
  grep -a "명시적 FEM\]\|롤아웃\]\|\[발산\]\|프레임 [123]/" $d/v_fem.log | cut -c1-600
  [ -f $d/qpt.log ] && cat $d/qpt.log | cut -c1-600
done
grep -q "상태" $XL/x_smoke_l/qpt.log || bad=1
sync_r2
[ $bad -eq 0 ] || { echo X1_SMOKE_FAIL; exit 1; }
echo X1_SMOKE_OK $(date +%T)
for G in 0 1 2 3 4 5 6; do bash exe/fem_w_gpu.sh $G exe/fem_x_jobs.py $XL < /dev/null > $W/fem_x1_g$G.out 2>&1 & done
wait
sleep 10
echo X1_SWEEP_DONE $(date +%T)
# 진단: 절 U 그대로 (폭주 vs 정상), 적분점·질량 검산
for d in $XL/x_l_*_dg $XF/x_f_*_dg; do
  [ -f $d/lagdump.pt ] || { echo "[덤프 없음] $d"; continue; }
  CUDA_VISIBLE_DEVICES=0 python -u exe/fem_dv_stats.py $d < /dev/null > $d/dv_stats.log 2>&1; echo "== $(basename $d) dv rc $?"; tail -14 $d/dv_stats.log | cut -c1-300
  CUDA_VISIBLE_DEVICES=0 python -u exe/fem_x_qpt.py $d < /dev/null > $d/qpt.log 2>&1; echo "   qpt rc $?"; cut -c1-400 $d/qpt.log
done
python exe/fem_p_period.py --data $FD --pred $XF/x_f_*/pred.pt > $XF/period1.txt 2>&1
python exe/fem_r_table.py $XL/x_l_* > $XL/table1.md
python exe/fem_r_table.py $XF/x_f_* > $XF/table1.md
cat $XL/table1.md $XF/table1.md | cut -c1-260
sync_r2
echo X1_DONE $(date +%T)
# (4) 단독 측정·최종 렌더 (모든 반복 완주한 판만)
LB="--mode lag --opt --mass bary --floor vel --rstat"
FB="--mode lag --opt --mass bary --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
mon() {
  O=$1
  while [ ! -f $O/summary.json ] && [ ! -f $O/EX_FAIL ]; do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 != 0 && $2 > 500')
    [ -n "$u" ] && echo "$(date +%T) $u" >> $O/CONTAM
    sleep 5
  done
}
timed() {
  RT=$1; TG=$2; shift 2
  for k in 1 2 3; do
    O=$W/lbours/$RT/$TG
    until mkdir $W/gpulock_w/0 2>/dev/null; do sleep 20; done
    ( until [ -f $O/gpu_at_start.txt ]; do sleep 1; done; mon $O ) & MP=$!
    EX_ROOT=$RT bash exe/fem_ex_run.sh $TG 0 "$@"
    sleep 6; kill $MP 2>/dev/null; rmdir $W/gpulock_w/0
    if [ -f $O/CONTAM ]; then echo "[오염] $TG 시도 $k: $(head -2 $O/CONTAM)"; mv $O ${O}_contam$k; else return 0; fi
  done
}
fin() {
  S=$1; T=$2; shift 2
  for a in 1 2 3; do
    O=$W/lbours/fem_x_$S/${T}_F$a
    until mkdir $W/gpulock_w/0 2>/dev/null; do sleep 20; done
    ( until [ -f $O/gpu_at_start.txt ]; do sleep 1; done; mon $O ) & MP=$!
    SHAPE=$S EX_ROOT=fem_x_$S bash exe/fem_ex_final.sh ${T}_F$a 0 "$@"
    kill $MP 2>/dev/null; rmdir $W/gpulock_w/0
    [ -f $O/vj/RENDER_DONE ] && { echo "[최종] $S $T: 시도 $a 완주·렌더 $( [ -f $O/CONTAM ] && echo '(측정 중 다른 카드 사용)')"; return; }
    echo "[최종] $S $T: 시도 $a 발산/실패"
  done
}
for S in lego ficus; do
  P=$([ $S = lego ] && echo x_l_ || echo x_f_)
  read -r V N ARGS <<< "$(python exe/fem_x_pick.py $W/lbours/fem_x_$S $P)"
  echo "[선택] $S: $V N$N ($ARGS)"
  [ "$V" = NONE ] && { echo "[선택] $S: 모든 반복 완주한 설정 없음 -- 측정·영상 없음"; continue; }
  BASE=$([ $S = lego ] && echo "$LB" || echo "$FB")
  for r in 1 2 3; do timed fem_x_$S m_${P#x_}${V}_n${N}_r$r $BASE --no_dump $ARGS --N $N; done
  fin $S f_${P#x_}${V}_n$N $BASE --dump_f32 $ARGS --N $N
done
ls -d $XF/f_f_* $XF/m_f_* 2>/dev/null && python exe/fem_p_period.py --data $FD --pred $XF/f_f_*/pred.pt $XF/m_f_*/pred.pt > $XF/period2.txt 2>&1
ls -d $XL/f_l_* $XL/m_l_* 2>/dev/null && python exe/fem_r_table.py $XL/f_l_* $XL/m_l_* > $XL/table2.md
ls -d $XF/f_f_* $XF/m_f_* 2>/dev/null && python exe/fem_r_table.py $XF/f_f_* $XF/m_f_* > $XF/table2.md
sync_r2
echo X2_DONE $(date +%T)
