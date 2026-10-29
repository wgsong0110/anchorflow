#!/bin/bash
# 절 W 2 차: 1 차 스윕 (fem_w1.sh) 끝난 뒤 -- 형상마다 모든 반복 완주한 가장 작은 N 의 판 (exe/fem_w_pick.py) 을
#   (a) 단독 재측정 3 회 (--no_dump; GPU0, 7 장 500 MiB 아래 대기 -- 실행 중 다른 카드에 작업이 올라오면 CONTAM 표식, 그 회는 다시)
#   (b) F 덤프 켠 최종 실행 + 3DGS 렌더 (fem_ex_final.sh, 같은 단독 조건; 발산하면 3 번까지)  -> 표·ficus 주기·보존
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
until grep -q "W1_DONE\|W1_SMOKE_FAIL" $W/fem_w1.out 2>/dev/null; do sleep 60; done
grep -q W1_DONE $W/fem_w1.out || { echo "[1 차 실패]"; exit 1; }
echo "[1 차 끝 확인] $(date +%T)"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LB="--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --floor vel --rstat --bsdiag -1"
FB="--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1"
mon() {  # mon <출력 디렉토리> -- 측정 실행 동안 GPU 1~6 에 500 MiB 넘는 작업이 보이면 CONTAM 에 기록
  O=$1
  while [ ! -f $O/summary.json ] && [ ! -f $O/EX_FAIL ]; do
    u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 != 0 && $2 > 500')
    [ -n "$u" ] && echo "$(date +%T) $u" >> $O/CONTAM
    sleep 5
  done
}
timed() {  # timed <루트> <태그> <인자...> -- 단독 측정, 오염되면 다시 (3 번까지; 오염된 실행 디렉토리는 <태그>_contam<k> 로 옮겨 둔다)
  RT=$1; TG=$2; shift 2
  for k in 1 2 3; do
    O=$W/lbours/$RT/$TG
    ( until [ -f $O/gpu_at_start.txt ]; do sleep 1; done; mon $O ) & MP=$!
    EX_ROOT=$RT bash exe/fem_ex_run.sh $TG 0 "$@"
    sleep 6; kill $MP 2>/dev/null
    if [ -f $O/CONTAM ]; then echo "[오염] $TG 시도 $k: $(head -2 $O/CONTAM)"; mv $O ${O}_contam$k; else return 0; fi
  done
}
fin() {  # fin <형상> <태그> <인자...>
  S=$1; T=$2; shift 2
  for a in 1 2 3; do
    O=$W/lbours/fem_w_$S/${T}_F$a
    ( until [ -f $O/gpu_at_start.txt ]; do sleep 1; done; mon $O ) & MP=$!
    SHAPE=$S EX_ROOT=fem_w_$S bash exe/fem_ex_final.sh ${T}_F$a 0 "$@"
    kill $MP 2>/dev/null
    [ -f $O/vj/RENDER_DONE ] && { echo "[최종] $S $T: 시도 $a 에서 완주·렌더 $( [ -f $O/CONTAM ] && echo '(측정 중 다른 카드 사용 -- ms 는 재측정 값을 쓴다)')"; return; }
    echo "[최종] $S $T: 시도 $a 발산/실패"
  done
}
for S in lego ficus; do
  P=$([ $S = lego ] && echo w_l_ || echo w_f_)
  read -r V N ARGS <<< "$(python exe/fem_w_pick.py $W/lbours/fem_w_$S $P)"
  echo "[선택] $S: $V N$N ($ARGS)"
  [ "$V" = NONE ] && { echo "[선택] $S: 모든 반복 완주한 설정 없음 -- 측정·영상 없음"; continue; }
  BASE=$([ $S = lego ] && echo "$LB" || echo "$FB")
  for r in 1 2 3; do timed fem_w_$S m_${P#w_}${V}_n${N}_r$r $BASE --no_dump $ARGS --N $N; done
  fin $S f_${P#w_}${V}_n$N $BASE --dump_f32 $ARGS --N $N
done
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_w_ficus/f_f_*/pred.pt $W/lbours/fem_w_ficus/m_f_*/pred.pt > $W/lbours/fem_w_ficus/period2.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_w_lego/f_l_* $W/lbours/fem_w_lego/m_l_* > $W/lbours/fem_w_lego/table2.md
python exe/fem_r_table.py $W/lbours/fem_w_ficus/f_f_* $W/lbours/fem_w_ficus/m_f_* > $W/lbours/fem_w_ficus/table2.md
python exe/fem_s_cons.py $W/lbours/fem_w_lego/f_l_* $W/lbours/fem_w_lego/m_l_* $W/lbours/fem_w_ficus/f_f_* $W/lbours/fem_w_ficus/m_f_* > $W/lbours/fem_w_lego/cons2.txt 2>&1
rclone copy $W/lbours/fem_w_lego r2:storage/result/anchorflow/lbours/fem_w_lego -q
rclone copy $W/lbours/fem_w_ficus r2:storage/result/anchorflow/lbours/fem_w_ficus -q
echo W2_DONE $(date +%T)
