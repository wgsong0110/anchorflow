#!/bin/bash
# 멱등 병렬 드라이버. 몇 번 다시 실행해도 **미완료 작업만** 다시 띄운다.
#
# 규칙 (2026-09-30 사고 재발 방지):
#   - 모든 실행에 `< /dev/null` -- stdin 을 물고 죽어 tty 를 망가뜨리는 일을 막는다
#   - 작업마다 **독립 tmux 세션**. 대화형 판에는 이 스크립트 한 줄만 들어간다
#   - 작업마다 완료 표식(`$DONE/<작업>.ok`). 있으면 건너뛴다
#   - 이미 그 작업의 세션이 살아 있으면 건너뛴다 (중복 실행 금지)
#
# 사용: bash exe/cluster_driver.sh [phase]      phase: teacher | student | bench
set -u
W=${AF_WORK:-/home/dkta/work}
R=$W/anchorflow
DONE=$W/done
LOG=$W/dlog
mkdir -p $DONE $LOG
PHASE=${1:-teacher}
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$R/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache

have () { [ -f "$DONE/$1.ok" ]; }
alive () { tmux has-session -t "$1" 2>/dev/null; }

# 작업 하나를 띄운다: job <이름> <GPU> <명령...>
job () {
  local NAME=$1 GPU=$2; shift 2
  if have "$NAME"; then echo "  [건너뜀] $NAME (완료)"; return 0; fi
  if alive "$NAME"; then echo "  [진행중] $NAME"; return 0; fi
  tmux new-session -d -s "$NAME" \
    "export PATH=/home/dkta/.conda/envs/af/bin:\$PATH \
       AF_WORK=$W PYTHONPATH=$R/lib PYTHONIOENCODING=utf-8 \
       MPLCONFIGDIR=/home/dkta/.mplcache CUDA_VISIBLE_DEVICES=$GPU; \
     { $* ; } < /dev/null >> $LOG/$NAME.log 2>&1 \
       && touch $DONE/$NAME.ok || echo FAILED >> $LOG/$NAME.log"
  echo "  [띄움] $NAME -> GPU $GPU"
}

case $PHASE in
teacher)
  echo "== 단계: i-PG 교사 생성 (하드 Dirichlet)"
  # 패치는 멱등이다. 표식으로 한 번만 돌린다.
  if ! have patch; then
    python $R/exe/patch_ipg_handle.py --ipg $W/i-physgaussian < /dev/null \
      > $LOG/patch.log 2>&1 \
      && python $R/exe/patch_ipg_fillcache.py --ipg $W/i-physgaussian < /dev/null \
      >> $LOG/patch.log 2>&1 \
      && touch $DONE/patch.ok || { echo "  [실패] 패치"; tail -5 $LOG/patch.log; exit 1; }
    echo "  [완료] 패치"; grep -a "패치" $LOG/patch.log | tail -3
  else
    echo "  [건너뜀] 패치 (완료)"
  fi
  for NG in 100 58; do
    NAME=ipg_$NG
    # 채움 캐시는 두 해상도가 **공유**한다 (particle_filling.n_grid 를 고정했으므로
    # 입자 집합이 같다). 하나가 만들면 다른 쪽은 즉시 읽는다.
    job $NAME $([ $NG = 100 ] && echo 0 || echo 1) \
      "cd $W/i-physgaussian && AF_PGFILL_NPY=$W/ipg/fill_mic_t.npy \
       AF_H_SCEN=$W/ipg/scen_s400706.npz AF_H_R=0.15 \
       python -u gs_simulation.py --model_path $W/pgmodel/mic_whitebg-trained \
         --config $W/ipg/cfg_ng$NG.json --output_path $W/ipg/out_$NG \
         --output_h5 --implicit --solver newton_gmres"
  done
  ;;
bench)
  echo "== 단계: 속도 측정 (단독 실행 -- 다른 작업이 없을 때만)"
  if tmux ls 2>/dev/null | grep -qvE "^(bench|k18)" ; then
    echo "  [보류] 다른 세션이 돌고 있다. 경합 상태의 수치는 버리게 된다"
    tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' '; echo
    exit 0
  fi
  job bench 0 "bash $R/exe/bench_arch_speed.sh"
  ;;
*)
  echo "알 수 없는 단계: $PHASE (teacher | student | bench)"; exit 1;;
esac
echo "== 상태"
for f in $DONE/*.ok; do [ -e "$f" ] && echo "  완료: $(basename ${f%.ok})"; done
tmux ls 2>/dev/null | cut -d: -f1 | sed 's/^/  세션: /'
