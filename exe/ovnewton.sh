#!/bin/bash
# 출력만 최적화 + 뉴턴 걸음 프레임당 N(기본 40) 회 고정 (사용자 10-23): 망 nt(pf3obj nt) 발산이 뉴턴 걸음 자체 탓인지 확인.
#   망 nt 는 매 프레임 U0 에서 감쇠 뉴턴 걸음 p 로 목표 U0-p 를 만들어 Adam 50 회 따라가기를 40 번 (--ov_roll 2000, AF_OV_NEWTON_EVERY=50).
#   여기서는 같은 p(_newton_target)를 노드 출력 변수에 바로: U <- U - p 를 정확히 N 번 (선탐색·안전장치 없음), 시작 = --ov_init predict.
#   명령줄 = pf3obj/nt/<T>/cmd.txt 마지막 줄(망 nt 실행) 그대로에서 --ov_vars net -> out, 출력 경로, GPU, AF_OV_NEWTON_STEPS 만 바꾼다.
#   위치·변수·p 가 nan/inf 가 되면 그 프레임에서 멈추고 기록 (_ovnet_fit).
# OVN_STAGE=test: L000 2 프레임 (출력 $U/test_<커밋>/L000)   OVN_STAGE=full: L000·0004 80 프레임 GPU 두 장 동시 -> 3DGS 렌더
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <이 스크립트를 부르는 부트 스크립트> <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD); STAGE=${OVN_STAGE:?OVN_STAGE=test 또는 full}
NSTEPS=${OVN_STEPS:-40}
U=$W/lbours/ovnewton; R=r2:storage/result/anchorflow/lbours/ovnewton; NT=$W/lbours/pf3obj/nt
mkdir -p $U
echo "[시작] $(date) 단계 $STAGE 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} 걸음 $NSTEPS"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
tokdiff(){ comm -$1 <(tr ' ' '\n' <<<"$2" | sort) <(tr ' ' '\n' <<<"$3" | sort) | paste -sd' '; }
# mkcmd <T> <출력> <gpu> <프레임수>
mkcmd(){ local T=$1 O=$2 g=$3 n=$4
  local NC; NC=$(tail -1 $NT/$T/cmd.txt | sed 's/^\[[^]]*\] //')
  sed -e "s#AF_OVNET_R2=[^ ]*#AF_OVNET_R2=$R/${O#$U/}#" -e "s#$NT/$T#$O#g" -e "s#CUDA_VISIBLE_DEVICES=[0-9]*#CUDA_VISIBLE_DEVICES=$g#" \
      -e "s#--ov_vars net#--ov_vars out#" -e "s#--eval_len 80#--eval_len $n#" -e "s#^env #env AF_OV_NEWTON_STEPS=$NSTEPS #" <<<"$NC"; }
pick2(){ # 메모리 적은 카드 둘 (프로세스 없는 카드 먼저)
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F', *' -v b="$busy" '{p=(index(b,$2)>0)?1:0; print p, $3, $1}' \
    | sort -k1,1n -k2,2n | head -2 | awk '{print $3}' | paste -sd' '; }
run1(){ local T=$1 O=$2 g=$3 n=$4; mkdir -p $O
  local CMD; CMD=$(mkcmd $T $O $g $n)
  local NC; NC=$(tail -1 $NT/$T/cmd.txt | sed 's/^\[[^]]*\] //')
  { echo "[실행 명령 $(date '+%F %T') 커밋 $C] $CMD"
    echo "[망 nt 명령 대비] 빠짐: '$(tokdiff 23 "$NC" "$CMD")' | 더함: '$(tokdiff 13 "$NC" "$CMD")'"; } | tee -a $O/cmd.txt
  $CMD >> $O/run_all.log 2>&1 < /dev/null
  rclone copy $O $R/${O#$U/} -q; }
render(){ local T=$1 O=$2 g=$3
  [ -f $O/roll_net.pt ] || { echo "[오류] $T 덤프 없음"; return 1; }
  [ -f $O/h5_net/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_net.pt --traj $(gt_of $T) --out $O/h5_net > $O/h5_net.log 2>&1
  [ -f $O/net.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $O/h5_net --config $W/lbdata/config.json --shape lego --out $O/net.mp4 > $O/r_net.log 2>&1 < /dev/null
  [ -f $O/net.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  rclone copy $O $R/${O#$U/} -q; }

if [ $STAGE = test ]; then
  read g1 g2 <<<"$(pick2)"; echo "[GPU] $g1 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
  O=$U/test_$C/L000; run1 L000 $O $g1 2
  grep -a "출력 뉴턴\|망 프레임별\|Traceback\|Error" $O/run_all.log | tail -20
  grep -a -q OVNET_FIT_DONE $O/run_all.log && echo OVN_TEST_OK || echo OVN_TEST_BAD
  rclone copy $U/test_$C $R/test_$C -q; exit 0
fi

# ---- full ----
( while true; do sleep 300; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
read g1 g2 <<<"$(pick2)"; echo "[GPU] L000 $g1, 0004 $g2  $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
( run1 L000 $U/L000 $g1 80; render L000 $U/L000 $g1; echo "[끝] L000 $(date '+%F %T')" ) & P1=$!
( run1 0004 $U/0004 $g2 80; render 0004 $U/0004 $g2; echo "[끝] 0004 $(date '+%F %T')" ) & P2=$!
wait $P1 $P2
rclone copy $U $R -q
echo OVN_FULL_DONE > $U/OVN_FULL_DONE; rclone copy $U $R -q; echo "OVN_FULL_DONE $(date)"
