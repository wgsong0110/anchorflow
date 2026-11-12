#!/bin/bash
# repflow A (L2 추적) '우리' 칸 육면체 판 다시 재기 (사용자 11-12) 진입점. 사용: bash exe/go_repflow_hex.sh <커밋> <단계[,단계]>
#   s0   연기: hex 야코비안 검사 (wolf) + 세 형상 2 프레임 (영상 포함) + 지표 스크립트 한 번 (wolf 2 프레임)
#   tune ε·η 고르기 (기존 규칙 그대로: wolf 앞 3 프레임 평균 RMSE 최소, NaN 제외) ε {1e-3,1e-2,1e-1,1} x η {0.1,0.5,1,10,100}
#   main 세 형상 120 프레임 (tune 결과 ε·η), 영상·궤적 저장, 끝나면 rep_metrics (EMD 8192 헝가리안·물리·PSNR/SSIM/LPIPS/FID)
# 코드 = exe/rep_track_hex.py (224443c rep_track2 + hex + 이어 돌리기). 비교 방법 기존 결과는 건드리지 않는다.
# 노드 / ·/tmp 안 씀 (코드·자료·결과·TMPDIR·컴파일 캐시 /dev/shm/dkta). 카드는 공용 표식 디렉토리 /dev/shm/dkta/anerr_claim 로
# 계산 프로세스 0 인 카드만, 판 하나 = 카드 하나, 끝나면 바로 놓는다 (stabhex 등 다른 묶음과 겹치지 않게).
export PATH=/home/dkta/bin:$PATH
H=$1; STGS=$2
R=r2:storage/result/anchorflow/lbours/repflow_hex
O=/dev/shm/dkta/rfh; IN=$O/inputs; mkdir -p $O $IN /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rfh_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rfh_triton
export PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_${STGS//,/_}.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_${STGS//,/_}.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STGS; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4, $5}') 남음"
# ---- 입력 (R2): 흐름·aux = repflow/inputs, 목표 영상 = r10 (archive_20261009/r10)
for s in wolf bread ship; do
  for f in gflow_$s.npz aux_$s.npz; do
    [ -s $IN/$f ] || { rclone copyto r2:storage/result/anchorflow/repflow/inputs/$f $IN/$f.part -q && mv $IN/$f.part $IN/$f; }
  done
  [ -s $IN/${s}_target.mp4 ] || { rclone copyto r2:storage/result/anchorflow/archive_20261009/r10/${s}_target.mp4 $IN/${s}_target.mp4.part -q && mv $IN/${s}_target.mp4.part $IN/${s}_target.mp4; }
done
say "입력 $(ls -la $IN | awk 'NR>1{printf "%s %s, ", $9, $5}')"
say "PG 렌더러 $(ls -d /home/dkta/work/i-physgaussian 2>&1 | head -1); 3DGS $(ls -d /home/dkta/work/pgmodel/{wolf_whitebg,bread,ship_whitebg}-trained 2>&1 | tr '\n' ' ')"
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
TR="exe/rep_track_hex.py --opt riem --riem_energy logbarrier --method hex --tb none"
# 판 하나: run <폴더> <형상> <프레임 0=전부> <ε> <η> <영상 1/0> [추가]
run() { local D=$1 s=$2 nf=$3 e=$4 eta=$5 vid=$6; shift 6; local g me=$BASHPID n=${s}_hex c
  mkdir -p $D; [ -f $D/$n.npz ] && return 0
  g=$(claim_wait $me)
  c="$TR --flow $IN/gflow_$s.npz --aux $IN/aux_$s.npz --riem_eps $e --riem_lr $eta --out $D/$n.npz --resume $*"
  [ $nf != 0 ] && c="$c --frames $nf"
  [ $vid = 1 ] && c="$c --video $D/$n.mp4 --save_traj"
  echo "CUDA_VISIBLE_DEVICES=$g $PY -u $c" > $D/$n.cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $D/$n.apps_at_start.txt 2>&1
  say "$D/$n 카드 $g 시작 (ε $e η $eta 프레임 $nf)"
  CUDA_VISIBLE_DEVICES=$g $PY -u $c < /dev/null >> $D/$n.log 2>&1; local rc=$?
  rm -f $CL/$g
  say "$D/$n rc=$rc $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"
  return $rc; }
up() { rclone copy $1 $R/$2 -q --transfers 8 --exclude "*.frames/**"; }
metrics() { local D=$1 s=$2 g me=$BASHPID
  mkdir -p $D/metrics; [ -f $D/metrics/${s}_hex.json ] && return 0
  g=$(claim_wait $me)
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_metrics.py --res $D/${s}_hex.npz --flow $IN/gflow_$s.npz --video $D/${s}_hex.mp4 \
    --target_video $IN/${s}_target.mp4 --out $D/metrics/${s}_hex.json < /dev/null > $D/metrics/${s}_hex.log 2>&1
  local rc=$?; rm -f $CL/$g
  say "지표 $s rc=$rc $(grep -a '측정\|rror' $D/metrics/${s}_hex.log | tail -2 | tr '\n' ' ')"; }
for STG in ${STGS//,/ }; do
case $STG in
s0)
  D=$O/smoke; mkdir -p $D
  ( g=$(claim_wait $BASHPID); CUDA_VISIBLE_DEVICES=$g $PY -u $TR --flow $IN/gflow_wolf.npz --aux $IN/aux_wolf.npz --out $D/chk.npz --check_jac < /dev/null > $D/check_jac.log 2>&1; rm -f $CL/$g
    say "야코비안 검사 $(grep -a '검사\|시간\|rror' $D/check_jac.log | tr '\n' ' ' | cut -c1-600)" ) &
  for s in wolf bread ship; do ( run $D $s 2 1e-1 100 1 ) & sleep 2; done
  wait
  metrics $D wolf
  up $D smoke; say "s0 끝"
  ;;
rt)
  # 이어 돌리기 확인: A = 4 프레임 한 번에, B = 3 프레임 (t=2 체크포인트 남김) 뒤 같은 출력으로 --frames 4 --resume (t=3 부터)
  D=$O/rtest; mkdir -p $D
  ( g=$(claim_wait $BASHPID); X="$TR --flow $IN/gflow_bread.npz --aux $IN/aux_bread.npz --riem_eps 1e-1 --riem_lr 100 --ckpt_every 2 --save_traj"
    CUDA_VISIBLE_DEVICES=$g $PY -u $X --frames 4 --out $D/A.npz --video $D/A.mp4 < /dev/null > $D/A.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -u $X --frames 3 --out $D/B.npz --video $D/B.mp4 < /dev/null > $D/B1.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -u $X --frames 4 --out $D/B.npz --video $D/B.mp4 --resume < /dev/null > $D/B2.log 2>&1
    rm -f $CL/$g
    $PY -c "
import numpy as np
a, b = np.load('$D/A.npz'), np.load('$D/B.npz')
print('A', a['metrics'][:, 1].tolist()); print('B', b['metrics'][:, 1].tolist())
print('traj 최대 차', float(np.abs(a['traj'].astype(np.float32) - b['traj'].astype(np.float32)).max()), 'curve 길이', a['curve'].shape, b['curve'].shape)
" > $D/cmp.txt 2>&1
    say "이어 돌리기 확인 $(grep -a '이어 돌리기' $D/B2.log | head -1) | $(cat $D/cmp.txt | tr '\n' ' ')" )
  up $D rtest
  ;;
tune)
  D=$O/tune; mkdir -p $D; Q=$D/queue.txt
  [ -f $Q ] || for e in 1e-3 1e-2 1e-1 1; do for eta in 0.1 0.5 1 10 100; do echo "$e $eta"; done; done > $Q
  echo 0 > $D/qpos
  lane() { local i ln
    while :; do
      i=$(flock $D/qpos.lock bash -c "i=\$(cat $D/qpos); echo \$((i + 1)) > $D/qpos; echo \$i")
      ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
      set -- $ln; run $D/e$1_l$2 wolf 3 $1 $2 0; up $D/e$1_l$2 tune/e$1_l$2
    done; }
  for k in $(seq 1 $NG); do lane & sleep 2; done
  wait
  $PY - $D > $D/tune_pick.txt 2>&1 <<'EOF'
import glob, json, os, sys
import numpy as np
D = sys.argv[1]; res = []
for p in sorted(glob.glob(os.path.join(D, "e*_l*", "wolf_hex.npz"))):
    k = os.path.basename(os.path.dirname(p)); e, l = k[1:].split("_l")
    M = np.load(p)["metrics"]; v = float(np.mean(M[:, 1])) * 100
    res.append(dict(eps=e, eta=l, rmse3=v, det_min=float(np.min(M[:, 4])), inv_max=100 * float(np.max(M[:, 5]))))
ok = [r for r in res if r["rmse3"] == r["rmse3"]]
best = min(ok, key=lambda r: r["rmse3"])
for r in sorted(res, key=lambda r: (float(r["eps"]), float(r["eta"]))):
    print(f"ε {r['eps']:>5} η {r['eta']:>4}  RMSE(3) {r['rmse3']:.4f}%  det 최소 {r['det_min']:.3e}  뒤집힘 {r['inv_max']:.2f}%")
print("BEST", best["eps"], best["eta"])
json.dump(dict(all=res, best=best), open(os.path.join(D, "tune.json"), "w"), indent=1)
EOF
  say "tune 고름: $(grep BEST $D/tune_pick.txt) (빠진 판 $(( 20 - $(ls $D/e*/wolf_hex.npz 2>/dev/null | wc -l) )))"
  up $D tune; say "tune 끝"
  ;;
main)
  D=$O/main; mkdir -p $D
  read -r _ E ETA <<< "$(grep BEST $O/tune/tune_pick.txt)"
  [ -n "$ETA" ] || { say "tune 결과 없음 -> main 안 함"; exit 1; }
  say "main ε $E η $ETA"
  ( while :; do sleep 600; up $D main; done ) & SYNC=$!
  P=""; for s in wolf bread ship; do ( run $D $s 0 $E $ETA 1 && { up $D main; metrics $D $s; up $D main; } ) & P="$P $!"; sleep 5; done
  wait $P
  kill $SYNC 2>/dev/null
  up $D main; say "main 끝"
  ;;
esac
done
say "끝"; echo done > $O/go_${STGS//,/_}.END; rclone copyto $O/go_${STGS//,/_}.END $R/go_${STGS//,/_}.END -q
