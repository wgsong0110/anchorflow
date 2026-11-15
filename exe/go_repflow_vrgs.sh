#!/bin/bash
# repflow A (L2 추적) 에 VR-GS (XPBD 사면체 케이지) 를 GS-Verse 와 따로 넣는 묶음 (사용자 11-15). 사용: bash exe/go_repflow_vrgs.sh <커밋> <단계[,단계]>
#   s0   연기: wolf 2 프레임 (영상 포함) + 지표 한 번
#   tune ε·η 고르기 (기존 규칙 그대로: wolf 앞 3 프레임 평균 RMSE 최소, NaN 제외) ε {1e-3,1e-2,1e-1,1} x η {0.1,0.5,1,10,100}
#   main 세 형상 120 프레임 (tune 결과 ε·η), 영상·궤적 저장, 끝나면 rep_metrics (EMD 8192 헝가리안·물리·PSNR/SSIM/LPIPS/FID)
# 코드 = exe/rep_track_vrgs.py (= rep_track_hex.py + 방법 vrgsx). 계량 = 사면체 det 로그 장벽 (--riem_energy logbarrier), 케이지 = vrgs_xpbd 와 같은 규칙 (정점 ≤ 30K).
# vast.ai: 클러스터 경로를 흉내 (/home/dkta/work, /dev/shm/dkta -> /workspace/shm). env GPUS (카드 수), PAR (동시 실행 수, 기본 카드당 3). 결과는 R2 lbours/repflow_vrgs.
export PATH=/home/dkta/bin:$PATH
H=$1; STGS=$2
R=r2:storage/result/anchorflow/lbours/repflow_vrgs
O=/dev/shm/dkta/rfv; IN=$O/inputs; mkdir -p $O $IN /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rfv_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rfv_triton
export PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=${PY:-python}
NG=${GPUS:-$(nvidia-smi -L | wc -l)}; PAR=${PAR:-$((NG * 3))}
L=$O/go_${STGS//,/_}.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_${STGS//,/_}.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STGS 카드 $NG 동시 $PAR; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4, $5}') 남음"
for s in wolf bread ship; do
  for f in gflow_$s.npz aux_$s.npz; do
    [ -s $IN/$f ] || { rclone copyto r2:storage/result/anchorflow/repflow/inputs/$f $IN/$f.part -q && mv $IN/$f.part $IN/$f; }
  done
  [ -s $IN/${s}_target.mp4 ] || { rclone copyto r2:storage/result/anchorflow/archive_20261009/r10/${s}_target.mp4 $IN/${s}_target.mp4.part -q && mv $IN/${s}_target.mp4.part $IN/${s}_target.mp4; }
done
say "입력 $(ls -la $IN | awk 'NR>1{printf "%s %s, ", $9, $5}')"
say "PG 렌더러 $(ls -d /home/dkta/work/i-physgaussian 2>&1 | head -1); 3DGS $(ls -d /home/dkta/work/pgmodel/{wolf_whitebg,bread,ship_whitebg}-trained 2>&1 | tr '\n' ' ')"
TR="exe/rep_track_vrgs.py --opt riem --riem_energy logbarrier --method vrgsx --tb none"
# 카드 배정: 작업 번호 % 카드 수 (같은 카드에 PAR/NG 개까지)
SLOT=$O/slot; mkdir -p $SLOT
waitpar() { while [ $(ls $SLOT | wc -l) -ge $PAR ]; do sleep 5; done; }
# 판 하나: run <폴더> <형상> <프레임 0=전부> <ε> <η> <영상 1/0> [추가]
run() { local D=$1 s=$2 nf=$3 e=$4 eta=$5 vid=$6; shift 6; local n=${s}_vrgsx c g
  mkdir -p $D; [ -f $D/$n.npz ] && return 0
  g=$(( $(date +%s%N) % NG ))
  c="$TR --flow $IN/gflow_$s.npz --aux $IN/aux_$s.npz --riem_eps $e --riem_lr $eta --out $D/$n.npz --resume $*"
  [ $nf != 0 ] && c="$c --frames $nf"
  if [ $vid = 1 ]; then
    c="$c --save_traj"
    if [ $s = ship ]; then   # ship 3DGS 모델은 R2 에 없고 클러스터에만 있다 (올릴 방법 없음, 사용자 11-15) -> 영상·시각 지표 없이 (논문 표 각주)
      [ -d /home/dkta/work/pgmodel/ship_whitebg-trained/point_cloud ] && c="$c --video $D/$n.mp4" || say "ship 3DGS 모델 없음 -> 영상·시각 지표 없이"
    else c="$c --video $D/$n.mp4"; fi
  fi
  echo "CUDA_VISIBLE_DEVICES=$g $PY -u $c" > $D/$n.cmd.txt
  say "$D/$n 카드 $g 시작 (ε $e η $eta 프레임 $nf)"
  CUDA_VISIBLE_DEVICES=$g $PY -u $c < /dev/null >> $D/$n.log 2>&1; local rc=$?
  say "$D/$n rc=$rc $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"
  return $rc; }
up() { rclone copy $1 $R/$2 -q --transfers 8 --exclude "*.frames/**"; }
metrics() { local D=$1 s=$2
  mkdir -p $D/metrics; [ -f $D/metrics/${s}_vrgsx.json ] && return 0
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/rep_metrics.py --res $D/${s}_vrgsx.npz --flow $IN/gflow_$s.npz --video $D/${s}_vrgsx.mp4 \
    --target_video $IN/${s}_target.mp4 --out $D/metrics/${s}_vrgsx.json < /dev/null > $D/metrics/${s}_vrgsx.log 2>&1
  say "지표 $s rc=$? $(grep -a '측정\|rror' $D/metrics/${s}_vrgsx.log | tail -2 | tr '\n' ' ')"; }
for STG in ${STGS//,/ }; do
case $STG in
s0)
  D=$O/smoke; mkdir -p $D
  run $D wolf 2 1e-1 100 1
  metrics $D wolf
  up $D smoke; say "s0 끝"
  ;;
tune)
  D=$O/tune; mkdir -p $D
  ( while :; do sleep 300; up $D tune; done ) & SYNC=$!
  P=""
  for e in 1e-3 1e-2 1e-1 1; do for eta in 0.1 0.5 1 10 100; do
    waitpar
    ( f=$SLOT/$BASHPID; touch $f; run $D/e${e}_l${eta} wolf 3 $e $eta 0; up $D/e${e}_l${eta} tune/e${e}_l${eta}; rm -f $f ) &
    P="$P $!"; sleep 3
  done; done
  wait $P
  kill $SYNC 2>/dev/null
  $PY - $D > $D/tune_pick.txt 2>&1 <<'PYEOF'
import glob, json, os, sys
import numpy as np
D = sys.argv[1]; res = []
for p in sorted(glob.glob(os.path.join(D, "e*_l*", "wolf_vrgsx.npz"))):
    k = os.path.basename(os.path.dirname(p)); e, l = k[1:].split("_l")
    M = np.load(p)["metrics"]; v = float(np.mean(M[:, 1])) * 100
    res.append(dict(eps=e, eta=l, rmse3=v, det_min=float(np.min(M[:, 4])), inv_max=100 * float(np.max(M[:, 5]))))
ok = [r for r in res if r["rmse3"] == r["rmse3"]]
best = min(ok, key=lambda r: r["rmse3"])
for r in sorted(res, key=lambda r: (float(r["eps"]), float(r["eta"]))):
    print(f"ε {r['eps']:>5} η {r['eta']:>4}  RMSE(3) {r['rmse3']:.4f}%  det 최소 {r['det_min']:.3e}  뒤집힘 {r['inv_max']:.2f}%")
print("BEST", best["eps"], best["eta"])
json.dump(dict(all=res, best=best), open(os.path.join(D, "tune.json"), "w"), indent=1)
PYEOF
  say "tune 고름: $(grep BEST $D/tune_pick.txt) (빠진 판 $(( 20 - $(ls $D/e*/wolf_vrgsx.npz 2>/dev/null | wc -l) )))"
  up $D tune; say "tune 끝"
  ;;
main)
  D=$O/main; mkdir -p $D
  [ -f $O/tune/tune_pick.txt ] || rclone copyto $R/tune/tune_pick.txt $O/tune/tune_pick.txt -q
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
