#!/bin/bash
# repflow A: Spring-Gaus (공식 표현 초기 설정, rep_track_sg.py 방법 springgaus) 판. 사용:
#   bash exe/go_repflow_sg.sh <커밋> tune                 wolf 앞 3 프레임 평균 RMSE 최소로 ε·η 고르기 (ε {1e-3,1e-2,1e-1,1} x η {0.1,0.5,1,10,100}), 카드 GPUS 장에 나눠
#   bash exe/go_repflow_sg.sh <커밋> main <장면> <카드>   120 프레임 + 영상 + rep_metrics. ε·η = R2 lbours/repflow1115/sg/tune_pick.txt (없으면 생길 때까지 기다림)
# 장면 wolf·bread = R2 repflow/inputs (+ 목표 영상 archive_20261009/r10), ficus·vasedeck·plane = R2 lbours/repflow1115/inputs/<장면>.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2; S=${3:-}; GPU=${4:-0}
RA=r2:storage/result/anchorflow; R=$RA/lbours/repflow1115; W=/home/dkta/work
O=/dev/shm/dkta/rfsg; IN=$O/inputs; mkdir -p $O $IN /dev/shm/dkta/tmp $W/pgmodel
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rfsg_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rfsg_triton PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=${PY:-python}; NG=${GPUS:-$(nvidia-smi -L | wc -l)}
L=$O/go_sg_${STG}_${S}.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/sg/go_sg_${STG}_${S}.log -q; }
fetch_model() { local m=$1; [ -d $W/pgmodel/$m/point_cloud/iteration_30000 ] || rclone copy $RA/pgmodel/$m $W/pgmodel/$m -q --exclude "chkpnt*" --exclude "point_cloud/iteration_7000/**" --exclude "point_cloud/iteration_60000/**"; }
inputs() { local s=$1 f
  case $s in
    wolf|bread) for f in gflow_$s.npz aux_$s.npz; do [ -s $IN/$f ] || rclone copyto $RA/repflow/inputs/$f $IN/$f -q; done
                [ -s $IN/${s}_target.mp4 ] || rclone copyto $RA/archive_20261009/r10/${s}_target.mp4 $IN/${s}_target.mp4 -q;;
    *) for f in gflow_$s.npz aux_$s.npz ${s}_target.mp4; do [ -s $IN/$f ] || { until rclone lsf $R/inputs/$s/$f 2>/dev/null | grep -q .; do sleep 30; done; rclone copyto $R/inputs/$s/$f $IN/$f -q; }; done;;
  esac; }
TR="exe/rep_track_sg.py --opt riem --riem_energy elastic --method springgaus --tb none"
up() { rclone copy $1 $R/sg/$2 -q --transfers 8; }
case $STG in
tune)
  say "tune 시작 $(hostname) 커밋 $H 카드 $NG"; S=wolf; inputs wolf; D=$O/tune; mkdir -p $D; i=0; P=""
  for e in 1e-3 1e-2 1e-1 1; do for eta in 0.1 0.5 1 10 100; do
    ( while [ $(pgrep -fc "rep_track_sg.py.*--frames 3") -ge $((NG * 3)) ]; do sleep 5; done
      n=e${e}_l${eta}; mkdir -p $D/$n; CUDA_VISIBLE_DEVICES=$((i % NG)) $PY -u $TR --flow $IN/gflow_wolf.npz --aux $IN/aux_wolf.npz --riem_eps $e --riem_lr $eta --frames 3 --out $D/$n/wolf_sg.npz < /dev/null > $D/$n/wolf_sg.log 2>&1
      say "tune $n rc=$? $(grep -a '요약' $D/$n/wolf_sg.log | cut -c1-200)" ) &
    P="$P $!"; i=$((i + 1)); sleep 4
  done; done; wait $P
  $PY - $D > $D/tune_pick.txt 2>&1 <<'PYEOF'
import glob, json, os, sys
import numpy as np
D = sys.argv[1]; res = []
for p in sorted(glob.glob(os.path.join(D, "e*_l*", "wolf_sg.npz"))):
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
  up $D tune; rclone copyto $D/tune_pick.txt $R/sg/tune_pick.txt -q; say "tune 고름: $(grep BEST $D/tune_pick.txt)"
  ;;
main)
  fetch_model $(case $S in wolf) echo wolf_whitebg-trained;; bread) echo bread-trained;; ficus) echo ficus_whitebg-trained;; vasedeck) echo vasedeck_whitebg-trained;; plane) echo plane-trained;; esac)
  inputs $S
  until rclone lsf $R/sg/tune_pick.txt 2>/dev/null | grep -q .; do sleep 30; done
  read -r _ E ETA <<< "$(rclone cat $R/sg/tune_pick.txt | grep BEST)"
  D=$O/main; mkdir -p $D $D/metrics; n=sg_$S
  say "$n 카드 $GPU 시작 ε $E η $ETA"
  ( while :; do sleep 600; up $D main/$S; done ) & SYNC=$!
  [ -f $D/$n.npz ] || { CUDA_VISIBLE_DEVICES=$GPU $PY -u $TR --flow $IN/gflow_$S.npz --aux $IN/aux_$S.npz --riem_eps $E --riem_lr $ETA --out $D/$n.npz --resume --video $D/$n.mp4 --save_traj < /dev/null >> $D/$n.log 2>&1; say "$n rc=$? $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"; }
  up $D main/$S
  [ -f $D/metrics/$n.json ] || { CUDA_VISIBLE_DEVICES=$GPU $PY -u exe/rep_metrics.py --res $D/$n.npz --flow $IN/gflow_$S.npz --video $D/$n.mp4 --target_video $IN/${S}_target.mp4 --out $D/metrics/$n.json < /dev/null > $D/metrics/$n.log 2>&1; say "지표 $n rc=$? $(grep -a '측정\|rror' $D/metrics/$n.log | tail -2 | tr '\n' ' ')"; }
  kill $SYNC 2>/dev/null; up $D main/$S; say "$n 끝"
  ;;
esac
