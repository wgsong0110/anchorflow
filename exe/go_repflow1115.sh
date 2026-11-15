#!/bin/bash
# repflow A (L2 추적) 에 PG 장면 ficus·vasedeck·plane 추가 (사용자 11-15): 장면 하나 = 인스턴스 하나 (카드 4장). 사용: bash exe/go_repflow1115.sh <커밋> <장면> <단계[,단계]>
#   prep  입력 만들기 (rep_prep_scene -> gauss_flow -> simplicits 가중치 -> 목표 영상) -> R2 lbours/repflow1115/inputs/<장면>
#   cal   (bread 1 프레임) 기존 r10 첫 프레임 값과 맞춰 보는 명령줄 확인 -- R2 lbours/repflow1115/cal/
#   main  다섯 방법 (hex, PhysTwin k1/10/100/1000, GS-Verse, VR-GS, Simplicits) 120 프레임 + rep_metrics. PhysTwin 은 R2 lbours/repflow1115/flags.txt (cal 로 정한 추가 플래그) 가 생길 때까지 기다린다.
# ε·η = 기존 규칙으로 wolf 앞 3 프레임 평균 RMSE 최소에서 고른 값을 방법별로 그대로 (hex 1e-3/0.5, PhysTwin 1e-3/10, GS-Verse 1e-3/1, Simplicits 1e-1/0.1, VR-GS 1e-3/1).
# 코드 = exe/rep_track_vrgs.py (rep_track_hex + vrgsx; 다른 방법 코드는 r10 과 같음). 클러스터 경로 흉내 (/home/dkta/work, /dev/shm/dkta).
export PATH=/home/dkta/bin:$PATH
H=$1; S=$2; STGS=$3
RA=r2:storage/result/anchorflow; R=$RA/lbours/repflow1115
O=/dev/shm/dkta/rf15; IN=$O/inputs; W=/home/dkta/work; PG=$W/i-physgaussian; mkdir -p $O $IN /dev/shm/dkta/tmp $W/pgmodel
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rf15_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rf15_triton PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=${PY:-python}; NG=${GPUS:-$(nvidia-smi -L | wc -l)}
L=$O/go_${S}.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_${S}.log -q; }
case $S in
  ficus)   MODEL=ficus_whitebg-trained;   CFG=$PG/config/ficus_config.json; DX=0.04;;
  vasedeck) MODEL=vasedeck_whitebg-trained; CFG=$W/cfg/vasedeck_config.json; DX=0.0166666667;;
  plane)   MODEL=plane-trained;           CFG=$W/cfg/plane_config.json; DX=0.0166666667;;
  bread)   MODEL=bread-trained;;
esac
say "시작 $(hostname) 커밋 $H 장면 $S 단계 $STGS 카드 $NG"
fetch_model() { local m=$1; [ -d $W/pgmodel/$m/point_cloud/iteration_30000 ] || rclone copy $RA/pgmodel/$m $W/pgmodel/$m -q --exclude "chkpnt*" --exclude "point_cloud/iteration_7000/**" --exclude "point_cloud/iteration_60000/**"; }
OPT="--opt riem --tb none"
# 방법 -> 명령줄 (ε η 고정)
cmd_of() {  # $1 장면 $2 방법
  local s=$2 m=$1 base="exe/rep_track_vrgs.py $OPT --flow $IN/gflow_$S.npz --aux $IN/aux_$S.npz"
  case $m in
    hex)      echo "$base --method hex --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 0.5 --hex_dx $DX";;
    vrgsx)    echo "$base --method vrgsx --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
    gsverse)  echo "$base --method vrgs --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
    simp)     echo "$base --method simplicits --riem_energy logbarrier --riem_eps 1e-1 --riem_lr 0.1 --simp $IN/simp_$S.pt";;
    pt*)      echo "$base --method phystwin --riem_energy elastic --riem_eps 1e-3 --riem_lr 10 --riem_k ${m#pt} $PTF";;
  esac; }
for STG in ${STGS//,/ }; do
case $STG in
prep)
  fetch_model $MODEL
  mkdir -p $W/cfg; rclone copy $RA/lbours/paper_remeasure2/assets $W/cfg -q --include "*_config.json"
  [ -f $PG/config/ficus_config.json ] || true
  cd $PG && $PY -u $OLDPWD/exe/rep_prep_scene.py --name $S --cfg $CFG --model $W/pgmodel/$MODEL --out_dir $IN > $O/prep_$S.log 2>&1; cd $OLDPWD
  say "aux $(tail -1 $O/prep_$S.log | cut -c1-250)"
  $PY -u exe/gauss_flow.py --fill $IN/gauss_sim_$S.npy --pts $IN/aux_${S}_pts.npy --n 0 --out $IN/gflow_$S.npz >> $O/prep_$S.log 2>&1
  say "flow $(grep -a '흐름\|스텝\|Error' $O/prep_$S.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
  ( CUDA_VISIBLE_DEVICES=0 $PY -u exe/simplicits_weights.py --pts $IN/aux_${S}_pts.npy --out $IN/simp_$S.pt > $O/simp_$S.log 2>&1; say "simplicits $(tail -1 $O/simp_$S.log | cut -c1-200)" ) &
  ( cd $PG && CUDA_VISIBLE_DEVICES=1 $PY -u $OLDPWD/exe/rep_render.py --aux $IN/aux_$S.npz --target --flow $IN/gflow_$S.npz --out $IN/${S}_target.mp4 > $O/target_$S.log 2>&1; say "목표 영상 $(tail -1 $O/target_$S.log | cut -c1-200)" ) &
  wait
  rclone copy $IN $R/inputs/$S -q --include "*_$S*" --include "*$S.*" --include "${S}_target.mp4"; touch $IN/PREP_DONE_$S; say "prep 끝 $(ls $IN | tr '\n' ' ')"
  ;;
cal)
  # bread 1 프레임: 기존 r10 첫 프레임 값 (PhysTwin k10 RMSE 0.325%, GS-Verse 0.324%, Simplicits 0.546%) 과 맞는 플래그를 찾는다
  S0=$S; S=bread; fetch_model bread-trained
  for f in gflow_bread.npz aux_bread.npz simp_bread.pt; do [ -s $IN/$f ] || rclone copyto $RA/repflow/inputs/$f $IN/$f -q; done
  D=$O/cal; mkdir -p $D; i=0
  for v in "pt10:" "pt10:--pt_realtime" "gsverse:" "simp:"; do
    m=${v%%:*}; PTF=${v#*:}; g=$((i % NG)); i=$((i + 1)); n=$m$([ -n "$PTF" ] && echo _rt)
    ( c=$(cmd_of $m); CUDA_VISIBLE_DEVICES=$g $PY -u $c --frames 1 --out $D/$n.npz < /dev/null > $D/$n.log 2>&1
      say "cal $n $(grep -a 't=  1\]\|rror' $D/$n.log | tail -2 | cut -c1-200 | tr '\n' ' ')"; rclone copy $D $R/cal -q ) &
  done
  wait
  # 흐름 생성이 기존 (bread) 과 같은지: 같은 기본 인자로 다시 적분해 저장된 궤적과 대조
  CUDA_VISIBLE_DEVICES=0 $PY - $IN > $D/flowcheck.txt 2>&1 <<'PYEOF'
import sys, numpy as np, torch
sys.path.insert(0, "exe"); import gauss_flow as gf
IN = sys.argv[1]; Z = np.load(f"{IN}/gflow_bread.npz"); A = np.load(f"{IN}/aux_bread.npz", allow_pickle=True)
fld = gf.field_seq(0, 12, sig=(0.1, 0.3), amp=(2.0, 5.0), kinds=",".join(gf.KINDS).split(","))
print("field 최대 차", float(np.abs(np.asarray(fld, dtype=float) - Z["field"].astype(float)).max()))
x0 = torch.as_tensor(Z["X0"], dtype=torch.float64, device="cuda")
tr, _, _ = gf.run_flow(x0, fld, 120, 10, 1 / 30.0, 0.5, keep_F=False)
print("traj 최대 차", float((tr.cpu().numpy() - Z["traj"]).__abs__().max()), "X0 == aux G", float(np.abs(Z["X0"] - A["G"]).max()))
PYEOF
  say "흐름 대조 $(cat $D/flowcheck.txt | tr '\n' ' ' | cut -c1-300)"; rclone copy $D $R/cal -q; say "cal 끝"; S=$S0
  ;;
main)
  fetch_model $MODEL
  for f in gflow_$S.npz aux_$S.npz simp_$S.pt ${S}_target.mp4; do
    [ -s $IN/$f ] || { until rclone lsf $R/inputs/$S/$f 2>/dev/null | grep -q .; do sleep 30; done; sleep 20; rclone copyto $R/inputs/$S/$f $IN/$f -q; }   # prep 가 다른 인스턴스에서 끝나기를 기다림
  done
  D=$O/main; mkdir -p $D
  ( while :; do sleep 600; rclone copy $D $R/main/$S -q; done ) & SYNC=$!
  one() { local m=$1 g=$2 n=${1}_$S c vid
    if [ ${m#pt} != $m ]; then while ! rclone lsf $R/flags.txt >/dev/null 2>&1; do sleep 30; done; PTF=$(rclone cat $R/flags.txt | grep -v '^#' | head -1); fi
    c=$(cmd_of $m); [ -f $D/$n.npz ] || { say "$n 카드 $g 시작: $c"
      echo "CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --video $D/$n.mp4 --save_traj" > $D/$n.cmd.txt
      CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --video $D/$n.mp4 --save_traj < /dev/null >> $D/$n.log 2>&1; say "$n rc=$? $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"; }
    rclone copy $D $R/main/$S -q
    mkdir -p $D/metrics; [ -f $D/metrics/$n.json ] || { CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_metrics.py --res $D/$n.npz --flow $IN/gflow_$S.npz --video $D/$n.mp4 \
      --target_video $IN/${S}_target.mp4 --out $D/metrics/$n.json < /dev/null > $D/metrics/$n.log 2>&1; say "지표 $n rc=$? $(grep -a '측정\|rror' $D/metrics/$n.log | tail -2 | tr '\n' ' ')"; }
    rclone copy $D $R/main/$S -q; }
  P=""; i=0
  for m in ${JOBS:-hex vrgsx gsverse simp pt1 pt10 pt100 pt1000}; do one $m $((i % NG)) & P="$P $!"; i=$((i + 1)); sleep 5; done
  wait $P; kill $SYNC 2>/dev/null
  rclone copy $D $R/main/$S -q; say "main 끝"
  ;;
esac
done
say "끝 $STGS"; echo done > $O/go_${S}_${STGS//,/_}.END; rclone copyto $O/go_${S}_${STGS//,/_}.END $R/go_${S}_${STGS//,/_}.END -q
