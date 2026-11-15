#!/bin/bash
# rep4 (11-15): 표현력 실험 장면 4 개 (wolf·bread = PG 흐름, 나머지 둘 = 새 형상 PG 장면 nshape 정답 궤적) 의 우리 판 (hex) 다시 돌리기 + 진단 + nshape 다섯 방법.
#   사용: bash exe/go_rep4.sh <커밋> '<장면:판 ...>'   (사용자 11-15: GaussianFluent 장면은 뺀다)
#   nshape 장면 판 = fix (우리 판) | pt1 pt10 pt100 pt1000 | gsverse | vrgsx | sg (ε·η = 기존 규칙 값 그대로, go_repflow_gf.sh 와 같은 명령).  판 base = 기존 hex (진단만 추가 --hex_diag, 결과 불변), feas = + 되돌림 정의역 (--feas), fix = + 입자 위치 장벽 (--hex_pbar --feas; 새 공통 규칙)
#   판 hex = 기존 우리 판 그대로 (플래그 없음; 사용자 11-15: 뒤집힘·화질 고치기 중단, L2 오차만).  env NOVIS=1 이면 추적 (RMSE 는 추적기 요약) 만 -- 지표·위상·렌더 안 함.
#   ε·η·dx = 기존 그대로 (hex 1e-3/0.5; wolf·bread dx 0.02 = 기존, nshape dx = grid_lim/n_grid). 카드 = 나열 순서 % 카드 수.
#   wolf·bread: rep_track_sg --video (r10 과 같은 렌더) + rep_metrics (목표 영상 = archive_20261009/r10).
#   nshape: 정답 = R2 datasets/anchorflow/nshape/gt/<장면> (i-PhysGaussian 공식 러너 h5, 입자 전부; rep_prep_gf 로 같은 정규화), 지표 = rep_metrics_gf · rep_topo_gf,
#           시각 = nshape 묶음과 같은 3DGS 렌더 경로 (gfs_render 정답 / rep_render_gf 방법, 같은 카메라) + gfs_post vid.
#   결과 R2 lbours/rep4/<장면>/<판>/
export PATH=/home/dkta/bin:$PATH
H=$1; JOBS=$2
RA=r2:storage/result/anchorflow; R4=$RA/lbours/rep4
O=/dev/shm/dkta/rep4; mkdir -p $O /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rep4_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rep4_triton PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=${PY:-python}; W=/home/dkta/work; PG=$W/i-physgaussian; GF=$W/GaussianFluent; AFX=$PWD
L=$O/go_$(hostname).log; say() { echo "[rep4 $(date '+%F %T')] $*" >> $L; rclone copyto $L $R4/go_$(hostname).log -q; }
say "시작 커밋 $H 판 $JOBS"
flags_of() { case $1 in hex) echo "";; base) echo "--hex_diag";; feas) echo "--hex_diag --feas";; fix) echo "--hex_diag --hex_pbar --feas";; fixL) echo "--hex_diag --hex_pbar --feas --hex_lmax_gauss";; esac; }
prep_pg() { local s=$1 IN=$O/in_$s m
  mkdir -p $IN
  for f in gflow_$s.npz aux_$s.npz; do [ -s $IN/$f ] || rclone copyto $RA/repflow/inputs/$f $IN/$f -q; done
  [ -s $IN/${s}_target.mp4 ] || rclone copyto $RA/archive_20261009/r10/${s}_target.mp4 $IN/${s}_target.mp4 -q
  m=$(basename $($PY -c "import numpy as np; print(np.load('$IN/aux_$s.npz', allow_pickle=True)['model'])"))
  [ -d $W/pgmodel/$m/point_cloud/iteration_30000 ] || rclone copy $RA/pgmodel/$m $W/pgmodel/$m -q --exclude "chkpnt*" --exclude "point_cloud/iteration_7000/**" --exclude "point_cloud/iteration_60000/**"; }
prep_ns() { local s=$1 IN=$O/in_$1 GT=r2:storage/datasets/anchorflow/nshape/gt/$1 MD=r2:storage/datasets/anchorflow/nshape/model/${1%%_*}
  rclone lsf r2:storage/datasets/anchorflow/nshape/gt_pg/$s/simulation_ply 2>/dev/null | grep -q . && GT=r2:storage/datasets/anchorflow/nshape/gt_pg/$s   # 원본 PhysGaussian 정답 (CM186 PG 경로)
  mkdir -p $IN
  [ -d $O/h5_$s ] || rclone copy $GT/simulation_ply $O/h5_$s -q --transfers 16 --include "sim_*.h5"
  [ -s $IN/config.json ] || rclone copyto $GT/config.json $IN/config.json -q
  if [ ! -s $IN/gflow_$s.npz ]; then
    if rclone lsf $R4/inputs/$s/gflow_$s.npz 2>/dev/null | grep -q .; then rclone copy $R4/inputs/$s $IN -q; else
      $PY -u exe/rep_prep_gf.py --h5_dir $O/h5_$s --name $s --out_dir $IN > $O/prep_$s.log 2>&1; say "prep $s $(tail -1 $O/prep_$s.log | cut -c1-200)"
      rclone copy $IN $R4/inputs/$s -q; fi; fi
  [ -d $GF/gaussian-splatting ] || rclone copy $RA/_vastmirror/GaussianFluent $GF --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q
  [ -d $GF/model_$s ] || { mkdir -p $GF/model_$s; rclone copy $MD $GF/model_$s -q --exclude "chkpnt*" --exclude "point_cloud/iteration_7000/**"; }
  if [ ! -d $O/gt_frames_$s ]; then
    local NF=$($PY -c "import numpy as np; print(np.load('$IN/gflow_$s.npz')['traj'].shape[0])")
    ( cd $GF && $PY -u $AFX/exe/gfs_render.py --gf $GF --model_path $GF/model_$s --config $IN/config.json --frames $((NF - 1)) --sim $O/h5_$s --out $O/gt_$s.mp4 < /dev/null > $O/gt_render_$s.log 2>&1 )
    mv $O/gt_${s}_frames $O/gt_frames_$s 2>/dev/null || mv $O/gt_frames $O/gt_frames_$s; rclone copyto $O/gt_$s.mp4 $R4/$s/gt.mp4 -q
    say "정답 렌더 $s $(tail -1 $O/gt_render_$s.log | cut -c1-120) $(ls $O/gt_frames_$s | wc -l) 장"; fi; }
one() { local s=$1 v=$2 g=$3 D=$O/$1/$2 n=hex_$1 IN=$O/in_$1 c R=$R4/$1/$2
  mkdir -p $D/metrics
  ( while :; do sleep 600; rclone copy $D $R -q --exclude "*.frames/**" --exclude "render/*_frames/**"; done ) & local SY=$!
  case $s in
  wolf|bread)
    c="exe/rep_track_sg.py --opt riem --tb none --method hex --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 0.5 --flow $IN/gflow_$s.npz --aux $IN/aux_$s.npz $(flags_of $v)"
    [ -f $D/$n.npz ] || { echo "CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --video $D/$n.mp4 --save_traj" > $D/$n.cmd.txt; say "$s/$v 카드 $g 시작: $c"
      CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --video $D/$n.mp4 --save_traj < /dev/null >> $D/$n.log 2>&1; say "$s/$v rc=$? $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"; }
    [ -f $D/metrics/$n.json ] || { CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_metrics.py --res $D/$n.npz --flow $IN/gflow_$s.npz --video $D/$n.mp4 \
      --target_video $IN/${s}_target.mp4 --out $D/metrics/$n.json < /dev/null > $D/metrics/$n.log 2>&1; say "지표 $s/$v rc=$? $(grep -a '측정\|rror' $D/metrics/$n.log | tail -2 | tr '\n' ' ')"; }
    ;;
  *)
    local DX=$($PY -c "import json; c=json.load(open('$IN/config.json')); print(c.get('grid_lim',2.0)/c.get('n_grid',50))")
    local KF=$(cd $GF && $PY $AFX/exe/rep_render_gf.py --gf $GF --model_path $GF/model_$s --config $IN/config.json --count 1 2>&1 | grep -a GS_NUM | tail -1 | awk '{print $2}')
    local B="exe/rep_track_sg.py --opt riem --tb none --flow $IN/gflow_$s.npz --aux $IN/aux_$s.npz"
    case $v in
      hex|base|feas|fix|fixL) c="$B --method hex --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 0.5 --hex_dx $DX $(flags_of $v)";;
      vrgsx)   n=vrgsx_$s; c="$B --method vrgsx --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
      gsverse) n=gsverse_$s; c="$B --method vrgs --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
      sg)      n=sg_$s; read -r _ E ETA <<< "$(rclone cat $RA/lbours/repflow1115/sg/tune_pick.txt | grep BEST)"
               c="$B --method springgaus --riem_energy elastic --riem_eps $E --riem_lr $ETA";;
      pt*)     n=${v}_$s; c="$B --method phystwin --riem_energy elastic --riem_eps 1e-3 --riem_lr 10 --riem_k ${v#pt} --pt_realtime";;
    esac
    [ -f $D/$n.npz ] || { echo "CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --save_traj --traj32 --save_F ${KF:-0}" > $D/$n.cmd.txt; say "$s/$v 카드 $g 시작 (가우시안 $KF): $c"
      CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --save_traj --traj32 --save_F ${KF:-0} < /dev/null >> $D/$n.log 2>&1; say "$s/$v rc=$? $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"; }
    rclone copy $D $R -q
    [ -n "${NOVIS:-}" ] || [ -f $D/metrics/$n.json ] || { CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_metrics_gf.py --res $D/$n.npz --flow $IN/gflow_$s.npz --sim $O/h5_$s --out $D/metrics/$n.json < /dev/null > $D/metrics/$n.log 2>&1; say "지표 $s/$v rc=$? $(grep -a '측정\|rror' $D/metrics/$n.log | tail -2 | tr '\n' ' ')"; }
    [ -n "${NOVIS:-}" ] || [ -f $D/metrics/${n}_topo.json ] || { $PY -u exe/rep_topo_gf.py --res $D/$n.npz --flow $IN/gflow_$s.npz --out $D/metrics/${n}_topo.json < /dev/null > $D/metrics/${n}_topo.log 2>&1; say "위상 $s/$v rc=$? $(tail -1 $D/metrics/${n}_topo.log | cut -c1-260)"; }
    [ -n "${NOVIS:-}" ] || [ -f $D/metrics/${n}_visual.json ] || { mkdir -p $D/render; ( cd $GF && CUDA_VISIBLE_DEVICES=$g $PY -u $AFX/exe/rep_render_gf.py --gf $GF --model_path $GF/model_$s --config $IN/config.json --res $D/$n.npz --aux $IN/aux_$s.npz --out $D/render/$n.mp4 < /dev/null > $D/render/$n.log 2>&1 )
      CUDA_VISIBLE_DEVICES=$g $PY -u exe/gfs_post.py vid --gt_png $O/gt_frames_$s --our_png $D/render/${n}_frames --out_json $D/metrics/${n}_visual.json --out_mp4 $D/render/${n}_cmp.mp4 --fps 24 < /dev/null > $D/metrics/${n}_visual.log 2>&1
      say "시각 $s/$v rc=$? $(tail -1 $D/metrics/${n}_visual.log | cut -c1-160)"; }
    ;;
  esac
  kill $SY 2>/dev/null; rclone copy $D $R -q --exclude "*.frames/**" --exclude "render/*_frames/**"; say "$s/$v 끝"; }
# 입력 준비 (장면마다 한 번)
for s in $(for j in $JOBS; do echo ${j%%:*}; done | sort -u); do case $s in wolf|bread) prep_pg $s;; *) prep_ns $s;; esac; done
say "입력 $(ls $O | tr '\n' ' ')"
NG=$(nvidia-smi -L | wc -l); GM=(${GPUMAP:-}); P=""; i=0; for j in $JOBS; do one ${j%%:*} ${j#*:} ${GM[$i]:-$((i % NG))} & P="$P $!"; i=$((i + 1)); sleep 5; done
wait $P; say "끝"
