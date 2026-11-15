#!/bin/bash
# repflow A: GF 장면 정답 궤적 (GF CD-MPM 입자 전부) 을 다섯 표현이 L2 로 따라가기 (사용자 11-15). 사용: bash exe/go_repflow_gf.sh <커밋> <장면> [방법들 (기본 전부)]
# 장면 = R2 datasets/anchorflow/gf_scenes/gt/<장면>. 방법 = hex vrgsx gsverse sg pt1 pt10 pt100 pt1000 (카드 = 방법 번호 % 카드 수).
# ε·η = 기존 규칙 (wolf 앞 3 프레임 평균 RMSE 최소): hex 1e-3/0.5, PhysTwin 1e-3/10, GS-Verse 1e-3/1, VR-GS 1e-3/1, Spring-Gaus = R2 lbours/repflow1115/sg/tune_pick.txt. 시각 지표 = GF 공식 렌더 경로 (rep_render_gf.py: 방법 궤적 + F, gfs_render.py: 정답, 같은 카메라) PSNR/SSIM/LPIPS (gfs_post.py vid).
export PATH=/home/dkta/bin:$PATH
H=$1; S=$2; JOBS=${3:-hex vrgsx gsverse sg pt1 pt10 pt100 pt1000}
RA=r2:storage/result/anchorflow; R=$RA/lbours/repflow1115/gf/$S; GT=r2:storage/datasets/anchorflow/gf_scenes/gt/$S
O=/dev/shm/dkta/rfgf/$S; IN=$O/in; D=$O/main; mkdir -p $IN $D/metrics /dev/shm/dkta/tmp
export TMPDIR=/dev/shm/dkta/tmp TORCHINDUCTOR_CACHE_DIR=/dev/shm/dkta/rfgf_inductor TRITON_CACHE_DIR=/dev/shm/dkta/rfgf_triton PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=${PY:-python}; NG=${GPUS:-$(nvidia-smi -L | wc -l)}
L=$O/go.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go.log -q; }
say "시작 $(hostname) 커밋 $H 장면 $S 방법 $JOBS 카드 $NG"
if [ ! -f $IN/PREP_DONE ]; then
  # 여러 방법 묶음이 같은 장면을 따로 띄워도 준비는 한 번: 다른 인스턴스가 올린 입력이 R2 에 있으면 받는다
  if rclone lsf $R/inputs/gflow_$S.npz 2>/dev/null | grep -q .; then rclone copy $R/inputs $IN -q; else
    rclone copy $GT/simulation_ply $O/h5 -q --transfers 16 --include "sim_*.h5"; rclone copyto $GT/config.json $IN/config.json -q
    $PY -u exe/rep_prep_gf.py --h5_dir $O/h5 --name $S --out_dir $IN > $O/prep.log 2>&1; say "prep $(tail -1 $O/prep.log | cut -c1-200)"
    rclone copy $IN $R/inputs -q; fi
  [ -f $IN/config.json ] || rclone copyto $GT/config.json $IN/config.json -q
  [ -d $O/h5 ] || rclone copy $GT/simulation_ply $O/h5 -q --transfers 16 --include "sim_*.h5"
  touch $IN/PREP_DONE
fi
case $S in milk) MD=milk_0.03_psnr32;; kiwi) MD=kiwi_0.04_psnr42;; *) MD=$S;; esac
GF=/home/dkta/work/GaussianFluent; AFX=$PWD
[ -d $GF/gaussian-splatting ] || rclone copy $RA/_vastmirror/GaussianFluent $GF --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q
[ -d $GF/model_$S ] || { mkdir -p $GF/model_$S; rclone copy r2:storage/datasets/anchorflow/gf_scenes/model/$MD $GF/model_$S -q --exclude "chkpnt*" --exclude "point_cloud/iteration_7000/**"; }
KF=$(cd $GF && $PY $AFX/exe/rep_render_gf.py --gf $GF --model_path $GF/model_$S --config $IN/config.json --count 1 2>&1 | grep -a GS_NUM | tail -1 | awk '{print $2}'); say "가우시안 수 $KF"
NF=$($PY -c "import numpy as np; print(np.load('$IN/gflow_$S.npz')['traj'].shape[0])")
if [ ! -d $O/gt_frames ]; then ( cd $GF && $PY -u $AFX/exe/gfs_render.py --gf $GF --model_path $GF/model_$S --config $IN/config.json --frames $((NF - 1)) --sim $O/h5 --out $O/gt.mp4 < /dev/null > $O/gt_render.log 2>&1 ); mv $O/gt_frames $O/gt_png 2>/dev/null; mv $O/gt_png $O/gt_frames; say "정답 렌더 $(tail -1 $O/gt_render.log | cut -c1-120)"; rclone copy $O/gt.mp4 $R/render -q; fi
DX=$($PY -c "import json; c=json.load(open('$IN/config.json')); print(c.get('grid_lim',2.0)/c.get('n_grid',50))")
B="exe/rep_track_sg.py --opt riem --tb none --flow $IN/gflow_$S.npz --aux $IN/aux_$S.npz"
cmd_of() { case $1 in
  hex)     echo "$B --method hex --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 0.5 --hex_dx $DX";;
  vrgsx)   echo "$B --method vrgsx --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
  gsverse) echo "$B --method vrgs --riem_energy logbarrier --riem_eps 1e-3 --riem_lr 1";;
  sg)      until rclone lsf $RA/lbours/repflow1115/sg/tune_pick.txt 2>/dev/null | grep -q .; do sleep 30; done
           read -r _ E ETA <<< "$(rclone cat $RA/lbours/repflow1115/sg/tune_pick.txt | grep BEST)"
           echo "$B --method springgaus --riem_energy elastic --riem_eps $E --riem_lr $ETA";;
  pt*)     echo "$B --method phystwin --riem_energy elastic --riem_eps 1e-3 --riem_lr 10 --riem_k ${1#pt} --pt_realtime";;
esac; }
( while :; do sleep 600; rclone copy $D $R/main -q; done ) & SYNC=$!
one() { local m=$1 g=$2 n=${1}_$S c
  c=$(cmd_of $m)
  [ -f $D/$n.npz ] || { say "$n 카드 $g 시작: $c"; CUDA_VISIBLE_DEVICES=$g $PY -u $c --out $D/$n.npz --resume --save_traj --traj32 --save_F ${KF:-0} < /dev/null >> $D/$n.log 2>&1; say "$n rc=$? $(grep -a '요약\|rror' $D/$n.log | tail -2 | cut -c1-260 | tr '\n' ' ')"; }
  rclone copy $D $R/main -q
  [ -f $D/metrics/$n.json ] || { CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_metrics_gf.py --res $D/$n.npz --flow $IN/gflow_$S.npz --sim $O/h5 --out $D/metrics/$n.json < /dev/null > $D/metrics/$n.log 2>&1; say "지표 $n rc=$? $(grep -a '측정\|rror' $D/metrics/$n.log | tail -2 | tr '\n' ' ')"; }
  [ -f $D/metrics/${n}_topo.json ] || { $PY -u exe/rep_topo_gf.py --res $D/$n.npz --flow $IN/gflow_$S.npz --out $D/metrics/${n}_topo.json < /dev/null > $D/metrics/${n}_topo.log 2>&1; say "위상 $n rc=$? $(tail -1 $D/metrics/${n}_topo.log | cut -c1-260)"; }
  [ -f $D/metrics/${n}_visual.json ] || { mkdir -p $D/render; ( cd $GF && CUDA_VISIBLE_DEVICES=$g $PY -u $AFX/exe/rep_render_gf.py --gf $GF --model_path $GF/model_$S --config $IN/config.json --res $D/$n.npz --aux $IN/aux_$S.npz --out $D/render/$n.mp4 < /dev/null > $D/render/$n.log 2>&1 )
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/gfs_post.py vid --gt_png $O/gt_frames --our_png $D/render/${n}_frames --out_json $D/metrics/${n}_visual.json --out_mp4 $D/render/${n}_cmp.mp4 --fps 24 < /dev/null > $D/metrics/${n}_visual.log 2>&1
    say "시각 $n rc=$? $(tail -1 $D/metrics/${n}_visual.log | cut -c1-160)"; rm -rf $D/render/${n}_frames.tmp; }
  rclone copy $D $R/main -q; }
P=""; i=0; for m in $JOBS; do one $m $((i % NG)) & P="$P $!"; i=$((i + 1)); sleep 5; done
wait $P; kill $SYNC 2>/dev/null; rclone copy $D $R/main -q; say "끝"
