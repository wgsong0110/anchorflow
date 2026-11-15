#!/bin/bash
# 절 CM189 (사용자 11-15 "bread 를 점탄성으로 바닥에 떨구기"): bread_foam_drop 정답 (원본 PhysGaussian 8339ed6, 안 되면 i-PG 사본) + 우리 판 명령 준비. plastfix_boot.sh 뒤에.
#   (1) 바닥 FLOOR = 0 으로 1 프레임 돌려 처음 최저점 z0 -> FLOOR = z0 - 0.3 (2) 50 프레임 정답 (3) 변환 (4) 우리 판 명령 (chair_paste K1 명령에서 장면 경로만 바꿈) -> /root/pr/bread_foam_drop.cmd
set -u
SC=bread_foam_drop; R=r2:storage/result/anchorflow; MK=$R/lbours/plastfix; DS=r2:storage/datasets/anchorflow/nshape; MI=$R/_vastmirror
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
W=/home/dkta/work; G=/root/gfs; AF=/root/af; LOG=/root/pr/bread.log
say() { echo "[bread $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK/bread.log -q 2>/dev/null; }
[ -d $W/pgmodel/bread-trained ] || rclone copy $MI/pgmodel/bread-trained $W/pgmodel/bread-trained --transfers 16 -q
mkdir -p $G/hf/model; ln -sfn $W/pgmodel/bread-trained $G/hf/model/bread
CFG0=$AF/exe/nshape_cfg/$SC.json; O=$G/gt/$SC; mkdir -p $O /root/pr/bt
runpg() {  # $1 = PG 폴더, $2 = config, $3 = 출력
  (cd $1 && WARP_CACHE_PATH=$W/wpc timeout 5400 python -u gs_simulation.py --model_path $W/pgmodel/bread-trained --config $2 --output_path $3 --output_h5 < /dev/null > $3/run.log 2>&1); }
PGD=$W/PhysGaussian_upstream; RUNNER="PhysGaussian 8339ed6 gs_simulation.py (원본 그대로)"
python - $CFG0 /root/pr/bt/cfg.json 0.0 1 <<'P'
import json, sys; c = json.load(open(sys.argv[1])); c["boundary_conditions"][1]["point"][2] = float(sys.argv[3]); c["frame_num"] = int(sys.argv[4]); json.dump(c, open(sys.argv[2], "w"), indent=1)
P
t0=$(date +%s); runpg $PGD /root/pr/bt/cfg.json /root/pr/bt; rc=$?
if [ $rc != 0 ] || [ ! -f /root/pr/bt/simulation_ply/sim_0000000000.h5 ]; then
  say "[원본 PG 실패] rc=$rc $(grep -a rror /root/pr/bt/run.log | tail -1 | cut -c1-200) -> 채우기 고침 사본 (exe/fix_pg_filling.py)"
  cp -r $PGD ${PGD}_fill; python $AF/exe/fix_pg_filling.py --pg ${PGD}_fill >> /root/pr/bt/fix.log 2>&1; PGD=${PGD}_fill; RUNNER="PhysGaussian 8339ed6 + exe/fix_pg_filling.py (채우기 고유분해만 f64)"
  rm -rf /root/pr/bt/simulation_ply; runpg $PGD /root/pr/bt/cfg.json /root/pr/bt; rc=$?
fi
Z0=$(python -c "import h5py,numpy as np;print(float(np.array(h5py.File('/root/pr/bt/simulation_ply/sim_0000000000.h5')['x'])[2].min()))")
FL=$(python -c "print(round($Z0 - 0.3, 4))"); say "[바닥] 처음 최저점 $Z0 -> 바닥 $FL ($(( $(date +%s) - t0 )) 초, $RUNNER)"
CFG=$G/hf/config/$SC.json
python - $CFG0 $CFG $FL 50 <<'P'
import json, sys; c = json.load(open(sys.argv[1])); c["boundary_conditions"][1]["point"][2] = float(sys.argv[3]); c["frame_num"] = int(sys.argv[4]); json.dump(c, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
P
t0=$(date +%s); runpg $PGD $CFG $O; rc=$?; echo "$RUNNER" > $O/RUNNER; cp $CFG $O/config.json
n=$(ls $O/simulation_ply | grep -c h5); say "[정답] rc=$rc 프레임 $n $(( $(date +%s) - t0 )) 초 $(grep -a 'Total particles\|AF시간' $O/run.log | tr '\n' ' ' | cut -c1-200)"
rclone copy $O $DS/gt_pg/$SC --transfers 16 -q
cd $AF; export PYTHONPATH=$G/GaussianFluent:$G/GaussianFluent/gaussian-splatting
mkdir -p $G/ours/${SC}_K1; cp $CFG $G/ours/${SC}_K1/cfg_floor.json
python -u exe/pg_h5_to_pt.py --h5_dir $O/simulation_ply --config $CFG --out_pt $G/ag/$SC.pt --out_dir $G/ag/gt_$SC --out_dir_n 1 --allow_nonfinite > /root/pr/cv_$SC.log 2>&1
python -u exe/pg_h5_to_pt.py --h5_dir $O/simulation_ply --config $CFG --out_pt /root/pr/gtfull_$SC.pt --out_dir $G/ag/gtfull_$SC --allow_nonfinite > /root/pr/cvfull_$SC.log 2>&1
say "[변환] $(tail -1 /root/pr/cv_$SC.log | cut -c1-150)"
sed -e "s#chair_paste#$SC#g" -e "s#/root/gfs/hf/model/chair#/root/gfs/hf/model/bread#" /root/pr/chair_paste.cmd > /root/pr/$SC.cmd
python - /root/afr/exe/lb_render.py $SC bread-trained <<'P'
import sys; f, sc, ob = sys.argv[1:]; s = open(f).read(); old = 'MODEL.update({"chair_plasticine"'
if '"%s": "%s"' % (sc, ob) not in s:
    assert old in s; s = s.replace(old, 'MODEL.update({"%s": "%s"})\n%s' % (sc, ob, old), 1); open(f, "w").write(s)
P
touch /root/pr/BREAD_READY; say "[준비 끝] 명령 /root/pr/$SC.cmd"
