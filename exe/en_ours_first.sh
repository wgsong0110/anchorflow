#!/bin/bash
# elastic_new (사용자 11-15 "hotdog 는 우리 판을 먼저"): 정답 (i-PG 사본) 을 띄우고, 0 프레임 (채운 입자) 이 나오자마자 그 상태로 우리 K=1 (공통 수정 판) 을 먼저 돌려
# 한 칸 영상을 만든다 (오차는 의미 없음 -- 정답 자리에 0 프레임을 채움). 단독 시간 판은 정답 프로세스를 SIGSTOP 한 채로 (카드 단독). 정답이 끝나면 R2 에 올리고 vast_elastic_new.sh 로 두 칸.
#   env: R2_*, XS_SC, XS_OBJ, XS_CODE
set -u
SC=${XS_SC:?}; OBJ=${XS_OBJ:?}; TAG=elastic_new_$SC
R=r2:storage/result/anchorflow; RX=$R/lbours/elastic_new; DS=r2:storage/datasets/anchorflow/nshape; VD=$R/v20261009; MI=$R/_vastmirror
mkdir -p /root/sv; LOG=/root/sv/$TAG.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
W=/home/dkta/work; AF=/root/sv/af_$SC; AG=/dev/shm/dkta/ag_$SC; IPG=$W/i-physgaussian; MD=$W/pgmodel/$OBJ
mkdir -p $AF $AG /dev/shm/dkta/tmp; export HOME=/root PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp
rclone cat ${XS_CODE:?}/af_code.tar | tar -x -C $AF
[ -d $MD/point_cloud ] || rclone copy $DS/model/$OBJ $MD --transfers 16 -q
CFG=$W/cfg_$SC.json; cp $AF/exe/nshape_cfg/$SC.json $CFG; NF=$(python -c "import json;print(json.load(open('$CFG'))['frame_num'])")
say "[먼저 우리] $(python $AF/exe/elastic_new_floor.py $MD $CFG)"
OC=$W/gt/${SC}_ipg; mkdir -p $OC; t0=$(date +%s)
(cd $IPG && WARP_CACHE_PATH=$W/wpc2 timeout 3600 python -u gs_simulation.py --model_path $MD --config $CFG --output_path $OC --output_h5 < /dev/null > $OC/run.log 2>&1) & PC=$!
until [ -f $OC/simulation_ply/sim_0000000001.h5 ] || ! kill -0 $PC 2>/dev/null; do sleep 5; done
[ -f $OC/simulation_ply/sim_0000000001.h5 ] || { say "[정답 실패] $(tail -2 $OC/run.log)"; exit 1; }
GPAT="gs_simulation.py --model_path $MD --config $CFG --output_path $OC"; say "[정답 0 프레임] $(( $(date +%s) - t0 )) 초, 정답 pid $(pgrep -f "$GPAT" | tr '\n' ' ') 계속 돌림"
F0=$W/f0_$SC/simulation_ply; mkdir -p $F0; for i in $(seq 0 $NF); do ln -sf $OC/simulation_ply/sim_0000000000.h5 $F0/$(printf sim_%010d.h5 $i); done
cd $AF
python -u exe/pg_h5_to_pt.py --h5_dir $F0 --config $CFG --out_pt $AG/${SC}_f0.pt --out_dir $AG/gt0_$SC > $W/cv0_$SC.log 2>&1 || { say "[변환 실패] $(tail -1 $W/cv0_$SC.log)"; exit 1; }
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $W/BA_1.txt
mk() { local D=$1 drop=$2 nodump=$3
  local C=$(NS_DROP="$drop" NS_NODUMP=$nodump python exe/nshape_cmd.py $W/BA_1.txt $AG/${SC}_f0.pt $NF $SC $D $AG/gt0_$SC $AG/gt0_$SC/config.json 1)
  sed -e 's#python -u exe/#python -u exe/ssfix_wrap.py --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1 exe/#' -e 's/--ue_loc 50 /--ue_loc 51 /' -e 's/--wrx_col 1 /--wrx_col 0 /' <<< "$C"; }
D=$W/ours0/${SC}_K1; DT=$W/ours0/${SC}_K1_t; mkdir -p $D $DT
mk $D "" 0 > $D/cmd.txt; mk $DT "--pdiag --qjd --fhash --hgt --bar_cnt" 1 > $DT/cmd.txt
grep -q "ssfix_wrap.py --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1" $D/cmd.txt && grep -q -- "--ue_loc 51 " $D/cmd.txt && grep -q -- "--wrx_col 0 " $D/cmd.txt && grep -q -- "--mlow 1e-3" $D/cmd.txt || { say "[명령 확인 실패]"; exit 1; }
pkill -STOP -f "$GPAT"; say "[정답 멈춤 (SIGSTOP)] 우리 판 단독"
t1=$(date +%s); CUDA_VISIBLE_DEVICES=0 timeout 3600 bash -c "$(cat $DT/cmd.txt)" < /dev/null > $DT/run.log 2>&1; rcT=$?
t2=$(date +%s); CUDA_VISIBLE_DEVICES=0 timeout 3600 bash -c "$(cat $D/cmd.txt)" < /dev/null > $D/run.log 2>&1; rc=$?
pkill -CONT -f "$GPAT"; say "[정답 다시 (SIGCONT)]"
read MS RD FD DIV < <(python -c "
import json,re
t=json.load(open('$DT/summary.json')); s=json.load(open('$D/summary.json'))
m=[l for l in open('$DT/run.log',errors='ignore') if re.search(r'substep_fix\] 프레임 \d+ 다시 풂', l)]
r=re.search(r'다시 풂 (\d+)', m[-1]).group(1) if m else '0'
print('%.1f'%t['ms_mean'], r, s.get('frames_done'), s.get('diverged'))")
say "[우리 먼저] 단독 시간 판 rc=$rcT $(( t2 - t1 )) 초, 덤프 판 rc=$rc 완주 $FD/$NF 발산 $DIV, A6000 단독 $MS ms, 다시 푼 프레임 $RD (단독 판), $(grep -a 'substep_fix\] 프레임 [0-9]* 다시' $DT/run.log | tail -1)"
UP=$(python -c "
import json,math,numpy as np
c=json.load(open('$CFG')); R=np.eye(3)
for dg,ax in zip(c['rotation_degree'],c['rotation_axis']):
    cs,sn=math.cos(dg/180*3.1415926),math.sin(dg/180*3.1415926)
    R=np.array({0:[[1,0,0],[0,cs,-sn],[0,sn,cs]],1:[[cs,0,sn],[0,1,0],[-sn,0,cs]],2:[[cs,-sn,0],[sn,cs,0],[0,0,1]]}[ax])@R
print(','.join('%.6f'%q for q in R[2]))")
python -u exe/lb_roll_to_h5.py --dump $D/pred.pt --traj $AG/gt0_$SC --out $D/h5_ours > $D/h5.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim $D/h5_ours --config $CFG --shape $SC --out $D/ours_raw.mp4 --cam_back 1.6 --elev 40 --up $UP --back2 1.1 --png $D/ours_png < /dev/null > $D/r_ours.log 2>&1
python - $D/ours_png $D/ours_lab.mp4 $W/font.otf "우리 K=1  ${MS} ms (A6000 단독), 다시 푼 프레임 $RD" <<'PY'
import glob, sys, imageio, numpy as np
from PIL import Image, ImageDraw, ImageFont
fs = sorted(glob.glob(sys.argv[1] + "/*.png")); fnt = ImageFont.truetype(sys.argv[3], 28)
Wr = imageio.get_writer(sys.argv[2], fps=60, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2)
for f in fs:
    im = Image.open(f).convert("RGB"); ImageDraw.Draw(im).text((12, 8), sys.argv[4], fill=(0, 0, 0), font=fnt); Wr.append_data(np.array(im))
Wr.close(); print(len(fs))
PY
V=$D/${OBJ}_plastic_K1_ours.mp4
ffmpeg -y -loglevel error -i $D/ours_lab.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $V
say "[우리 한 칸 영상] $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $V) 디코드 오류 $(ffmpeg -v error -i $V -f null - 2>&1 | wc -l)"
rclone copyto $V $VD/${OBJ}_plastic_K1_ours.mp4 -q; rclone copy $D $RX/ours0/${SC}_K1 -q --include "*.json" --include "*.log" --include "cmd.txt"; rclone copy $DT $RX/ours0/${SC}_K1_t -q --include "*.json" --include "*.log" --include "cmd.txt"
echo "ok $MS $RD" | rclone rcat $RX/$TAG.OURS0
wait $PC; n=$(ls $OC/simulation_ply | grep -c h5); say "[정답 사본] 프레임 $n $(( $(date +%s) - t0 )) 초 (멈춘 시간 포함)"
[ $n -ge $((NF + 1)) ] || { say "[정답 실패]"; exit 1; }
echo "i-physgaussian 사본 gs_simulation.py (원본은 입자 채우기가 끝나지 않아 안 씀)" > $OC/RUNNER; cp $CFG $OC/config.json
rclone copy $OC $DS/gt_pg/$SC --transfers 16 -q; say "[정답] R2 올림 -> 두 칸 (vast_elastic_new.sh)"
bash $AF/exe/vast_elastic_new.sh
