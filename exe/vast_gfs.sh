#!/bin/bash
# 절 CM182 GF 공개 장면 (watermelon 제외): vast.ai 인스턴스 onstart 진입점. 정답 궤적 (GF 공식 러너·공식 config 그대로) + R2 업로드.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT, XS_TAG (인스턴스 이름), XS_HF (1 이면 HF -> R2 datasets/anchorflow/gf_scenes 업로드),
#        XS_SCENES (쉼표: 이름:모델:설정[:러너]), XS_STAGE (gt = 정답 궤적).  표식·로그 = R2 lbours/gfscenes/<XS_TAG>.*
#   GF 공식 러너 = gs_simulation.py.orig (저장소 원본). 채우기 입자 수와 껍질 마스크가 안 맞아 터지면 (IndexError) 그때만 R2 사본 gs_simulation.py (rind_beta 없을 때 껍질 beta 블록 끔) 로 다시.
set -u
TAG=${XS_TAG:-gfs}; STG=${XS_STAGE:-gt}
R=r2:storage/result/anchorflow; RX=$R/lbours/gfscenes; DS=r2:storage/datasets/anchorflow/gf_scenes
mkdir -p /root/gfs /root/sv; LOG=/root/sv/$TAG.log
say() { echo "[$TAG $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/$TAG.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?} 단계 $STG"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/$TAG.log -q
df -h /root /dev/shm | tee -a $LOG
export HOME=/root
if ! python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization" 2>/dev/null; then      # 표식 파일 대신 실제 import 로 판단 (옛 표식이 거짓일 수 있음)
  for try in 1 2 3; do
    pip install -q --default-timeout=120 "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib huggingface_hub lpips 2>&1 | tail -2 | tee -a $LOG
    pip install -q --default-timeout=120 --only-binary=:all: PyMCubes 2>&1 | tail -1 | tee -a $LOG
    cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
      curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
    pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
    python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization; assert torch.cuda.is_available(); print('[pip] torch', torch.__version__, torch.cuda.is_available())" >> $LOG 2>&1 && { touch /root/sv/pip.done; break; }
    say "[pip] 시도 $try 실패, 다시"
  done
fi
python -c "import torch, cupy, h5py, warp, sklearn, lpips, diff_gaussian_rasterization" 2>/dev/null || { say "[세팅 실패] pip"; echo "pip 실패" | rclone rcat $RX/$TAG.END; exit 1; }
# GF 코드 (R2 사본, 모델 제외) + 코드 묶음 (측정 스크립트)
W=/root/gfs; mkdir -p $W/hf
rclone copy $R/_vastmirror/GaussianFluent $W/GaussianFluent --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q
CODE=${XS_CODE:-$RX/code}      # 절 CM184: 코드 묶음 위치 (기본 = 공유 code/, 다른 에이전트 인스턴스와 겹치지 않게 따로 둘 수 있음)
AF=/root/sv/af; mkdir -p $AF; [ -d $AF/exe ] || rclone cat $CODE/af_code.tar | tar -x -C $AF; rclone copy $CODE/exe $AF/exe -q
python $AF/exe/patch_gf_bg.py $W/GaussianFluent/gs_simulation.py.orig $W/GaussianFluent/gs_orig_run.py; python $AF/exe/patch_gf_bg.py $W/GaussianFluent/gs_simulation.py $W/GaussianFluent/gs_patched_run.py
pip install -q "urllib3<2.6" "brotli<1.2" 2>&1 | tail -1 | tee -a $LOG      # HF 받기 TypeError: process() takes no keyword arguments (urllib3 2.6 + brotli)
# ---- HF -> R2 (모델·설정 통째, 로컬에는 안 받음; 이 인스턴스 디스크에만)
HFD="python $AF/exe/hfdl_std.py $W/hf"
if [ "${XS_HF:-0}" = 2 ]; then      # 이 인스턴스 장면만 HF 에서 직접 받아 R2 로
  PF=$(for e in $(echo ${XS_SCENES:-} | tr ',' ' '); do IFS=: read n m c r <<< "$e"; echo "model/$m/ config/$c"; done)
  $HFD $PF 2>&1 | tail -5 | tee -a $LOG
  rclone copy $W/hf $DS --transfers 16 -q --exclude ".cache/**"; say "[HF 일부 -> R2] $(du -sh $W/hf | cut -f1)"
elif [ "${XS_HF:-0}" = 1 ]; then
  $HFD config/ model/ --skip model/watermelon/ model/garden/ model/garden_ours/ 2>&1 | tail -5 | tee -a $LOG
  say "[HF] $(du -sh $W/hf | cut -f1) $(ls $W/hf/model | tr '\n' ' ')"
  rclone copy $W/hf $DS --transfers 16 -q --exclude ".cache/**"; say "[HF -> R2] datasets/anchorflow/gf_scenes 올림: $(rclone size $DS --json 2>/dev/null)"
else
  rclone copy $DS/model $W/hf/model --transfers 16 -q $(for m in $(echo ${XS_SCENES:-} | tr ',' '\n' | cut -d: -f2); do echo "--include /$m/**"; done)
  rclone copy $DS/config $W/hf/config -q
fi
# ---- 정답 궤적: 공식 러너 그대로
[ "$STG" = gt ] && [ ! -s $W/hf/config/$(echo ${XS_SCENES:-} | cut -d, -f1 | cut -d: -f3) ] && { say "[HF 실패] 설정 파일 없음"; echo "HF 실패" | rclone rcat $RX/$TAG.END; exit 1; }
gt1() { local name=$1 model=$2 cfg=$3 runner=${4:-gs_orig_run.py}; local O=$W/out/$name; mkdir -p $O
  cd $W/GaussianFluent; export PYTHONPATH=$W/GaussianFluent:$W/GaussianFluent/gaussian-splatting
  local t0=$(date +%s)
  timeout 5400 python -u $runner --model_path $W/hf/model/$model --config $W/hf/config/$cfg --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1; local rc=$?
  if [ $rc != 0 ] && grep -aq "IndexError" $O/run.log && [ $runner = gs_orig_run.py ]; then
    cp $O/run.log $O/run_orig_fail.log; rm -rf $O/simulation_ply
    say "  $name 원본 러너 IndexError -> 껍질 beta 블록 끈 사본으로"
    timeout 5400 python -u gs_patched_run.py --model_path $W/hf/model/$model --config $W/hf/config/$cfg --output_path $O --output_h5 < /dev/null > $O/run.log 2>&1; rc=$?; echo patched > $O/RUNNER
  else echo $runner > $O/RUNNER; fi
  local n=$(ls $O/simulation_ply 2>/dev/null | grep -c h5)
  say "  $name rc=$rc 프레임 파일 $n $(( $(date +%s) - t0 )) 초 $(grep -a 'rror' $O/run.log | tail -1 | cut -c1-200)"
  cp $W/hf/config/$cfg $O/config.json
  if [ $n -gt 1 ]; then
    cd $AF; python -u exe/measure_gf_tearing.py --h5_dir $O/simulation_ply --tag $name --out $O/tear > $O/tear.log 2>&1; say "  $name 찢어짐: $(tail -4 $O/tear.log | tr '\n' ' ' | cut -c1-400)"
  fi
  rclone copy $O $DS/gt/$name --transfers 16 -q; say "  $name R2 올림 (datasets/anchorflow/gf_scenes/gt/$name)"
  true; }
if [ "$STG" = gt ]; then
  PAR=${XS_PAR:-6}     # 사용자 지시 11-15: 장면은 차례가 아니라 동시에 (한 카드에 PAR 개, GF 시뮬은 카드 메모리 몇 GB)
  for e in $(echo ${XS_SCENES:-} | tr ',' ' '); do IFS=: read n m c r <<< "$e"
    case " cookie oreo bowl fruitninja milk2 " in *" $n "*) say "[건너뜀] $n (사용자 11-15: 논문에 없는 장면)"; continue;; esac
    NF=$(python -c "import json;print(json.load(open('$W/hf/config/$c'))['frame_num'] + 1)"); HAVE=$(rclone lsf $DS/gt/$n/simulation_ply 2>/dev/null | grep -c "h5$")
    [ "$HAVE" -ge "$NF" ] && { say "[건너뜀] $n 이미 끝남 (R2 프레임 $HAVE/$NF)"; continue; }
    while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 3; done
    say "[정답] $n"; ( gt1 $n $m $c ${r:-gs_orig_run.py} ) & sleep 2; done
  wait
fi
if [ "$STG" = render ]; then      # 정답 3DGS 영상: GF 공식 렌더 경로 (--render_img --compile_video), 배경 가우시안만 뺌, 같은 config 로 시뮬을 다시 돌며 그림
  python $AF/exe/patch_gf_bg.py $W/GaussianFluent/gs_simulation.py.orig $W/GaussianFluent/gs_nobg_run.py --nobg
  for e in $(echo ${XS_SCENES:-} | tr ',' ' '); do IFS=: read n m c r <<< "$e"; O=$W/render/$n; mkdir -p $O
    ( cd $W/GaussianFluent; export PYTHONPATH=$W/GaussianFluent:$W/GaussianFluent/gaussian-splatting
      python -u gs_nobg_run.py --model_path $W/hf/model/$m --config $W/hf/config/$c --output_path $O --output_h5 --render_img --compile_video < /dev/null > $O/run.log 2>&1 )
    ls $O/*.png 2>/dev/null | wc -l | xargs -I{} echo "  $n 렌더 프레임 {}" | tee -a $LOG
    FR=$(python -c "import json;print(max(2,round(1/json.load(open('$W/hf/config/$c'))['frame_dt']/1.4)))")
    ffmpeg -y -loglevel error -framerate $FR -i $O/%04d.png -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $O/gf_${n}_gt.mp4
    ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames,duration -of csv=p=0 $O/gf_${n}_gt.mp4 | tee -a $LOG
    rclone copyto $O/gf_${n}_gt.mp4 $R/v20261009/gf_${n}_gt.mp4 -q; say "  $n 영상 올림 $R/v20261009/gf_${n}_gt.mp4 $(grep -a 'rror' $O/run.log | tail -1 | cut -c1-200)"
    rclone copy $O $RX/render/$n --include "*.log" --include "0000.png" -q
  done
fi
if [ "$STG" = jb ]; then      # 젤리 + 총알 (GF 젤리 러너 gs_simulation_jellynacc.py): 총알 합치기 = 러너가 읽는 save_bullet_prop.pt (미공개) -> 러너 안 주석 블록 그대로 되살려 생성 + 검산
  GF=$W/GaussianFluent; O=$W/jb; mkdir -p $O; D=$DS/gt/jellybullet
  for m in gen run nobg; do python $AF/exe/patch_gf_jelly.py $GF/gs_simulation/jelly/gs_simulation_jellynacc.py $GF/gs_jb_$m.py --mode $m --dbg --dropbig; done
  cd $GF; export PYTHONPATH=$GF:$GF/gaussian-splatting; CFG=$W/hf/config/jelly_config_nacc.json
  if rclone lsf $D/ | grep -q "^simulation_ply/" && rclone lsf $D/ | grep -q save_bullet_prop.pt; then say "  [젤리+총알] R2 에 총알 파일·정답 이미 있음 -> 렌더만"; rclone copyto $D/save_bullet_prop.pt save_bullet_prop.pt -q; else
  python -u gs_jb_gen.py --model_path $W/hf/model/bullet_0_psnr36 --config $CFG --output_path $O/gen < /dev/null > $O/gen.log 2>&1; say "  총알 자산 생성 rc=$? $(grep -a '총알' $O/gen.log | tr '\n' ' ' | cut -c1-500)"
  [ -s save_bullet_prop.pt ] || { say "[젤리+총알 실패] save_bullet_prop.pt 없음 $(tail -2 $O/gen.log | cut -c1-200)"; rclone copy $O $D -q; echo "jb 실패" | rclone rcat $RX/$TAG.END; exit 1; }
  rclone copy save_bullet_prop.pt $D/ -q; cp $CFG $O/config.json
  t0=$(date +%s); timeout 7000 python -u gs_jb_run.py --model_path $W/hf/model/jelly --config $CFG --output_path $O/sim --output_h5 < /dev/null > $O/sim.log 2>&1; rc=$?
  say "  젤리+총알 정답 rc=$rc 프레임 $(ls $O/sim/simulation_ply 2>/dev/null | grep -c h5) $(( $(date +%s) - t0 )) 초 $(grep -a 'rror' $O/sim.log | tail -1 | cut -c1-200)"
  cd $AF; python -u exe/measure_gf_tearing.py --h5_dir $O/sim/simulation_ply --tag jellybullet --out $O/tear > $O/tear.log 2>&1; say "  젤리+총알 찢어짐: $(tail -4 $O/tear.log | tr '\n' ' ' | cut -c1-400)"
  rclone copy $O/sim $D/ --transfers 16 -q; rclone copy $O/tear $D/tear -q; rclone copy $O/gen.log $D/ -q; rclone copy $O/sim.log $D/ -q; say "  젤리+총알 R2 올림 $D"
  fi
  cd $GF; timeout 7000 python -u gs_jb_nobg.py --model_path $W/hf/model/jelly --config $CFG --output_path $O/rend --render_img --compile_video < /dev/null > $O/rend.log 2>&1; say "  젤리+총알 렌더 rc=$? 프레임 $(ls $O/rend/*.png 2>/dev/null | wc -l) $(grep -a '\[dbg\]' $O/rend.log | head -12 | cut -c1-300 | tr '\n' ' ') $(grep -a 'rror' $O/rend.log | tail -1 | cut -c1-200)"
  ffmpeg -y -loglevel error -framerate 28 -i $O/rend/%04d.png -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $O/gf_jellybullet_gt.mp4
  ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $O/gf_jellybullet_gt.mp4 | tee -a $LOG
  rclone copyto $O/gf_jellybullet_gt.mp4 $R/v20261009/gf_jellybullet_gt.mp4 -q; rclone copy $O/rend.log $D/ -q; say "  젤리+총알 영상 $R/v20261009/gf_jellybullet_gt.mp4"
fi
if [ "$STG" = ours ]; then      # 우리 판 + 지표 + 영상 (exe/gfs_ours.sh). XS_SCENES 항목 = 이름:모델:설정:프레임수:바닥(plane|cfg)
  export W AF DS R RX LOG TAG; export GF=$W/GaussianFluent; export -f say; mkdir -p $W/ag
  [ -n "${XS_DROP:-}" ] && export GFS_DROP="${XS_DROP//+/ }"; [ -n "${XS_CAUTO:-}" ] && export GFS_CAUTO="$XS_CAUTO"      # 절 CM185: 단독 시간 판 = 진단 플래그 뺌 (절 CM184 cm184x 와 같은 목록), 칸 상한
  say "[설정] GFS_DROP='${GFS_DROP:-}' GFS_CAUTO=${GFS_CAUTO:-기본 40000}"
  rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q
  if [[ ",${XS_SCENES:-}," == *",watermelon:"* ]]; then      # 절 CM185: 수박 = 정답 R2 _vastmirror/wm2/gf (GF 공식 러너, 절 CM9 이후 같은 정답), 모델 _vastmirror/gfmodel/model/watermelon -> gfs 배치로
    MI=$R/_vastmirror; mkdir -p $W/gt/watermelon $W/hf/model $W/hf/config
    rclone copy $MI/wm2/gf/simulation_ply $W/gt/watermelon/simulation_ply --transfers 16 -q; rclone copy $MI/gfmodel/model/watermelon $W/hf/model/watermelon --transfers 16 -q
    rclone copyto $MI/wm2/gf/config.json $W/hf/config/watermelon_wm2.json -q; say "  [수박 준비] 정답 h5 $(ls $W/gt/watermelon/simulation_ply | grep -c h5) 모델 $(ls $W/hf/model/watermelon | tr '\n' ' ')"
  fi
  for e in $(echo ${XS_SCENES:-} | tr ',' ' '); do IFS=: read n m c nf fl <<< "$e"; say "[우리 판] $n"; if [ "$n" = jellybullet ]; then bash $AF/exe/gfs_jb_ours.sh "$fl"; elif [[ "$fl" == *SWEEP* ]]; then      # 사용자 11-15: K 를 2, 4, 8 순서로 올려 처음 완주하는 값 (한 인스턴스에서 차례로; 발산하면 사후 처리 없이 다음 K)
    fl0=${fl//SWEEP+/}; fl0=${fl0//SWEEP/}
    for K in 2 4 8; do last=""; [ $K = 8 ] && last=1
      XS_SWEEP=1 XS_SWEEP_LAST=$last XS_SFX=_K$K bash $AF/exe/gfs_ours.sh $n $m $c $nf "${fl0}+--K=$K"; rc=$?; say "  [$n] K=$K rc=$rc"; [ $rc != 3 ] && break; done
  else bash $AF/exe/gfs_ours.sh $n $m $c $nf "$fl"; fi; done
fi
if [ "$STG" = prep ]; then      # 절 CM184: 세팅만 (모델·설정·코드), 실험은 ssh 로
  for e in $(echo ${XS_SCENES:-} | tr ',' ' '); do IFS=: read n m c nf fl <<< "$e"; rclone copy $DS/gt/$n/simulation_ply $W/gt/$n/simulation_ply --transfers 16 -q; say "  [prep] $n 정답 h5 $(ls $W/gt/$n/simulation_ply | grep -c h5)"; done
fi
if [ "$STG" = jr ]; then      # 젤리+총알 렌더만 다시 (R2 의 save_bullet_prop.pt 사용), 입력 통계 출력
  GF=$W/GaussianFluent; O=$W/jb; mkdir -p $O; D=$DS/gt/jellybullet; CFG=$W/hf/config/jelly_config_nacc.json
  python $AF/exe/patch_gf_jelly.py $GF/gs_simulation/jelly/gs_simulation_jellynacc.py $GF/gs_jb_nobg.py --mode nobg --dbg --dropbig
  cd $GF; export PYTHONPATH=$GF:$GF/gaussian-splatting; rclone copyto $D/save_bullet_prop.pt save_bullet_prop.pt -q
  timeout 7000 python -u gs_jb_nobg.py --model_path $W/hf/model/jelly --config $CFG --output_path $O/rend --render_img --compile_video < /dev/null > $O/rend.log 2>&1; say "  젤리+총알 렌더 rc=$? 프레임 $(ls $O/rend/*.png 2>/dev/null | wc -l) $(grep -a '\[dbg\]' $O/rend.log | head -12 | cut -c1-300 | tr '\n' ' ') $(grep -a 'rror' $O/rend.log | tail -1 | cut -c1-200)"
  rclone copy $O/rend.log $D/ -q
  if [ $(ls $O/rend/*.png 2>/dev/null | wc -l) -gt 10 ]; then
    ffmpeg -y -loglevel error -framerate 28 -i $O/rend/%04d.png -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $O/gf_jellybullet_gt.mp4
    ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $O/gf_jellybullet_gt.mp4 | tee -a $LOG
    rclone copyto $O/gf_jellybullet_gt.mp4 $R/v20261009/gf_jellybullet_gt.mp4 -q; say "  젤리+총알 영상 올림 $R/v20261009/gf_jellybullet_gt.mp4"
  fi
fi
say "[끝]"; echo "끝 $(date)" | rclone rcat $RX/$TAG.END
