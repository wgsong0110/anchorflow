#!/bin/bash
# 파레토 Spring-Gaus pillow2sofa·bread 반복 측정 (맨 뒤): 학습은 다시 안 함 -- 학습된 판 (R2 final_pa/bl/P2/p/springgaus_pillow2sofa_prep, final_pa/blx/SGB/p/springgaus_bread_prep) 으로
#   롤아웃·단독 시간만 RTX A6000 에서 3 회씩 (1 회차가 RTX 6000 Ada 였으므로 A6000 3 회로 바꿈). 명령 = 1 회차와 같음:
#   pillow2sofa = exe/sg_post_p2s.sh 의 stab_sg.py --mode run (코드 fpabl_cf55da5), bread = exe/blx_meas_bread.sh·blx_time_sgb.sh 의 같은 명령 (코드 blx_f222d061).
#   카드 0 에 다른 계산 프로세스가 없을 때 시작, 실행 중 5 초마다 기록. 결과 R2 lbours/final_pa/sgrep/<장면>/springgaus_<장면>_n<N>_r<k>/{summary.json,res.json,card.txt,log.txt}.
#   끝나면 R2 대조 뒤 스스로 destroy.  env: REP_TAG CONTAINER_ID CONTAINER_API_KEY
set -u
export HOME=/root
RA=r2:storage/result/anchorflow; RB=$RA/lbours/final_pa/bl; OX=$RA/lbours/final_pa/blx; RS=$RA/lbours/final_pa/sgrep; RX=$RA/lbours/paper_remeasure2; MI=$RA/_vastmirror
T=${REP_TAG:-sgrep}; L=/root/sgrep.log; say() { echo "[sgrep $(date -u '+%F %T')] $*" >> $L; rclone copyto $L $RS/marks/$T.log -q; }
say "부팅 $(hostname) $(nvidia-smi --query-gpu=name,power.limit,power.default_limit,memory.total --format=csv,noheader)"
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ "$PL" -lt "$PD" ]; then say "[전력 제한] $PL < $PD -- 일 안 함"; echo "$PL<$PD" | rclone rcat $RS/marks/$T.POWERCAP -q; exit 1; fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 unzip) >/dev/null 2>&1
mkdir -p /workspace/shm /home/dkta/.conda/envs/af/bin /home/dkta/bin /home/dkta/work/newsc /dev/shm
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python; ln -sf $(command -v rclone) /home/dkta/bin/rclone
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work; NW=$W/newsc; AG=/dev/shm/dkta/ag_data; O=/dev/shm/dkta/lbours/paper_remeasure2/stab; mkdir -p $AG $O/p
( [ -s $NW/pillow2sofa.pt ] || rclone copyto $RX/scenes/pillow2sofa/pillow2sofa.pt $NW/pillow2sofa.pt -q --multi-thread-streams 16 ) &
( [ -s $AG/bread.pt ] || rclone copyto $RA/lbours/fem_ag_data/bread.pt $AG/bread.pt -q ) &
( [ -d $W/i-physgaussian ] || rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q ) &
( cd /dev/shm/dkta && [ -d pylib/sg ] || rclone cat $RB/code/sg_env_cf55da5.tar | tar -x ) &
( F=$O/p/springgaus_pillow2sofa_prep; rclone copy $RB/P2/p/springgaus_pillow2sofa_prep $F -q --transfers 16 --exclude "mv/**" --exclude "sg_resume/**" --exclude "exp/**"
  [ -s $F/pillow2sofa.yaml ] || (cd $O/p && rclone cat $RB/code/sg_fit_p2s.tar | tar -x)
  EL=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['exp'])"); rclone copy $RB/P2/p/springgaus_pillow2sofa_prep/exp/$(basename $EL) $EL -q --exclude "evaluations/**" --exclude "dynamic_refine/**" ) &
( F=$O/p/springgaus_bread_prep; rclone copy $OX/SGB/p/springgaus_bread_prep $F -q --transfers 16 --exclude "mv/**" --exclude "sg_resume/**" --exclude "exp/**"
  EL=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['exp'])"); rclone copy $OX/SGB/p/springgaus_bread_prep/exp/$(basename $EL) $EL -q --exclude "evaluations/**" --exclude "dynamic_refine/**" ) &
mkdir -p /root/c_p2s /root/c_bread; rclone cat $RB/code/fpabl_cf55da5.tar | tar -x -C /root/c_p2s; rclone cat $OX/code/blx_f222d061.tar | tar -x -C /root/c_bread
if [ ! -f /root/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" h5py scikit-image lpips tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib open3d termcolor vastai 2>&1 | tail -1 >> $L
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl >> $L 2>&1
  python -c "import torch, h5py, lpips, diff_gaussian_rasterization, simple_knn; print('pip ok', torch.__version__)" >> $L 2>&1 && touch /root/pip.done
fi
wait
PPX=""; python -c "import open3d" 2>/dev/null || { mkdir -p /root/o3dstub/open3d; printf 'def __getattr__(n):\n    raise RuntimeError("open3d stub: " + n)\n' > /root/o3dstub/open3d/__init__.py; PPX=/root/o3dstub:; }
say "환경 끝 pip $(ls /root/pip.done 2>&1) p2s $(du -sh $NW/pillow2sofa.pt | cut -f1) bread $(du -sh $AG/bread.pt | cut -f1) 학습 판 $(ls $O/p/springgaus_pillow2sofa_prep/exp $O/p/springgaus_bread_prep/exp | tr '\n' ' ')"
export ST_SG_PYPATH=/dev/shm/dkta/pylib/sg ST_SG_REPO=/dev/shm/dkta/ext/Spring-Gaus
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$"; }
one() {   # <장면> <코드> <반복> [추가 env]
  local s=$1 C=$2 k=$3 F=$O/p/springgaus_$1_prep N DA
  N=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['n_step'])")
  [ $s = pillow2sofa ] && DA="--data $NW/pillow2sofa.pt --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp" || DA="--data $AG/bread.pt --tag bread --frames 300 --floor none --cfg_bc --pg_vdamp"
  local G=/workspace/sgrep/$s/springgaus_${s}_n${N}_r$k; mkdir -p $G
  [ -s $G/res.json ] && return 0
  while [ -n "$(others)" ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others | wc -l) 개 $(date -u '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader)" > $G/card.txt
  ( cd $C; [ $s = bread ] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PPX}/dev/shm/dkta/pylib/sg python -u exe/stab_sg.py --mode run --sg /dev/shm/dkta/ext/Spring-Gaus --fit $F --N $N --out $G $DA > $G/log.txt 2>&1 ) &
  local P=$! seen=""
  while kill -0 $P 2>/dev/null; do seen="$seen $(others)"; sleep 5; done; wait $P; local rc=$?
  echo "rc=$rc 끝 $(date -u '+%F %T') 실행 중 카드 0 계산 프로세스 (자기 포함): $(echo $seen | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')" >> $G/card.txt
  python - $G $N $k "$(echo $seen | tr ' ' '\n' | sort -u | grep -v '^$' | wc -l)" <<'P'
import json, sys
G, N, k, nf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
s = json.load(open(f"{G}/summary.json")); dv = s.get("diverged")
json.dump(dict(stable=not dv, err=(100 * s["err_mean"] if not dv else None), div=dv, ms=s.get("ms_mean"), ms_max=s.get("ms_max"), N=N, rep=k, others=max(nf - 1, 0)), open(f"{G}/res.json", "w"), indent=1)
P
  rclone copy $G $RS/$s/$(basename $G) -q
  say "$s r$k rc=$rc $(tr -d '\n ' < $G/res.json 2>/dev/null)"
}
for k in 1 2 3; do one pillow2sofa /root/c_p2s $k; done
for k in 1 2 3; do one bread /root/c_bread $k; done
if rclone check /workspace/sgrep $RS --one-way --size-only -q --exclude "marks/**"; then say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $RS/marks/$T.ALL_DONE
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} >> $L 2>&1
else say "[R2 확인 차이 있음] destroy 안 함"; echo "$(date)" | rclone rcat $RS/marks/$T.CHECK_FAIL; fi
