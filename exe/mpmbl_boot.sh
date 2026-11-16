#!/bin/bash
# [final_fix_mpmbl] vast onstart: 파레토 MPM 베이스라인 (i-PG 또는 OmniPhysGS) 한 방법 x 네 장면 (레고 L000·bread·pillow2sofa·wolf), 카드 하나에 직렬 (시간 = 단독).
#   장면마다: 판 a (시간 + 덤프 pred.pt) -> 발산하면 공식 범위 안에서 N 을 키워 다시 -> 판 b·c (시간) -> lb_roll_to_h5 + lb_render (정답 MPM 렌더와 같은 카메라) -> ffx_phys -> ffx_vis (--ref = R2 phys/mpm_<장면>/png)
#   코드 = fpabl_<FB>.tar (final_fix_vrgs) + ffxphys_<FX>.tar (final_fix_phys) + mpmbl_<MB_COMMIT>.tar (이 가지; exe/pareto_mpmbl.py).
#   결과 R2 lbours/final_fix/bl_mpm/<방법>_<장면>/, 표식 lbours/final_fix/bl_mpm/marks/<MB_TAG>.*
#   env: R2_* MB_COMMIT MB_METHOD (ipg|omni) MB_TAG [MB_SCENES 기본 "L000 bread pillow2sofa wolf"] [FB_COMMIT 0f5a510] [FX_COMMIT 7cc0289] CONTAINER_ID CONTAINER_API_KEY
set -u
FB=${FB_COMMIT:-0f5a510}; FXC=${FX_COMMIT:-7cc0289}; C=${MB_COMMIT:?}; M=${MB_METHOD:?}; TAG=${MB_TAG:?}; SCN=${MB_SCENES:-"L000 bread pillow2sofa wolf"}
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; OX=$R/lbours/final_pa/bl; FX=$R/lbours/final_fix; RB=$FX/bl_mpm; MK=$RB/marks/$TAG
mkdir -p /workspace/shm /workspace/tmpd /root/mb
LOG=/root/mb/vast.log
say() { echo "[mb $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK.log -q 2>/dev/null; }
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
say "[시작] $M 코드 $C ($FB + $FXC) $(hostname) 인스턴스 ${CONTAINER_ID:-?} 장면 $SCN"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG; free -g | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < $PD $(date)" | rclone rcat $MK.POWERCAP; say "[전력 제한] 일 안 함"; exit 1; fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; MI=$R/_vastmirror; mkdir -p $SD $AG $NW
# 자료 (pip 와 동시)
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --exclude ".git/**" --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q --include "lego_whitebg-trained/**" --include "bread-trained/**" --include "wolf_whitebg-trained/**" &
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q --include "config.json" --include "test_low/L000/**" &
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q & rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q & rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
[ -d $W/OmniPhysGS/src ] || (git clone -q https://github.com/wgsxm/OmniPhysGS $W/OmniPhysGS && git -C $W/OmniPhysGS checkout -q 5ab915b014467b911581b0c6df351713d3114a08) &
for s in $SCN; do rclone copy $FX/phys/mpm_$s/png /root/mb/ref/$s -q --transfers 16 & done
D=/root/mb/code; mkdir -p $D
rclone cat $OX/code/fpabl_$FB.tar | tar -x -C $D; rclone cat $FX/code/ffxphys_$FXC.tar | tar -x -C $D; rclone cat $RB/code/mpmbl_$C.tar | tar -x -C $D
if [ ! -f /root/mb/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai omegaconf einops 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") 2>&1 | tail -1 | tee -a $LOG
  python -c "import torch, warp, cupy, h5py, lpips, pyiqa, omegaconf, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1 | tee -a $LOG && touch /root/mb/pip.done
fi
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mb XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0
mkdir -p $CUPY_CACHE_DIR
python exe/ffx_phys_patch_render.py | tee -a $LOG
case " $SCN " in *" bread "*) [ -s $AG/gt_bread_full/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/mb/gtbread_unused.pt --out_dir $AG/gt_bread_full > /root/mb/gt_bread.log 2>&1;; esac
case " $SCN " in *" wolf "*) [ -s $AG/gt_wolf/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/mb/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/mb/gt_wolf.log 2>&1;; esac
case " $SCN " in *" pillow2sofa "*) PR_SC=pillow2sofa RX=$RX; source exe/pr2_scene_fetch.sh;; esac
say "[자료] $(for s in $SCN; do echo -n "$s ref $(ls /root/mb/ref/$s | wc -l) "; done) pgmodel $(ls $W/pgmodel | tr '\n' ' ') gt $(ls $AG | tr '\n' ' ')"
echo "[ok] $(date)" | rclone rcat $MK.setup.done
X=/workspace/mb; mkdir -p $X
( while sleep 180; do rclone copy $X $RB -q --transfers 8 --exclude "*/h5/**"; rclone copyto $LOG $MK.log -q; done ) & SY=$!

scene() {   # -> SIM (pareto_mpmbl 입력 인자) DATA GTD CF SHAPE FR FLOOR FUSED NOFF (공식 N) NPG (PG 공식 서브스텝)
  case $1 in
    L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel; FUSED=1
          SIM="--gt $GTD --config $CF --pt $DATA --frames 80";;
    pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none; FUSED=1
          SIM="--gt $NW/pillow2sofa_h5/simulation_ply --config $NW/pillow2sofa_config.json --pt $DATA --frames 300 --pg_cfg";;
    bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300; FLOOR=none; FUSED=0
          SIM="--gt $W/topo/pg_bread_plastic/simulation_ply --config $W/topo/bread_plastic_config.json --pt $DATA --frames 300 --pg_cfg";;
    wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel; FUSED=0
          SIM="--gt $W/topo/pg_wolf_sand/simulation_ply --config $W/PhysGaussian/config/wolf_config.json --pt $DATA --frames 50 --pg_cfg";;
  esac
  local cfgp=$(echo $SIM | grep -oP -- '--config \K\S+')
  # i-PG 공식 (gs_simulation.py --implicit --dt_multiplier k): 서브스텝 = max(1, round(int(frame_dt/substep_dt) / k)), 논문 최대 k = 20
  # OmniPhysGS 공식 dt = 3e-4 s (configs/*.yaml sim.dt) -> 서브스텝 = ceil(frame_dt / 3e-4)
  NPG=$(python -c "import json;c=json.load(open('$cfgp'));print(int(c['frame_dt']/c['substep_dt']))")
  if [ $M = ipg ]; then NOFF=$(python -c "print(max(1, round($NPG / 20)))"); else NOFF=$(python -c "import json,math;c=json.load(open('$cfgp'));print(math.ceil(c['frame_dt']/3e-4 - 1e-9))"); fi
}

matcheck() {   # [omni] 정답 경로 (pareto_mpm --pg_cfg) 가 실제로 쓴 재질 값: PG 명시 MPM 으로 6 프레임, 재질 키 안 넘김 vs 넘김 -> 정답과 더 가까운 쪽
  local s=$1 O=$X/matcheck_$s; mkdir -p $O
  for v in base extra; do
    [ -s $O/$v/err.json ] && continue
    local EXF=""; [ $v = extra ] && EXF="--mat_extra yield_stress,friction_angle"
    python -u exe/pareto_mpmbl.py --method pg --nsub $NPG $(echo $SIM | sed -E "s/--frames [0-9]+/--frames 6/") $EXF --out $O/$v > $O/$v.log 2>&1
  done
  python - $O <<'PY'
import json, sys
O = sys.argv[1]; r = {v: json.load(open(f"{O}/{v}/err.json")) for v in ("base", "extra")}
e = {v: r[v]["err_mean"] for v in r}; best = min(e, key=lambda k: e[k] if e[k] == e[k] else 9e9)
json.dump(dict(err=e, best=best), open(f"{O}/choice.json", "w"))
print(best, e)
PY
}

for s in $SCN; do
  scene $s; J=${M}_$s; O=$X/$J; mkdir -p $O
  OEX=""
  if [ $M = omni ]; then
    case $s in
      bread|wolf) ch=$(matcheck $s | tail -1); say "[재질 확인] $s $ch"
        best=$(python -c "import json;print(json.load(open('$X/matcheck_$s/choice.json'))['best'])")
        c2=$(echo $SIM | grep -oP -- '--config \K\S+')
        if [ "$best" = extra ]; then
          [ $s = bread ] && OEX="--omni_yield $(python -c "import json;print(json.load(open('$c2'))['yield_stress'])")"
          [ $s = wolf ] && OEX="--omni_fric $(python -c "import json;print(json.load(open('$c2'))['friction_angle'])")"
        fi;;
    esac
  fi
  export AF_IPG_FUSED=$FUSED
  # 판 a (시간 + 덤프). 발산하면 공식 범위 안에서 N 을 키움 (i-PG: k 20 -> 10 -> 5 -> 2 -> 1 = PG 공식 서브스텝; omni: dt 를 1/2 씩, 4 배까지)
  if [ $M = ipg ]; then CANDS=$(python -c "print(' '.join(str(max(1, round($NPG / k))) for k in (20, 10, 5, 2, 1)))"); else CANDS="$NOFF $((NOFF*2)) $((NOFF*4))"; fi
  N=""; TRIED=""
  for n in $CANDS; do
    R1=$O/N${n}_a
    if [ ! -s $R1/err.json ]; then
      say "[실행] $J N $n 판 a (덤프) fused=$FUSED $OEX"
      timeout 43200 python -u exe/pareto_mpmbl.py --method $M --nsub $n $SIM $OEX --time --dump --out $R1 > $O/N${n}_a.log 2>&1
      [ -s $R1/pred.pt ] && rclone copyto $R1/pred.pt $RB/$J/N${n}_a/pred.pt -q
    fi
    ok=$(python -c "import json;d=json.load(open('$R1/err.json'));print(1 if d['div'] is None and d['frames_done']==d['frames_total'] else 0)" 2>/dev/null || echo 0)
    say "[판 a] $J N $n ok=$ok $(tail -1 $O/N${n}_a.log | cut -c1-200)"
    TRIED="$TRIED $n:$ok"
    [ "$ok" = 1 ] && { N=$n; break; }
  done
  echo "$TRIED" > $O/tried.txt; echo "${N:-none} $NOFF $FUSED $OEX" > $O/chosen.txt
  [ -z "$N" ] && { say "[전부 발산] $J $TRIED"; continue; }
  for rep in b c; do
    [ -s $O/N${N}_$rep/err.json ] && continue
    say "[실행] $J N $N 판 $rep (시간)"
    timeout 43200 python -u exe/pareto_mpmbl.py --method $M --nsub $N $SIM $OEX --time --out $O/N${N}_$rep > $O/N${N}_$rep.log 2>&1
    say "[판 $rep] $J $(tail -1 $O/N${N}_$rep.log | cut -c1-200)"
  done
  # 렌더·물리·시각 (ffx_phys_drv.sh 의 베이스라인 작업과 같은 호출)
  P=$O/N${N}_a/pred.pt
  [ -f $O/png/$(printf %04d $FR).png ] || {
    [ -f $O/h5/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $P --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
    python -u exe/lb_render.py --sim $O/h5 --config $CF --shape $SHAPE --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1; }
  rm -rf $O/h5
  [ -s $O/phys.json ] || python -u exe/ffx_phys.py --method $M --scene $s --N $N --data $DATA --pred $P --floor $FLOOR --frames $FR --out $O/phys.json \
      --note "대표 설정 덤프 판 ($M N $N, 공식 판 a)" > $O/phys.log 2>&1
  [ -s $O/vis.json ] || python -u exe/ffx_vis.py --png $O/png --ref /root/mb/ref/$s --out $O/vis.json > $O/vis.log 2>&1
  say "[물리] $J $(tail -1 $O/phys.log | cut -c1-200)"; say "[시각] $J $(tail -1 $O/vis.log | cut -c1-200)"
  rclone copy $O $RB/$J -q --transfers 8 --exclude "h5/**"
  [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RB/$J/DONE -q
done
kill $SY 2>/dev/null
rclone copy $X $RB -q --transfers 8 --exclude "*/h5/**"
if rclone check $X $RB --one-way --size-only -q --exclude "*/h5/**" 2>/root/mb/check.txt; then
  say "[R2 확인 차이 0] -> destroy"; echo "$(date)" | rclone rcat $MK.ALL_DONE
  [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] && vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함"; echo "$(date)" | rclone rcat $MK.ALL_DONE_CHECK_FAIL
fi
