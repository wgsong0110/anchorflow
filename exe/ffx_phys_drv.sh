#!/bin/bash
# [final_fix_phys] 작업 실행기 (exe/ffx_phys_boot.sh 가 코드 폴더에서 부름). 작업마다 /workspace/fxp/<이름>/ {pred.pt (베이스라인), phys.json, png/, <이름>.mp4, vis.json, *.log}
#   A: 정답 MPM·우리 (바로) | B: 베이스라인 덤프 판 (stab_bl_setup 끝난 뒤, 대표 설정 = final_pa/bl pareto_bl.json·final_fix vrgs_pillow2sofa.json 그대로) | C: pyiqa 설치 뒤 시각 지표
set -u
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; PO=/dev/shm/dkta/lbours/paper_remeasure2/stab/p
R=r2:storage/result/anchorflow; RUNS=$R/lbours/final_fix/runs_nd; OX=$R/lbours/final_pa/bl
X=/workspace/fxp; mkdir -p $X
PY=/home/dkta/.conda/envs/af/bin/python
log() { echo "[drv $(date '+%F %T')] $*"; }
want() { [ -z "${FX_ONLY:-}" ] && return 0; case " $FX_ONLY " in *" $1 "*) return 0;; esac; return 1; }

scene() {   # -> DATA GTD CF SHAPE FR FLOOR DA
  case $1 in
    L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel; DA="--data $DATA --tag L000 --frames 80 --floor vel";;
    pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none
                 DA="--data $DATA --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp";;
    bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread; CF=$AG/gt_bread/config.json; SHAPE=bread; FR=300; FLOOR=none; DA="";;
    wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel; DA="";;
  esac
}
render_pred() {   # $1 = 작업 폴더 (pred.pt)
  local O=$1
  [ -f $O/png/$(printf %04d $FR).png ] && return 0
  [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
  $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $SHAPE --out $O/$(basename $O).mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
}
phys() {   # $1 이름 $2 방법 $3 N $4.. 추가 인자
  local O=$X/$1; shift; local m=$1 n=$2; shift 2
  [ -s $O/phys.json ] || $PY -u exe/ffx_phys.py --method $m --scene $T --N "$n" --data $DATA --floor $FLOOR --frames $FR --out $O/phys.json "$@" > $O/phys.log 2>&1
  log "[물리] $(basename $O) $(tail -1 $O/phys.log | cut -c1-300)"
}

phaseA() {
  for T in L000 pillow2sofa bread wolf; do
    scene $T
    NM=$($PY -c "import json;c=json.load(open('$CF'));print(int(round(c['frame_dt']/c['substep_dt'])) if 'substep_dt' in c else '?')")
    NI=$($PY -c "import json;c=json.load(open('$CF'));print(int(c['frame_dt']/c['substep_dt']) if 'substep_dt' in c else '?')")
    J=mpm_$T; O=$X/$J; mkdir -p $O
    if want $J; then
      echo "PG gs_simulation int(frame_dt/substep_dt) = $NI (round $NM), config $CF" > $O/N.txt
      phys $J mpm $NI --gt_h5 $GTD --note "정답 MPM (PhysGaussian 공식 config 서브스텝)"
      [ -f $O/png/$(printf %04d $FR).png ] || $PY -u exe/lb_render.py --sim $GTD --config $CF --shape $SHAPE --out $O/$J.mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
      log "[렌더] $J $(ls $O/png 2>/dev/null | wc -l) 장"
    fi
    for vk in def:K1 k2:K2 k4:K4; do
      v=${vk%%:*}; K=${vk##*:}; J=ours_${T}_$K; O=$X/$J; mkdir -p $O
      want $J || continue
      [ -s $O/pred.pt ] || rclone copyto $RUNS/${T}_${v}_r1/pred.pt $O/pred.pt -q
      rclone copyto $RUNS/${T}_${v}_r1/cmd.txt $O/cmd.txt -q; rclone copyto $RUNS/${T}_${v}_r1/commit.txt $O/commit.txt -q
      phys $J ours $K --pred $O/pred.pt --note "우리 새 기본 판 (절 CM191) 판 $v, runs_nd/${T}_${v}_r1"
      render_pred $O; log "[렌더] $J $(ls $O/png 2>/dev/null | wc -l) 장"
    done
  done
  touch $X/A.done
}

phaseB() {
  while [ ! -f /root/pr/bl.done ]; do sleep 30; done
  [ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
  PT=${ST_PT_REPO:-$W/PhysTwin}; SG=${ST_SG_REPO:-$W/Spring-Gaus}
  if [ ! -s $PO/springgaus_pillow2sofa_prep/sg_train.json ]; then
    rclone copy $OX/P2/p/springgaus_pillow2sofa_prep $PO/springgaus_pillow2sofa_prep -q --transfers 8
  fi
  SGN=$($PY -c "import json;print(json.load(open('$PO/springgaus_pillow2sofa_prep/sg_train.json'))['n_step'])" 2>/dev/null)
  for spec in "phystwin L000 162" "phystwin pillow2sofa 143" "vrgs L000 167" "vrgs pillow2sofa 200" "springgaus L000 200" "springgaus pillow2sofa ${SGN:-x}"; do
    set -- $spec; m=$1; T=$2; N=$3; scene $T; J=${m}_${T}_n$N; O=$X/$J
    want $J || continue
    [ "$N" = x ] && { log "[건너뜀] springgaus pillow2sofa 준비 판 sg_train.json 없음 (final_fix_sg 학습 중)"; continue; }
    mkdir -p $O
    if [ ! -s $O/pred.pt ]; then
      case $m in
        phystwin) PYTHONPATH=${ST_PT_PYPATH:-} $PY -u exe/stab_phystwin.py --mode run --phystwin $PT --fit $PO/phystwin_${T}_prep --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1;;
        springgaus) PYTHONPATH=${ST_SG_PYPATH:-} $PY -u exe/stab_sg.py --mode run --sg $SG --fit $PO/springgaus_${T}_prep --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1;;
        vrgs) E=$([ $T = L000 ] && echo 245525.5146265931 || echo 6071.540272375929)
              $PY -u exe/pareto_vrgs.py --mode run --out $O --E $E $DA --set ${N}x11 --dump_F > $O/run.log 2>&1;;
      esac
    fi
    log "[실행] $J $(tail -1 $O/run.log | cut -c1-200)"
    [ -s $O/pred.pt ] || { log "[덤프 없음] $J"; continue; }
    NF=""; [ $m != vrgs ] && NF=--no_F
    phys $J $m $N --pred $O/pred.pt $NF --note "대표 설정 덤프 판 ($m N $N)"
    render_pred $O; log "[렌더] $J $(ls $O/png 2>/dev/null | wc -l) 장"
  done
  touch $X/B.done
}

phaseA > $X/A.log 2>&1 &
PA=$!
phaseB > $X/B.log 2>&1 &
PB=$!
wait $PA $PB
cat $X/A.log $X/B.log
# C: 시각 지표
pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|opencv-python-headless|scikit-image|scipy)==") 2>&1 | tail -2
$PY -c "import pyiqa; print('[pyiqa]', pyiqa.__version__)"
for O in $X/*/; do
  J=$(basename $O); [ -d $O/png ] || continue
  want $J || continue
  T=$(echo $J | sed -E 's/^(ours|mpm|phystwin|vrgs|springgaus)_//; s/_(K[0-9]+|n[0-9]+)$//')
  REF=""; [ "${J#mpm_}" = "$J" ] && REF="--ref $X/mpm_$T/png"
  [ -s $O/vis.json ] || $PY -u exe/ffx_vis.py --png $O/png $REF --out $O/vis.json > $O/vis.log 2>&1
  log "[시각] $J $(tail -1 $O/vis.log | cut -c1-300)"
done
log "DRV_DONE"
