#!/bin/bash
# [final_fix_phys] 작업 실행기 (exe/ffx_phys_boot.sh 가 코드 폴더에서 부름): 카드마다 일꾼 하나, 공용 작업 목록에서 입력이 준비된 작업을 하나씩 잡음 (mkdir 잠금).
#   작업 = 정답 MPM (장면마다) | 우리 판 (R2 lbours/final_fix/runs_nd/<장면>_<판>_r1/pred.pt, metrics.json 있어야 준비) | 베이스라인 대표 설정 덤프 판 (stab_bl_setup 끝난 뒤)
#   작업마다 /workspace/fxp/<이름>/ {phys.json, png/, mp4, vis.json, 로그, 베이스라인 pred.pt} -> R2 lbours/final_fix/phys/<이름>/ (h5·우리 pred.pt 빼고), 올린 뒤 h5·우리 pred.pt 지움 (인스턴스 디스크)
#   시각 지표는 정답 MPM 렌더 PNG 가 있어야 (같은 장면 mpm 작업 먼저).
set -u
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; PO=/dev/shm/dkta/lbours/paper_remeasure2/stab/p
R=r2:storage/result/anchorflow; RUNS=$R/lbours/final_fix/runs_nd; OX=$R/lbours/final_pa/bl; RP=$R/lbours/final_fix/phys
X=/workspace/fxp; mkdir -p $X
PY=/home/dkta/.conda/envs/af/bin/python
DEADLINE=$(( $(date +%s) + ${FX_HOURS:-10} * 3600 ))
SCN="L000 pillow2sofa bread wolf"
VARS="def k2 k4 be mid q1 nofpi q27 q64 c075 c15 c20 ss1.dump ss2.dump ss4.dump ss8.dump ss16.dump ss32.dump ss64.dump ss128.dump"
JL=$X/jobs.txt
if [ ! -s $JL ]; then
  for s in $SCN; do echo "mpm $s"; done > $JL
  for s in $SCN; do for v in def k2 k4; do echo "ours $s $v"; done; done >> $JL
  for spec in "phystwin L000 162" "phystwin pillow2sofa 143" "vrgs L000 167" "vrgs pillow2sofa 200" "springgaus L000 200" "springgaus pillow2sofa sg"; do echo "$spec"; done >> $JL
  for s in $SCN; do for v in $VARS; do case " def k2 k4 " in *" $v "*) ;; *) echo "ours $s $v";; esac; done; done >> $JL
fi

scene() {   # -> DATA GTD CF SHAPE FR FLOOR DA
  case $1 in
    L000) DATA=$SD/L000.pt; GTD=$W/lbdata/test_low/L000; CF=$W/lbdata/config.json; SHAPE=lego; FR=80; FLOOR=vel; DA="--data $DATA --tag L000 --frames 80 --floor vel";;
    pillow2sofa) DATA=$NW/pillow2sofa.pt; GTD=$NW/gt_pillow2sofa; CF=$NW/gt_pillow2sofa/config.json; SHAPE=pillow2sofa; FR=300; FLOOR=none
                 DA="--data $DATA --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp";;
    bread) DATA=$AG/bread.pt; GTD=$AG/gt_bread_full; CF=$AG/gt_bread_full/config.json; SHAPE=bread; FR=300; FLOOR=none; DA="";;
    wolf) DATA=$AG/wolf.pt; GTD=$AG/gt_wolf; CF=$AG/gt_wolf/config.json; SHAPE=wolf; FR=50; FLOOR=vel; DA="";;
  esac
}
jname() { case $1 in mpm) echo mpm_$2;; ours) echo ours_$2_$3;; *) echo $1_$2_n$3;; esac; }
ready() {   # 입력 준비 여부
  case $1 in
    mpm) return 0;;
    ours) rclone lsf $RUNS/$2_$3_r1/metrics.json 2>/dev/null | grep -q . && rclone lsf $RUNS/$2_$3_r1/pred.pt 2>/dev/null | grep -q .;;
    springgaus) [ -f /root/pr/bl.done ] || return 1; [ "$3" != sg ] && return 0
                [ -s $PO/springgaus_pillow2sofa_prep/sg_train.json ] && return 0
                rclone lsf $OX/P2/p/springgaus_pillow2sofa_prep/sg_train.json 2>/dev/null | grep -q . || return 1
                rclone copy $OX/P2/p/springgaus_pillow2sofa_prep $PO/springgaus_pillow2sofa_prep -q --transfers 8; [ -s $PO/springgaus_pillow2sofa_prep/sg_train.json ];;
    *) [ -f /root/pr/bl.done ];;
  esac
}
render() {   # $1 작업 폴더 (pred.pt 또는 정답 폴더 $2)
  local O=$1 SIM=${2:-}
  [ -f $O/png/$(printf %04d $FR).png ] && return 0
  if [ -z "$SIM" ]; then
    [ -f $O/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5 > $O/roll_h5.log 2>&1
    SIM=$O/h5
  fi
  $PY -u exe/lb_render.py --sim $SIM --config $CF --shape $SHAPE --out $O/$(basename $O).mp4 --png $O/png --frames $FR < /dev/null > $O/render.log 2>&1
}

run_job() {   # $1 종류 $2 장면 $3 판·N ; 카드 = CUDA_VISIBLE_DEVICES
  local m=$1 T=$2 v=${3:-} J O N NF
  scene $T; J=$(jname $m $T $v); O=$X/$J; mkdir -p $O
  case $m in
    mpm)
      N=$($PY -c "import json;c=json.load(open('$CF'));print(int(c['frame_dt']/c['substep_dt']))")
      [ -s $O/phys.json ] || $PY -u exe/ffx_phys.py --method mpm --scene $T --N $N --data $DATA --gt_h5 $GTD --floor $FLOOR --frames $FR --out $O/phys.json \
        --note "정답 MPM (PhysGaussian 공식 config 서브스텝 int(frame_dt/substep_dt))" > $O/phys.log 2>&1
      render $O $GTD;;
    ours)
      N=$(case $v in def) echo 1;; k2) echo 2;; k4) echo 4;; *) echo $v;; esac)
      [ -s $O/pred.pt ] || [ -s $O/phys.json -a -f $O/png/$(printf %04d $FR).png ] || rclone copyto $RUNS/${T}_${v}_r1/pred.pt $O/pred.pt -q --multi-thread-streams 8
      rclone copyto $RUNS/${T}_${v}_r1/cmd.txt $O/cmd.txt -q; rclone copyto $RUNS/${T}_${v}_r1/commit.txt $O/commit.txt -q
      [ -s $O/phys.json ] || $PY -u exe/ffx_phys.py --method ours --scene $T --N "$N" --data $DATA --pred $O/pred.pt --floor $FLOOR --frames $FR --out $O/phys.json \
        --note "우리 새 기본 판 (절 CM191) 판 $v, runs_nd/${T}_${v}_r1" > $O/phys.log 2>&1
      render $O;;
    *)
      [ -f /root/pr/bl_env.sh ] && source /root/pr/bl_env.sh
      N=$v; [ "$N" = sg ] && N=$($PY -c "import json;print(json.load(open('$PO/springgaus_pillow2sofa_prep/sg_train.json'))['n_step'])")
      if [ ! -s $O/pred.pt ]; then
        case $m in
          phystwin) PYTHONPATH=${ST_PT_PYPATH:-} $PY -u exe/stab_phystwin.py --mode run --phystwin ${ST_PT_REPO:-$W/PhysTwin} --fit $PO/phystwin_${T}_prep --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1;;
          springgaus) PYTHONPATH=${ST_SG_PYPATH:-} $PY -u exe/stab_sg.py --mode run --sg ${ST_SG_REPO:-$W/Spring-Gaus} --fit $PO/springgaus_${T}_prep --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1;;
          vrgs) local E=$([ $T = L000 ] && echo 245525.5146265931 || echo 6071.540272375929)
                $PY -u exe/pareto_vrgs.py --mode run --out $O --E $E $DA --set ${N}x11 --dump_F > $O/run.log 2>&1;;
        esac
      fi
      echo "[실행] $J $(tail -1 $O/run.log | cut -c1-200)"
      [ -s $O/pred.pt ] || { echo "[덤프 없음] $J"; return 1; }
      NF=""; [ $m != vrgs ] && NF=--no_F
      [ -s $O/phys.json ] || $PY -u exe/ffx_phys.py --method $m --scene $T --N $N --data $DATA --pred $O/pred.pt $NF --floor $FLOOR --frames $FR --out $O/phys.json \
        --note "대표 설정 덤프 판 ($m N $N)" > $O/phys.log 2>&1
      render $O;;
  esac
  echo "[물리] $J $(tail -1 $O/phys.log | cut -c1-250)"
  echo "[렌더] $J $(ls $O/png 2>/dev/null | wc -l) 장"
  if [ $m != mpm ]; then
    while [ ! -f $X/mpm_$T/DONE ] && [ ! -f $X/mpm_$T/FAILED ] && [ $(date +%s) -lt $DEADLINE ]; do sleep 30; done
  fi
  REF=""; [ $m != mpm ] && REF="--ref $X/mpm_$T/png"
  [ -s $O/vis.json ] || $PY -u exe/ffx_vis.py --png $O/png $REF --out $O/vis.json > $O/vis.log 2>&1
  echo "[시각] $J $(tail -1 $O/vis.log | cut -c1-250)"
  rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt"
  [ $m = ours ] || [ ! -f $O/pred.pt ] || rclone copyto $O/pred.pt $RP/$J/pred.pt -q
  [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RP/$J/DONE -q
  rm -rf $O/h5; [ $m = ours ] && rm -f $O/pred.pt
  return 0
}

worker() {   # $1 카드 (일꾼 번호 WID; 카드 하나에 일꾼 여럿 가능 -- 작업 대부분이 R2 전송·CPU 지표)
  export CUDA_VISIBLE_DEVICES=$1
  while :; do
    local got=0 left=0
    while read -r m s v; do
      J=$(jname $m $s $v)
      [ -f $X/$J/DONE ] || [ -f $X/$J/FAILED ] && continue
      [ -d $X/$J.claim ] && { left=1; continue; }
      left=1
      [ -f $X/$J.ready ] || { ready $m $s $v < /dev/null && touch $X/$J.ready; } || continue
      mkdir $X/$J.claim 2>/dev/null || continue
      echo "[w${WID:-$1} $(date '+%F %T')] 시작 $J"
      run_job $m $s $v < /dev/null
      [ -f $X/$J/DONE ] || { echo "[w${WID:-$1}] 실패 $J"; touch $X/$J/FAILED; rclone copy $X/$J $RP/$J -q --exclude "h5/**" --exclude "/pred.pt" < /dev/null; }
      got=1; break
    done < $JL
    [ $left = 0 ] && break
    [ $(date +%s) -gt $DEADLINE ] && { echo "[w${WID:-$1}] 기한 끝"; break; }
    [ $got = 0 ] && sleep 60
  done
  echo "[w${WID:-$1} $(date '+%F %T')] 일꾼 끝"
}

# 앞 인스턴스에서 끝난 작업 (R2 DONE) 은 건너뜀 -- bread 는 빼고 (앞 판의 bread 정답 렌더 폴더가 0 프레임뿐이었음)
rclone lsf $RP --dirs-only 2>/dev/null | while read -r d; do J=${d%/}; case $J in *_bread*|*bread_*) continue;; esac
  rclone lsf $RP/$J/DONE 2>/dev/null | grep -q . && { mkdir -p $X/$J; touch $X/$J/DONE; echo "[건너뜀 R2 DONE] $J"; }; done
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); NW_=${FX_WORKERS:-$NG}
WP=""
for w in $(seq 0 $((NW_ - 1))); do (WID=$w; worker $((w % NG))) > $X/w$w.log 2>&1 & WP="$WP $!"; sleep 5; done
wait $WP
cat $X/w*.log | grep -E "실패|덤프 없음|기한" ; while read -r m s v; do J=$(jname $m $s $v); [ -f $X/$J/DONE ] || echo "미완 $J"; done < $JL
echo "DRV_DONE"
