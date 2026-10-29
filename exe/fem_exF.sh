#!/bin/bash
# 절 L 속도만 잰 두 판 (F 덤프 없이) 을 원래 커밋·원래 명령 그대로 F 덤프만 켜서 다시 돌리고 3DGS 렌더 (사용자 10-29 영상 요청).
#   lego  s2_sub_tm_n128  (커밋 9ca13f3, sub N128 사면체 등분 질량) -> fem_ex_lego/s2_sub_tm_n128_F
#   ficus f_lag_tm_n640   (커밋 9a36512, lag N640 사면체 등분 질량) -> fem_ex_ficus/f_lag_tm_n640_F
# 실행은 원래 커밋 워크트리 ($W/wt/ex_<C>) 의 fem_ex_run.sh (속도 측정 단독 대기 포함), 렌더는 이 커밋의 lb_roll_to_h5 + lb_render (fem_ex_final.sh 와 같은 식, ψ 없음).
# 띄우기 전에 원래 cmd.txt 의 인자에서 --no_dump 만 뺀 것과 이번 인자가 같은지 대조하고, 다르면 멈춘다. 끝 표식 EXF_DONE.
cd "$(dirname "$0")/.." || exit 1
HW=$(pwd); W=/home/dkta/work
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LEGO_ARGS="--mode sub --N 128 --opt --near32 --mass tet"
FICUS_ARGS="--mode lag --N 640 --opt --mass tet --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"

one() {  # <원래 태그> <EX_ROOT> <커밋> <SHAPE> <GTD> <CFG> <인자...>
  local OT=$1 ROOT=$2 C=$3 SH=$4 GTD=$5 CFG=$6; shift 6
  local NT=${OT}_F; local O=$W/lbours/$ROOT/$NT; local R=r2:storage/result/anchorflow/lbours/$ROOT/$NT; local WT=$W/wt/ex_$C
  [ -d $WT ] || (cd $W/anchorflow && git worktree add -q --detach $WT $C) || { echo "[오류] worktree $C"; return 1; }
  [ "$(git -C $WT rev-parse --short HEAD)" = "$C" ] || { echo "[오류] $WT 커밋이 $C 아님"; return 1; }
  # 명령 대조: 원래 cmd.txt (커밋·인자, --no_dump 뺌) vs 이번
  local ORIG=$(cat $W/lbours/$ROOT/$OT/cmd.txt)
  local OC=$(echo "$ORIG" | awk '{print $3}')
  local OA=$(echo "$ORIG" | sed -E 's/.*fem_ex_sim\.py --out [^ ]+ //; s/ --no_dump//')
  echo "[대조] 원래: $ORIG"
  echo "[대조] 이번: 커밋 $C ... --out $O $*"
  [ "$OC" = "$C" ] && [ "$OA" = "$*" ] || { echo "[오류] 명령 불일치 (원래 커밋 $OC 인자 '$OA')"; return 1; }
  echo "[대조] 같음 (커밋 $C, --no_dump 만 뺌)"
  (cd $WT && EX_ROOT=$ROOT bash exe/fem_ex_run.sh $NT 0 "$@")
  [ -f $O/EX_DONE ] && [ -f $O/roll_fem.pt ] || { echo "[오류] $NT 실행 실패"; return 1; }
  local NA=$(sed -E 's/.*fem_ex_sim\.py --out [^ ]+ //' $O/cmd.txt); local NC=$(awk '{print $3}' $O/cmd.txt)
  [ "$NA" = "$OA" ] && [ "$NC" = "$C" ] && echo "[대조] 실행된 cmd.txt 도 같음" || echo "[경고] 실행된 cmd.txt 가 다름: $(cat $O/cmd.txt)"
  echo "[원래] $(grep -a '롤아웃\]' $W/lbours/$ROOT/$OT/err_fem.txt)"
  echo "[이번] $(grep -a '롤아웃\]' $O/err_fem.txt)"
  [ -f $O/h5_fem/meta.json ] || python -u $HW/exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj
  [ -f $O/vj/fem_$SH.mp4 ] || CUDA_VISIBLE_DEVICES=0 python -u $HW/exe/lb_render.py --sim $O/h5_fem --config $CFG --shape $SH --out $O/vj/fem_$SH.mp4 \
     --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $O/vj/fem_$SH.mp4 ] && touch $O/vj/RENDER_DONE || tail -3 $O/vj/r_fem.log
  ls -la $O/vj
  rclone copy $O $R -q
}
one s2_sub_tm_n128 fem_ex_lego 9ca13f3 lego $W/lbdata/test_low/L000 $W/lbdata/config.json $LEGO_ARGS
one f_lag_tm_n640 fem_ex_ficus 9a36512 ficus $W/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt/config.json $FICUS_ARGS
echo EXF_DONE
