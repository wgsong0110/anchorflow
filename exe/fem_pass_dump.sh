#!/bin/bash
# 절 AH.17 기준 통과 16 씬 렌더용 덤프 + 3DGS 알파블렌딩 렌더 (비측정). 측정 회 (ah_bwg_*_c*) 는 --no_dump 라 덤프가 없어,
# 같은 명령에 --dump_f32 만 더해 커밋 a6f2e29 워크트리 ($W/wt/af_a6f2e29) 에서 다시 돌린다. 렌더 경로는 exe/fem_ah_final.sh 의 rend() 그대로.
# 사용: bash exe/fem_pass_dump.sh <GPU 목록 1,2,3>   (씬을 카드마다 차례로 나눠 카드별 동시 실행)
# 출력 /tmp/dkta/lbours/fem_pass_{lego,ficus}/pd_<씬> (R2 같은 경로), R2 확인 뒤 /tmp 의 roll_fem.pt·h5_fem 만 지움.
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; SW=$W/wt/af_a6f2e29; GL=${1:?GPU 목록}
cd $SW || exit 1
[ "$(git rev-parse --short HEAD)" = a6f2e29 ] || { echo "[오류] 워크트리 커밋 $(git rev-parse --short HEAD)"; exit 1; }
DD=/tmp/dkta/af_data; [ -f $DD/0019.pt ] || DD=$W/lbpt/test
RB=r2:storage/result/anchorflow/lbours; OL=/tmp/dkta/lbours/fem_pass_lego; OF=/tmp/dkta/lbours/fem_pass_ficus; mkdir -p $OL $OF
FL="--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.75 --vfl_w gtk --vfl_rate --vfl_npd 0.005 --vfl_band 0 --ah_fast"
LEGO="--floor vel --node_h 0.028284 --frames 80"
FIC="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
SC="0000 0002 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0019 L000 F"
echo "[시작] $(date '+%F %T') 워크트리 $(git log -1 --oneline) 카드 $GL"
lg_args() { local t=$1; if [ $t = L000 ]; then echo "--data $W/one/L000/L000.pt --tag TL000"; else echo "--data $DD/$t.pt --tag T$t"; fi; }
gtdir() { local t=$1; if [ $t = L000 ]; then echo $W/lbdata/test_low/L000; else echo $W/lbdata/test/$t; fi; }
rend() {   # <루트> <태그> <모양> <정답 디렉토리> <config> <카드>  -- exe/fem_ah_final.sh 그대로
  local O=/tmp/dkta/lbours/$1/$2 S=$3 GD=$4 CF=$5 G=$6
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $2"; return; }
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GD --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj
  [ -f $O/vj/fem_$S.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CF --shape $S --out $O/vj/fem_$S.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $O/vj/fem_$S.mp4 ] && touch $O/vj/RENDER_DONE
  rclone copy $O $RB/$1/$2 -q
  if rclone check $O $RB/$1/$2 --one-way --size-only -q 2>/dev/null; then rm -rf $O/roll_fem.pt $O/h5_fem; echo "[R2 확인·/tmp 정리] $2"; else echo "[R2 확인 실패 -- /tmp 유지] $2"; fi
}
one() { local G=$1 t=$2
  if [ $t = F ]; then
    EX_ROOT=fem_pass_ficus bash exe/fem_ab_run.sh pd_F $G $FIC --dump_f32 --N 640 $FL < /dev/null > /dev/null 2>&1
    rend fem_pass_ficus pd_F ficus $W/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt/config.json $G
  else
    EX_ROOT=fem_pass_lego bash exe/fem_ab_run.sh pd_$t $G $(lg_args $t) $LEGO --dump_f32 --N 267 $FL < /dev/null > /dev/null 2>&1
    rend fem_pass_lego pd_$t lego $(gtdir $t) $W/lbdata/config.json $G
  fi
  local O=$OL/pd_$t; [ $t = F ] && O=$OF/pd_F
  echo "$(date +%T) 카드 $G $t $(cat $O/err_fem.txt 2>/dev/null | tail -1 | cut -c1-160) 렌더 $([ -f $O/vj/RENDER_DONE ] && echo 끝 || echo 실패)"
}
IFS=, read -ra GS <<< "$GL"; n=${#GS[@]}; i=0; declare -A Q
for t in $SC; do g=${GS[$((i % n))]}; Q[$g]="${Q[$g]} $t"; i=$((i + 1)); done
for g in "${GS[@]}"; do ( for t in ${Q[$g]}; do one $g $t; done ) > $OL/pass_g$g.log 2>&1 & done
wait
cat $OL/pass_g*.log
echo "PASS_DONE $(date +%T)"; touch $OL/PASS_DONE; rclone copy $OL $RB/fem_pass_lego --include "pass_g*.log" --include PASS_DONE -q
