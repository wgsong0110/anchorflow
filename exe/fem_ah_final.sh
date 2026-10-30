#!/bin/bash
# 절 AH 최종 판: (1) 덤프·렌더 (비측정, 카드 1~6): 새 판 0019·0003·0015·0001·L000·ficus, 절 AF 공통 설정 L000 N267 (비교 칸) -> 3DGS 알파블렌딩 렌더
# (2) 단독 측정 GPU0 + gpu0.lock x 3 회: lego 20 + L000 + ficus (덤프 없음, 회마다 카드 1~6 이 빌 때까지 최대 AH_WAIT 초 기다림, 시작·끝 7 장 상태 기록).
# 환경: AH_FIN (새 판 플래그, REC 포함 전체), AH_FN (판 이름). 출력 /tmp/dkta/lbours/fem_ah_{lego,ficus}/<태그> (R2 같은 경로), 큰 덤프는 R2 확인 뒤 /tmp 쪽만 지움.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
W=/home/dkta/work; FN=${AH_FN:?판 이름}; FL=${AH_FIN:?플래그}; WT=${AH_WAIT:-900}
DD=/tmp/dkta/af_data; [ -f $DD/0019.pt ] || DD=$W/lbpt/test
OL=/tmp/dkta/lbours/fem_ah_lego; OF=/tmp/dkta/lbours/fem_ah_ficus; RB=r2:storage/result/anchorflow/lbours; mkdir -p $OL $OF
NL=267; NF=640
LEGO="--floor vel --node_h 0.028284 --frames 80"
FIC="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
AFR="--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 판 $FN: $FL"
lg_args() { local t=$1; if [ $t = L000 ]; then echo "--data $W/one/L000/L000.pt --tag TL000"; else echo "--data $DD/$t.pt --tag T$t"; fi; }
gtdir() { local t=$1; if [ $t = L000 ]; then echo $W/lbdata/test_low/L000; else echo $W/lbdata/test/$t; fi; }
rend() {   # <루트> <태그> <모양> <정답 디렉토리> <config> <카드>
  local O=/tmp/dkta/lbours/$1/$2 S=$3 GD=$4 CF=$5 G=$6
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $2"; return; }
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GD --out $O/h5_fem > $O/h5_fem.log 2>&1
  mkdir -p $O/vj
  [ -f $O/vj/fem_$S.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CF --shape $S --out $O/vj/fem_$S.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $O/vj/fem_$S.mp4 ] && touch $O/vj/RENDER_DONE
  rclone copy $O $RB/$1/$2 -q
  if rclone check $O $RB/$1/$2 --one-way --size-only -q 2>/dev/null; then rm -rf $O/roll_fem.pt $O/h5_fem; echo "[R2 확인·/tmp 정리] $2"; else echo "[R2 확인 실패 -- /tmp 유지] $2"; fi
}
# (1) 덤프 실행 + 렌더 (카드 1~6 동시)
dr() { local G=$1 t=$2 fl=$3 tg=$4
  EX_ROOT=fem_ah_lego bash exe/fem_ab_run.sh $tg $G $(lg_args $t) $LEGO --rbdiag --dump_f32 $fl --N $NL < /dev/null > /dev/null 2>&1
  rend fem_ah_lego $tg lego $(gtdir $t) $W/lbdata/config.json $G; }
i=0
for t in 0019 0003 0015 0001 0017 L000; do dr $(( (i % 6) + 1 )) $t "$FL" v_${FN}_${t} > $OL/vr_${FN}_$t.log 2>&1 & i=$((i + 1)); done
dr 6 L000 "$AFR" v_af_L000 > $OL/vr_af_L000.log 2>&1 &
( EX_ROOT=fem_ah_ficus bash exe/fem_ab_run.sh v_${FN}_F 5 $FIC --dump_f32 --dump_pred $FL --N $NF < /dev/null > /dev/null 2>&1
  rend fem_ah_ficus v_${FN}_F ficus $W/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt/config.json 5 ) > $OF/vr_${FN}_F.log 2>&1 &
wait
cat $OL/vr_${FN}_*.log $OL/vr_af_L000.log $OF/vr_${FN}_F.log
echo "[렌더 끝] $(date +%T)"
# (2) 단독 측정
idle() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $2+0>10{b=1} END{exit b}'; }
for r in 1 2 3; do for t in $(seq -f "%04g" 0 19) L000 F; do
  w=0; until idle || [ $w -ge $WT ]; do sleep 15; w=$((w + 15)); done
  if [ $t = F ]; then TG=ah_${FN}_F_t$r; EX_TIMED=1 EX_ROOT=fem_ah_ficus bash exe/fem_ab_run.sh $TG 0 $FIC --no_dump $FL --N $NF < /dev/null > /dev/null 2>&1; O=$OF/$TG
  else TG=ah_${FN}_${t}_t$r; EX_TIMED=1 EX_ROOT=fem_ah_lego bash exe/fem_ab_run.sh $TG 0 $(lg_args $t) $LEGO --no_dump $FL --N $NL < /dev/null > /dev/null 2>&1; O=$OL/$TG; fi
  echo "$(date +%T) $TG (대기 $w s) $(grep -a '롤아웃\]' $O/v_fem.log | cut -c1-150) | 카드 1~6 시작 $(awk -F, '$1!=0{print $2}' $O/gpu_at_start.txt | tr -d ' \n') 끝 $(awk -F, '$1!=0{print $2}' $O/gpu_at_end.txt | tr -d ' \n')"
done; done
echo "AHF_DONE $(date +%T)"; touch $OL/AHF_${FN}_DONE; rclone copy $OL/AHF_${FN}_DONE $RB/fem_ah_lego -q
